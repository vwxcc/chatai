from __future__ import annotations
import hashlib
import hmac
import os
import secrets
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_DIR / "data"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", PROJECT_DIR / "uploads"))
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", PROJECT_DIR / "frontend"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "chatstudio.db"
SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-only-change-me")
MAX_NAME_LENGTH = int(os.getenv("MAX_NAME_LENGTH", "80"))
MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "30000"))
MAX_SEARCH_LENGTH = int(os.getenv("MAX_SEARCH_LENGTH", "200"))

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_db() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db

def init_db() -> None:
    with closing(get_db()) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL DEFAULT 'Новый чат',
            archived INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            user_id INTEGER,
            role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
            content TEXT NOT NULL,
            model TEXT,
            provider TEXT,
            routing_set TEXT,
            parent_message_id INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (parent_message_id) REFERENCES messages(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            stored_filename TEXT NOT NULL UNIQUE,
            mime_type TEXT,
            size INTEGER NOT NULL,
            path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS message_files (
            message_id INTEGER NOT NULL,
            file_id INTEGER NOT NULL,
            PRIMARY KEY (message_id,file_id),
            FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
            FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS providers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            base_url TEXT NOT NULL,
            api_key_env TEXT,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            model_name TEXT NOT NULL,
            temperature REAL,
            max_tokens INTEGER,
            timeout INTEGER,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(provider_id,model_name),
            FOREIGN KEY (provider_id) REFERENCES providers(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS routing_sets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS routing_set_models (
            routing_set_id INTEGER NOT NULL,
            model_config_id INTEGER NOT NULL,
            priority INTEGER NOT NULL,
            PRIMARY KEY (routing_set_id,model_config_id),
            UNIQUE(routing_set_id,priority),
            FOREIGN KEY (routing_set_id) REFERENCES routing_sets(id) ON DELETE CASCADE,
            FOREIGN KEY (model_config_id) REFERENCES model_configs(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS task_routes (
            task_key TEXT PRIMARY KEY,
            routing_set_id INTEGER,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (routing_set_id) REFERENCES routing_sets(id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chats_user_updated ON chats(user_id,updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_messages_chat_created ON messages(chat_id,created_at);
        CREATE INDEX IF NOT EXISTS idx_files_user_created ON files(user_id,created_at DESC);
        """)
        db.commit()

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return "scrypt$16384$8$1$" + salt.hex() + "$" + derived.hex()

def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm,n,r,p,salt_hex,digest_hex = stored.split("$")
        if algorithm != "scrypt":
            return False
        derived = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=int(n), r=int(r), p=int(p))
        return hmac.compare_digest(derived.hex(),digest_hex)
    except (ValueError,TypeError):
        return False

def current_user(request: Request) -> sqlite3.Row:
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(401,"Требуется авторизация")
    with closing(get_db()) as db:
        user = db.execute("SELECT id,email,name,created_at,updated_at FROM users WHERE id=?",(int(user_id),)).fetchone()
    if user is None:
        request.session.clear()
        raise HTTPException(401,"Сессия недействительна")
    return user

def get_owned_chat(db: sqlite3.Connection, chat_id: int, user_id: int) -> sqlite3.Row:
    chat = db.execute(
        "SELECT id,user_id,title,archived,created_at,updated_at FROM chats WHERE id=? AND user_id=?",
        (chat_id,user_id)
    ).fetchone()
    if chat is None:
        raise HTTPException(404,"Чат не найден")
    return chat

class RegisterRequest(BaseModel):
    name: str = Field(min_length=1,max_length=MAX_NAME_LENGTH)
    email: str = Field(min_length=3,max_length=320)
    password: str = Field(min_length=8,max_length=256)

class LoginRequest(BaseModel):
    email: str
    password: str

class CreateChatRequest(BaseModel):
    title: str = Field(default="Новый чат",min_length=1,max_length=200)

class RenameChatRequest(BaseModel):
    title: str = Field(min_length=1,max_length=200)

class MessageRequest(BaseModel):
    chat_id: int
    content: str = Field(min_length=1,max_length=MAX_PROMPT_LENGTH)
    parent_message_id: int | None = None

class RoutingSetRequest(BaseModel):
    name: str = Field(min_length=1,max_length=120)
    description: str | None = Field(default=None,max_length=500)

class ProviderRequest(BaseModel):
    name: str = Field(min_length=1,max_length=120)
    base_url: str = Field(min_length=1,max_length=500)
    api_key_env: str | None = Field(default=None,max_length=120)

class ModelConfigRequest(BaseModel):
    provider_id: int
    name: str = Field(min_length=1,max_length=120)
    model_name: str = Field(min_length=1,max_length=200)
    temperature: float | None = Field(default=None,ge=0,max=2)
    max_tokens: int | None = Field(default=None,ge=1)
    timeout: int | None = Field(default=None,ge=1)

class RoutingSetModelRequest(BaseModel):
    model_config_id: int
    priority: int = Field(ge=1)

class TaskRouteRequest(BaseModel):
    routing_set_id: int | None = None

app = FastAPI(title="ChatStudio API",version="0.1.0")
app.add_middleware(SessionMiddleware,secret_key=SESSION_SECRET,session_cookie="chatstudio_session",same_site="lax",https_only=False,max_age=60*60*24*30)
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=False,allow_methods=["*"],allow_headers=["*"])
init_db()

@app.get("/api/health")
def health():
    return {"status":"ok"}

@app.post("/api/auth/register")
def register(payload:RegisterRequest,request:Request):
    name,email=payload.name.strip(),payload.email.strip().lower()
    if not name or "@" not in email:
        raise HTTPException(400,"Некорректные данные")
    with closing(get_db()) as db:
        if db.execute("SELECT id FROM users WHERE email=?",(email,)).fetchone():
            raise HTTPException(409,"Пользователь уже существует")
        ts=now_iso()
        cur=db.execute("INSERT INTO users(email,password_hash,name,created_at,updated_at) VALUES(?,?,?,?,?)",(email,hash_password(payload.password),name,ts,ts))
        db.commit()
        user_id=int(cur.lastrowid)
    request.session["user_id"]=user_id
    return {"user":{"id":user_id,"email":email,"name":name}}

@app.post("/api/auth/login")
def login(payload:LoginRequest,request:Request):
    email=payload.email.strip().lower()
    with closing(get_db()) as db:
        user=db.execute("SELECT id,email,name,password_hash FROM users WHERE email=?",(email,)).fetchone()
    if user is None or not verify_password(payload.password,user["password_hash"]):
        raise HTTPException(401,"Неверный email или пароль")
    request.session["user_id"]=int(user["id"])
    return {"user":{"id":int(user["id"]),"email":user["email"],"name":user["name"]}}

@app.post("/api/auth/logout")
def logout(request:Request):
    request.session.clear()
    return {"ok":True}

@app.get("/api/auth/me")
def me(request:Request):
    return {"user":dict(current_user(request))}

@app.get("/api/chats")
def list_chats(request:Request):
    user=current_user(request)
    with closing(get_db()) as db:
        rows=db.execute("SELECT id,title,archived,created_at,updated_at FROM chats WHERE user_id=? ORDER BY updated_at DESC",(user["id"],)).fetchall()
    return {"chats":[dict(x) for x in rows]}

@app.post("/api/chats")
def create_chat(payload:CreateChatRequest,request:Request):
    user=current_user(request)
    title=payload.title.strip() or "Новый чат"
    ts=now_iso()
    with closing(get_db()) as db:
        cur=db.execute("INSERT INTO chats(user_id,title,created_at,updated_at) VALUES(?,?,?,?)",(user["id"],title,ts,ts))
        db.commit()
        chat=get_owned_chat(db,int(cur.lastrowid),int(user["id"]))
    return {"chat":dict(chat)}

@app.get("/api/chats/{chat_id}")
def get_chat(chat_id:int,request:Request):
    user=current_user(request)
    with closing(get_db()) as db:
        chat=get_owned_chat(db,chat_id,int(user["id"]))
        messages=db.execute(
            "SELECT id,chat_id,user_id,role,content,model,provider,routing_set,parent_message_id,created_at FROM messages WHERE chat_id=? ORDER BY created_at ASC,id ASC",
            (chat_id,)
        ).fetchall()
    return {"chat":dict(chat),"messages":[dict(x) for x in messages]}

@app.patch("/api/chats/{chat_id}")
def rename_chat(chat_id:int,payload:RenameChatRequest,request:Request):
    user=current_user(request)
    title=payload.title.strip()
    if not title:
        raise HTTPException(400,"Название не может быть пустым")
    with closing(get_db()) as db:
        get_owned_chat(db,chat_id,int(user["id"]))
        db.execute("UPDATE chats SET title=?,updated_at=? WHERE id=? AND user_id=?",(title,now_iso(),chat_id,user["id"]))
        db.commit()
        chat=get_owned_chat(db,chat_id,int(user["id"]))
    return {"chat":dict(chat)}

@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id:int,request:Request):
    user=current_user(request)
    with closing(get_db()) as db:
        get_owned_chat(db,chat_id,int(user["id"]))
        db.execute("DELETE FROM chats WHERE id=? AND user_id=?",(chat_id,user["id"]))
        db.commit()
    return {"ok":True}

@app.get("/api/search")
def search(request:Request,q:str=""):
    user=current_user(request)
    query=q.strip()[:MAX_SEARCH_LENGTH]
    if not query:
        return {"results":[]}
    like="%" + query + "%"
    with closing(get_db()) as db:
        rows=db.execute(
            "SELECT c.id AS chat_id,c.title AS chat_title,m.id AS message_id,m.role,m.content,m.created_at FROM messages m JOIN chats c ON c.id=m.chat_id WHERE c.user_id=? AND (c.title LIKE ? OR m.content LIKE ?) ORDER BY m.created_at DESC LIMIT 50",
            (user["id"],like,like)
        ).fetchall()
    return {"results":[dict(x) for x in rows]}

@app.post("/api/requests")
def create_request(payload:MessageRequest,request:Request):
    user=current_user(request)
    with closing(get_db()) as db:
        get_owned_chat(db,payload.chat_id,int(user["id"]))
        ts=now_iso()
        cur=db.execute(
            "INSERT INTO messages(chat_id,user_id,role,content,parent_message_id,created_at) VALUES(?,?,\'user\',?,?,?)",
            (payload.chat_id,user["id"],payload.content,payload.parent_message_id,ts)
        )
        db.execute("UPDATE chats SET updated_at=? WHERE id=? AND user_id=?",(ts,payload.chat_id,user["id"]))
        db.commit()
        message=db.execute("SELECT id,chat_id,user_id,role,content,model,provider,routing_set,parent_message_id,created_at FROM messages WHERE id=?",(cur.lastrowid,)).fetchone()
    return {"message":dict(message),"status":"queued"}


@app.get("/api/routing/providers")
def list_providers(request:Request):
    current_user(request)
    with closing(get_db()) as db:
        rows=db.execute("SELECT id,name,base_url,api_key_env,enabled,created_at,updated_at FROM providers ORDER BY name COLLATE NOCASE").fetchall()
    return {"providers":[dict(x) for x in rows]}

@app.post("/api/routing/providers")
def create_provider(payload:ProviderRequest,request:Request):
    current_user(request)
    ts=now_iso()
    with closing(get_db()) as db:
        try:
            cur=db.execute("INSERT INTO providers(name,base_url,api_key_env,created_at,updated_at) VALUES(?,?,?,?,?)",(payload.name.strip(),payload.base_url.strip(),payload.api_key_env,ts,ts))
            db.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(409,"Провайдер с таким названием уже существует")
        row=db.execute("SELECT id,name,base_url,api_key_env,enabled,created_at,updated_at FROM providers WHERE id=?",(cur.lastrowid,)).fetchone()
    return {"provider":dict(row)}

@app.get("/api/routing/models")
def list_models(request:Request):
    current_user(request)
    with closing(get_db()) as db:
        rows=db.execute("SELECT mc.id,mc.name,mc.model_name,mc.temperature,mc.max_tokens,mc.timeout,mc.enabled,p.id AS provider_id,p.name AS provider_name FROM model_configs mc JOIN providers p ON p.id=mc.provider_id ORDER BY p.name COLLATE NOCASE,mc.name COLLATE NOCASE").fetchall()
    return {"models":[dict(x) for x in rows]}

@app.post("/api/routing/models")
def create_model(payload:ModelConfigRequest,request:Request):
    current_user(request)
    ts=now_iso()
    with closing(get_db()) as db:
        if db.execute("SELECT id FROM providers WHERE id=?",(payload.provider_id,)).fetchone() is None:
            raise HTTPException(404,"Провайдер не найден")
        try:
            cur=db.execute("INSERT INTO model_configs(provider_id,name,model_name,temperature,max_tokens,timeout,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",(payload.provider_id,payload.name.strip(),payload.model_name.strip(),payload.temperature,payload.max_tokens,payload.timeout,ts,ts))
            db.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(409,"Такая модель уже существует у этого провайдера")
        row=db.execute("SELECT mc.id,mc.name,mc.model_name,mc.temperature,mc.max_tokens,mc.timeout,mc.enabled,p.id AS provider_id,p.name AS provider_name FROM model_configs mc JOIN providers p ON p.id=mc.provider_id WHERE mc.id=?",(cur.lastrowid,)).fetchone()
    return {"model":dict(row)}

@app.post("/api/routing/sets/{routing_set_id}/models")
def add_model_to_routing_set(routing_set_id:int,payload:RoutingSetModelRequest,request:Request):
    current_user(request)
    with closing(get_db()) as db:
        if db.execute("SELECT id FROM routing_sets WHERE id=?",(routing_set_id,)).fetchone() is None:
            raise HTTPException(404,"Набор маршрутизации не найден")
        if db.execute("SELECT id FROM model_configs WHERE id=? AND enabled=1",(payload.model_config_id,)).fetchone() is None:
            raise HTTPException(404,"Модель не найдена или отключена")
        if db.execute("SELECT 1 FROM routing_set_models WHERE routing_set_id=? AND priority=?",(routing_set_id,payload.priority)).fetchone():
            raise HTTPException(409,"Этот приоритет уже занят")
        try:
            db.execute("INSERT INTO routing_set_models(routing_set_id,model_config_id,priority) VALUES(?,?,?)",(routing_set_id,payload.model_config_id,payload.priority))
            db.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(409,"Модель уже находится в этом наборе")
    return {"ok":True}

@app.put("/api/routing/tasks/{task_key}")
def set_task_route(task_key:str,payload:TaskRouteRequest,request:Request):
    current_user(request)
    if task_key not in {"main_generation","title_generation","suggestions_generation"}:
        raise HTTPException(400,"Неизвестная AI-задача")
    with closing(get_db()) as db:
        if payload.routing_set_id is not None and db.execute("SELECT id FROM routing_sets WHERE id=?",(payload.routing_set_id,)).fetchone() is None:
            raise HTTPException(404,"Набор маршрутизации не найден")
        db.execute("INSERT INTO task_routes(task_key,routing_set_id,updated_at) VALUES(?,?,?) ON CONFLICT(task_key) DO UPDATE SET routing_set_id=excluded.routing_set_id,updated_at=excluded.updated_at",(task_key,payload.routing_set_id,now_iso()))
        db.commit()
    return {"ok":True,"task_key":task_key,"routing_set_id":payload.routing_set_id}

@app.get("/api/routing/sets")
def list_routing_sets(request:Request):
    current_user(request)
    with closing(get_db()) as db:
        sets=db.execute("SELECT id,name,description,created_at,updated_at FROM routing_sets ORDER BY name COLLATE NOCASE").fetchall()
        result=[]
        for rs in sets:
            models=db.execute(
                "SELECT rsm.priority,mc.id AS model_id,mc.name AS model_name,mc.model_name AS provider_model,p.id AS provider_id,p.name AS provider_name FROM routing_set_models rsm JOIN model_configs mc ON mc.id=rsm.model_config_id JOIN providers p ON p.id=mc.provider_id WHERE rsm.routing_set_id=? ORDER BY rsm.priority",
                (rs["id"],)
            ).fetchall()
            result.append({**dict(rs),"models":[dict(x) for x in models]})
    return {"routing_sets":result}

@app.post("/api/routing/sets")
def create_routing_set(payload:RoutingSetRequest,request:Request):
    current_user(request)
    name=payload.name.strip()
    with closing(get_db()) as db:
        try:
            ts=now_iso()
            cur=db.execute("INSERT INTO routing_sets(name,description,created_at,updated_at) VALUES(?,?,?,?)",(name,payload.description,ts,ts))
            db.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(409,"Набор с таким названием уже существует")
        row=db.execute("SELECT id,name,description,created_at,updated_at FROM routing_sets WHERE id=?",(cur.lastrowid,)).fetchone()
    return {"routing_set":dict(row)}

@app.get("/api/routing/tasks")
def list_task_routes(request:Request):
    current_user(request)
    with closing(get_db()) as db:
        rows=db.execute("SELECT tr.task_key,tr.routing_set_id,rs.name AS routing_set_name,tr.updated_at FROM task_routes tr LEFT JOIN routing_sets rs ON rs.id=tr.routing_set_id ORDER BY tr.task_key").fetchall()
    return {"tasks":[dict(x) for x in rows]}

@app.get("/api/files")
def list_files(request:Request):
    user=current_user(request)
    with closing(get_db()) as db:
        rows=db.execute("SELECT id,filename,mime_type,size,created_at FROM files WHERE user_id=? ORDER BY created_at DESC",(user["id"],)).fetchall()
    return {"files":[dict(x) for x in rows]}

@app.get("/")
def frontend_root():
    index=FRONTEND_DIR/"index.html"
    if not index.exists():
        raise HTTPException(404,"Frontend not found")
    return FileResponse(index)

@app.get("/{path:path}")
def frontend_files(path:str):
    candidate=FRONTEND_DIR/path
    if candidate.is_file():
        return FileResponse(candidate)
    index=FRONTEND_DIR/"index.html"
    if index.exists():
        return FileResponse(index)
    raise HTTPException(404,"Not found")
