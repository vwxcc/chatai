from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import html
import json
import logging
import os
import re
import shutil
import tempfile
from contextlib import asynccontextmanager
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import aiosqlite
import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatType
from aiogram.filters import Command, CommandStart
from aiogram.types import FSInputFile, Message, Update
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("chatai")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ALLOWED = {int(x.strip()) for x in os.getenv("ALLOWED_USER_IDS", "").split(",")
           if x.strip().isdigit()}
ARCHIVE = os.getenv("ARCHIVE_CHANNEL_ID", "").strip() or None

AI_KEY = os.getenv("AI_API_KEY", "").strip()
AI_BASE = os.getenv("AI_API_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
AI_MODEL = os.getenv("AI_MODEL", "").strip()
AI_MODELS = [x.strip() for x in os.getenv("AI_FALLBACK_MODELS", "").split(",") if x.strip()]
AI_TIMEOUT = float(os.getenv("AI_TIMEOUT", "180"))
AI_TOKENS = int(os.getenv("AI_MAX_TOKENS", "7000"))
MAX_PROMPT = int(os.getenv("MAX_PROMPT_CHARS", "30000"))
MAX_IMAGE = int(os.getenv("MAX_IMAGE_BYTES", str(12 * 1024 * 1024)))
HISTORY_LIMIT = int(os.getenv("HISTORY_MESSAGES", "16"))

TTS_KEY = os.getenv("TTS_API_KEY", "").strip() or AI_KEY
TTS_BASE = os.getenv("TTS_API_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
TTS_MODEL = os.getenv("TTS_MODEL", "openai/gpt-4o-mini-tts-2025-12-15").strip()
TTS_MODELS = [x.strip() for x in os.getenv("TTS_FALLBACK_MODELS", "").split(",") if x.strip()]
DEFAULT_VOICE = os.getenv("TTS_DEFAULT_VOICE", "alloy").strip()
DEFAULT_SPEED = float(os.getenv("TTS_DEFAULT_SPEED", "1.0"))
TTS_CHUNK = max(600, min(int(os.getenv("TTS_CHUNK_CHARS", "2600")), 4500))
TTS_MAX = int(os.getenv("TTS_MAX_CHARS", "24000"))
TTS_TIMEOUT = float(os.getenv("TTS_TIMEOUT", "180"))

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(os.getenv("DATABASE_PATH", str(DATA_DIR / "chatai.sqlite3"))).expanduser()
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

PUBLIC_URL = (os.getenv("WEBHOOK_URL", "").strip().rstrip("/")
              or os.getenv("RENDER_EXTERNAL_URL", "").strip().rstrip("/"))
HOOK_PATH = "/telegram/webhook"
HOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()
if not HOOK_SECRET and BOT_TOKEN:
    HOOK_SECRET = hashlib.sha256(BOT_TOKEN.encode()).hexdigest()

dp = Dispatcher()
bot: Bot | None = None
polling_task: asyncio.Task | None = None
ai_lock = asyncio.Semaphore(max(1, int(os.getenv("AI_CONCURRENCY", "2"))))

SYSTEM_PROMPT = r"""
Ты — полезный русскоязычный ИИ-помощник. Решай внимательно, объясняй ход решения, проверяй вычисления. Если прислали изображение, изучи его. Не выдумывай факты и указывай, если данных не хватает.

Верни только один JSON-объект с полями "answer_html" и "speech_text".
answer_html — ответ для Telegram. Используй только поддерживаемые Telegram HTML-теги: <b>, <i>, <u>, <s>, <code>, <pre>, <a href="https://...">, <blockquote>. Не используй Markdown-заголовки, LaTeX-рендеринг и другие HTML-теги. Экранируй символы & и < в обычном тексте. Код показывай в <pre><code>...</code></pre> и не меняй исходный код. Формулы пиши читаемым текстом или Unicode-символами.
speech_text — полноценная версия ответа для естественной русской озвучки, не краткое резюме. Не читай HTML-разметку вслух.
МАТЕМАТИКА: каждую важную скобку проговаривай явно («открываем круглую скобку», «закрываем круглую скобку»; аналогично для квадратных и фигурных). Дробь читай структурно: «дробь, в числителе ..., черта дроби, в знаменателе ..., конец дроби». Степени: «икс в квадрате», «икс в третьей степени», иначе «в степени ...». Корни, модули, индексы и знаки минус не пропускай. Интеграл читай как «интеграл от ... до ...», затем всё выражение, затем обязательно «дифференциал икс» или нужной переменной. Производные, пределы, суммы и логарифмы читай явно.
ГЕОМЕТРИЯ: называй угол АВС, треугольник АВС, отрезки, параллельность, перпендикулярность и градусы; различай точки, отрезки и углы.
ФИЗИКА: произноси индексы и единицы измерения (метр в секунду, метр в секунду в квадрате, ньютон, джоуль). Векторы и направления называй словами.
КОД: читай последовательно по строкам, называй важные скобки, кавычки, двоеточия, запятые, присваивание, сравнения и границы блоков. Идентификаторы произноси различимо. Не исправляй код молча.
Обычный текст произноси естественно, не озвучивай каждую точку. Не добавляй в речь факты, которых нет в ответе, и не читай сам JSON.
"""

ALLOWED_TAGS = {"b", "strong", "i", "em", "u", "ins", "s", "strike", "del",
                "code", "pre", "a", "blockquote"}


class Sanitizer(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.stack: list[str] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "br":
            self.parts.append("\n")
            return
        if tag not in ALLOWED_TAGS:
            return
        item = f"<{tag}"
        if tag == "a":
            href = dict(attrs).get("href") or ""
            if href.startswith(("https://", "http://", "tg://")):
                item += f' href="{html.escape(href, quote=True)}"'
        self.parts.append(item + ">")
        self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag not in self.stack:
            return
        idx = len(self.stack) - 1 - self.stack[::-1].index(tag)
        for open_tag in reversed(self.stack[idx:]):
            self.parts.append(f"</{open_tag}>")
        del self.stack[idx:]

    def handle_data(self, data):
        self.parts.append(html.escape(data, quote=False))

    def finish(self):
        for tag in reversed(self.stack):
            self.parts.append(f"</{tag}>")
        self.stack.clear()
        return "".join(self.parts)


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in {"p", "div", "li", "pre", "blockquote", "br"}:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)

    def result(self):
        return re.sub(r"\n{3,}", "\n\n", "".join(self.parts)).strip()


def safe_html(value: str) -> str:
    parser = Sanitizer()
    try:
        parser.feed(value)
        parser.close()
        return parser.finish()
    except Exception:
        return html.escape(value)


def plain_html(value: str) -> str:
    parser = TextExtractor()
    try:
        parser.feed(value)
        parser.close()
        return parser.result()
    except Exception:
        return re.sub(r"<[^>]*>", "", value)


def split_text(text: str, limit: int = 3900) -> list[str]:
    text = text.strip()
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        cut = max(text.rfind("\n", 0, limit), text.rfind(" ", 0, limit))
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    return [x for x in chunks if x]


async def send_plain(message: Message, text: str):
    sent = []
    for chunk in split_text(text):
        sent.append(await message.answer(chunk, disable_web_page_preview=True))
    return sent


async def send_answer(message: Message, answer: str):
    answer = safe_html(answer).strip() or "Не удалось сформировать ответ."
    if len(answer) <= 3900:
        try:
            return [await message.answer(answer, parse_mode="HTML", disable_web_page_preview=True)]
        except Exception as exc:
            log.warning("HTML send failed; sending plain text: %s", exc)
    return await send_plain(message, plain_html(answer))


def db_connect():
    return aiosqlite.connect(DB_PATH, timeout=20)


async def init_db():
    async with db_connect() as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("""CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id,id)")
        await db.execute("""CREATE TABLE IF NOT EXISTS preferences(
            user_id INTEGER PRIMARY KEY, voice TEXT NOT NULL, speed REAL NOT NULL)""")
        await db.commit()


async def get_history(user_id: int):
    async with db_connect() as db:
        cursor = await db.execute(
            "SELECT role,content FROM messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, HISTORY_LIMIT))
        rows = await cursor.fetchall()
    return [{"role": row[0], "content": row[1]} for row in reversed(rows)]


async def save_turn(user_id: int, prompt: str, answer: str):
    async with db_connect() as db:
        await db.execute("INSERT INTO messages(user_id,role,content) VALUES(?, 'user', ?)",
                         (user_id, prompt[:MAX_PROMPT]))
        await db.execute("INSERT INTO messages(user_id,role,content) VALUES(?, 'assistant', ?)",
                         (user_id, answer[:40000]))
        await db.execute("""DELETE FROM messages WHERE user_id=? AND id NOT IN
            (SELECT id FROM messages WHERE user_id=? ORDER BY id DESC LIMIT ?)""",
                         (user_id, user_id, max(2, HISTORY_LIMIT)))
        await db.commit()


async def reset_history(user_id: int):
    async with db_connect() as db:
        await db.execute("DELETE FROM messages WHERE user_id=?", (user_id,))
        await db.commit()


async def get_prefs(user_id: int):
    async with db_connect() as db:
        cursor = await db.execute("SELECT voice,speed FROM preferences WHERE user_id=?", (user_id,))
        row = await cursor.fetchone()
    return (str(row[0]), float(row[1])) if row else (DEFAULT_VOICE, DEFAULT_SPEED)


async def set_prefs(user_id: int, voice: str | None = None, speed: float | None = None):
    old_voice, old_speed = await get_prefs(user_id)
    voice = (voice or old_voice).strip()[:80]
    speed = speed if speed is not None else old_speed
    async with db_connect() as db:
        await db.execute("""INSERT INTO preferences(user_id,voice,speed) VALUES(?,?,?)
            ON CONFLICT(user_id) DO UPDATE SET voice=excluded.voice,speed=excluded.speed""",
                         (user_id, voice, speed))
        await db.commit()
    return voice, speed


def parse_ai_result(raw: str) -> tuple[str, str]:
    text = raw.strip()
    text = re.sub(r"^\x60{3}(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*\x60{3}$", "", text)
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        left, right = text.find("{"), text.rfind("}")
        if left >= 0 and right > left:
            try:
                data = json.loads(text[left:right + 1])
            except json.JSONDecodeError:
                pass
    if isinstance(data, dict):
        answer = str(data.get("answer_html") or data.get("answer") or "").strip()
        speech = str(data.get("speech_text") or "").strip()
        if answer:
            answer = safe_html(answer)
            return answer, speech or plain_html(answer)
    return html.escape(text) or "Не удалось получить ответ.", text or "Не удалось получить ответ."


async def call_ai(prompt: str, history: list[dict], image: tuple[str, str] | None):
    if not AI_KEY:
        raise RuntimeError("Не задан AI_API_KEY.")
    if not AI_MODEL:
        raise RuntimeError("Не задан AI_MODEL — укажи slug модели.")
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
    if image:
        mime, encoded = image
        user_content: Any = [
            {"type": "text", "text": prompt or "Реши задачу на изображении."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]
    else:
        user_content = prompt
    messages.append({"role": "user", "content": user_content})
    endpoint = f"{AI_BASE}/chat/completions"
    last_error = "неизвестная ошибка"
    async with ai_lock:
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            for model in [AI_MODEL, *AI_MODELS]:
                body = {"model": model, "messages": messages, "temperature": 0.3,
                        "max_tokens": AI_TOKENS, "response_format": {"type": "json_object"}}
                headers = {"Authorization": f"Bearer {AI_KEY}", "Content-Type": "application/json"}
                try:
                    response = await client.post(endpoint, headers=headers, json=body)
                    if response.status_code in {400, 422}:
                        body.pop("response_format", None)
                        response = await client.post(endpoint, headers=headers, json=body)
                    response.raise_for_status()
                    raw = response.json()["choices"][0]["message"]["content"]
                    if isinstance(raw, list):
                        raw = "".join(str(x.get("text", "")) for x in raw if isinstance(x, dict))
                    if not str(raw).strip():
                        raise RuntimeError("Модель вернула пустой ответ")
                    answer, speech = parse_ai_result(str(raw))
                    return answer, speech, model
                except Exception as exc:
                    last_error = f"{model}: {exc}"
                    log.warning("AI model failed: %s", last_error)
    raise RuntimeError(f"Все модели ИИ завершились ошибкой: {last_error[:700]}")


def split_speech(text: str, limit: int) -> list[str]:
    text = re.sub(r"[ \t]+\n", "\n", text).strip()
    if len(text) <= limit:
        return [text] if text else []
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?;:])\s+|\n+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > limit:
            if current:
                pieces.append(current)
                current = ""
            words = sentence.split()
            for word in words:
                if len(current) + len(word) + 1 > limit:
                    if current:
                        pieces.append(current)
                    current = word
                else:
                    current = f"{current} {word}".strip()
            continue
        if len(current) + len(sentence) + 1 > limit:
            if current:
                pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


async def request_tts(text: str, voice: str, speed: float):
    if not TTS_KEY:
        raise RuntimeError("Не задан TTS_API_KEY или общий AI_API_KEY.")
    endpoint = TTS_BASE if TTS_BASE.endswith("/audio/speech") else f"{TTS_BASE}/audio/speech"
    last_error = "неизвестная ошибка TTS"
    async with httpx.AsyncClient(timeout=TTS_TIMEOUT) as client:
        for model in [TTS_MODEL, *TTS_MODELS]:
            try:
                response = await client.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {TTS_KEY}", "Content-Type": "application/json"},
                    json={"model": model, "input": text, "voice": voice,
                          "response_format": "mp3", "speed": speed},
                )
                if response.status_code >= 400:
                    try:
                        details = response.json()
                    except Exception:
                        details = response.text[:400]
                    raise RuntimeError(f"HTTP {response.status_code}: {str(details)[:400]}")
                content_type = response.headers.get("content-type", "").lower()
                if "json" in content_type or len(response.content) < 100:
                    raise RuntimeError(f"API вернуло не аудио (Content-Type: {content_type})")
                return response.content, model
            except Exception as exc:
                last_error = f"{model}: {exc}"
                log.warning("TTS model failed: %s", last_error)
    raise RuntimeError(f"Не удалось создать аудио: {last_error[:700]}")


async def ffmpeg(*args: str):
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FileNotFoundError("ffmpeg не установлен")
    proc = await asyncio.create_subprocess_exec(
        exe, "-hide_banner", "-loglevel", "error", *args,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    _, err = await proc.communicate()
    if proc.returncode:
        raise RuntimeError(err.decode("utf-8", "replace")[-800:])


async def make_audio(text: str, voice: str, speed: float, folder: str):
    if len(text) > TTS_MAX:
        raise RuntimeError(f"Сценарий на {len(text)} символов превышает TTS_MAX_CHARS={TTS_MAX}.")
    chunks = split_speech(text, TTS_CHUNK)
    if not chunks:
        raise RuntimeError("Пустой сценарий озвучивания.")
    root = Path(folder)
    paths, models = [], []
    for i, chunk in enumerate(chunks, 1):
        data, model = await request_tts(chunk, voice, speed)
        path = root / f"part_{i:03d}.mp3"
        path.write_bytes(data)
        paths.append(path)
        models.append(model)
    final_mp3 = root / "answer.mp3"
    concat = root / "concat.txt"
    concat.write_text("".join(f"file '{p.as_posix()}'\n" for p in paths), encoding="utf-8")
    try:
        await ffmpeg("-y", "-f", "concat", "-safe", "0", "-i", str(concat),
                     "-vn", "-codec:a", "libmp3lame", "-q:a", "3", str(final_mp3))
    except (FileNotFoundError, RuntimeError) as exc:
        log.warning("Could not re-encode MP3; joining chunks: %s", exc)
        final_mp3.write_bytes(b"".join(p.read_bytes() for p in paths))
    final_ogg = root / "answer.ogg"
    try:
        await ffmpeg("-y", "-i", str(final_mp3), "-vn", "-ac", "1", "-ar", "48000",
                     "-c:a", "libopus", "-b:a", "48k", str(final_ogg))
        ogg = final_ogg
    except (FileNotFoundError, RuntimeError) as exc:
        log.warning("Could not create OGG voice note: %s", exc)
        ogg = None
    return final_mp3, ogg, ", ".join(dict.fromkeys(models))


async def get_image(message: Message):
    file_id, mime = None, "image/jpeg"
    if message.photo:
        file_id = message.photo[-1].file_id
    elif message.document:
        doc = message.document
        if doc.mime_type not in {"image/jpeg", "image/png", "image/webp"}:
            return None
        if doc.file_size and doc.file_size > MAX_IMAGE:
            raise RuntimeError(f"Изображение больше лимита {MAX_IMAGE // (1024 * 1024)} МБ.")
        file_id, mime = doc.file_id, doc.mime_type or "image/jpeg"
    if not file_id:
        return None
    info = await message.bot.get_file(file_id)
    if info.file_size and info.file_size > MAX_IMAGE:
        raise RuntimeError(f"Изображение больше лимита {MAX_IMAGE // (1024 * 1024)} МБ.")
    from io import BytesIO
    buffer = BytesIO()
    await message.bot.download_file(info.file_path, destination=buffer)
    data = buffer.getvalue()
    if len(data) > MAX_IMAGE:
        raise RuntimeError(f"Изображение больше лимита {MAX_IMAGE // (1024 * 1024)} МБ.")
    return mime, base64.b64encode(data).decode("ascii")
