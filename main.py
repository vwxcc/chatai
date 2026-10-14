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

# Main LLM API: OpenAI-compatible by default, with custom JSON mode for other providers.
AI_KEY = os.getenv("AI_API_KEY", "").strip()
AI_BASE = os.getenv("AI_API_BASE_URL", "").strip().rstrip("/")
AI_PATH = os.getenv("AI_API_PATH", "/chat/completions").strip()
AI_MODE = os.getenv("AI_API_MODE", "openai_chat").strip().lower()
AI_MODEL = os.getenv("AI_MODEL", "").strip()
AI_MODELS = [x.strip() for x in os.getenv("AI_FALLBACK_MODELS", "").split(",") if x.strip()]
AI_AUTH_HEADER = os.getenv("AI_AUTH_HEADER", "Authorization").strip()
AI_AUTH_PREFIX = os.getenv("AI_AUTH_PREFIX", "Bearer ").replace("\\t", "\t")
AI_EXTRA_HEADERS = os.getenv("AI_EXTRA_HEADERS_JSON", "{}").strip()
AI_REQUEST_TEMPLATE = os.getenv("AI_REQUEST_TEMPLATE_JSON", "").strip()
AI_RESPONSE_PATH = os.getenv("AI_RESPONSE_TEXT_PATH", "choices.0.message.content").strip()
AI_TIMEOUT = float(os.getenv("AI_TIMEOUT", "180"))
AI_TOKENS = int(os.getenv("AI_MAX_TOKENS", "7000"))
AI_TEMPERATURE = float(os.getenv("AI_TEMPERATURE", "0.3"))
MAX_PROMPT = int(os.getenv("MAX_PROMPT_CHARS", "30000"))
MAX_IMAGE = int(os.getenv("MAX_IMAGE_BYTES", str(12 * 1024 * 1024)))
HISTORY_LIMIT = int(os.getenv("HISTORY_MESSAGES", "16"))

# Voice API is intentionally independent: its base URL, key, auth, request and response can differ.
TTS_KEY = os.getenv("TTS_API_KEY", "").strip() or AI_KEY
TTS_BASE = os.getenv("TTS_API_BASE_URL", "").strip().rstrip("/")
TTS_PATH = os.getenv("TTS_API_PATH", "/audio/speech").strip()
TTS_MODE = os.getenv("TTS_API_MODE", "openai_speech").strip().lower()
TTS_MODEL = os.getenv("TTS_MODEL", "").strip()
TTS_MODELS = [x.strip() for x in os.getenv("TTS_FALLBACK_MODELS", "").split(",") if x.strip()]
TTS_AUTH_HEADER = os.getenv("TTS_AUTH_HEADER", "Authorization").strip()
TTS_AUTH_PREFIX = os.getenv("TTS_AUTH_PREFIX", "Bearer ").replace("\\t", "\t")
TTS_EXTRA_HEADERS = os.getenv("TTS_EXTRA_HEADERS_JSON", "{}").strip()
TTS_REQUEST_TEMPLATE = os.getenv("TTS_REQUEST_TEMPLATE_JSON", "").strip()
TTS_RESPONSE_MODE = os.getenv("TTS_RESPONSE_MODE", "raw_audio").strip().lower()
TTS_RESPONSE_AUDIO_PATH = os.getenv("TTS_RESPONSE_AUDIO_PATH", "audio").strip()
TTS_RESPONSE_URL_PATH = os.getenv("TTS_RESPONSE_URL_PATH", "audio_url").strip()
TTS_VOICE_FIELD = os.getenv("TTS_VOICE_FIELD", "voice").strip()
TTS_SPEED_FIELD = os.getenv("TTS_SPEED_FIELD", "speed").strip()
TTS_FORMAT_FIELD = os.getenv("TTS_FORMAT_FIELD", "response_format").strip()
TTS_AUDIO_FORMAT = os.getenv("TTS_AUDIO_FORMAT", "mp3").strip()
DEFAULT_VOICE = os.getenv("TTS_DEFAULT_VOICE", "").strip()
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
webhook_tasks: set[asyncio.Task] = set()
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


def _endpoint(base: str, path: str) -> str:
    path = (path or "").strip()
    if path.startswith(("http://", "https://")):
        return path
    if not base:
        raise RuntimeError("Не задан API Base URL.")
    if not path:
        return base
    suffix = "/" + path.lstrip("/")
    if base.endswith(suffix):
        return base
    return base + suffix


def _api_headers(key: str, auth_header: str, auth_prefix: str, extra_json: str) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if key and auth_header:
        headers[auth_header] = f"{auth_prefix}{key}"
    if extra_json:
        try:
            extra = json.loads(extra_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Некорректный JSON в дополнительных заголовках API: {exc}") from exc
        if not isinstance(extra, dict):
            raise RuntimeError("Дополнительные заголовки API должны быть JSON-объектом.")
        headers.update({str(k): str(v) for k, v in extra.items()})
    return headers


def _template_value(value: Any, variables: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {k: _template_value(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [_template_value(v, variables) for v in value]
    if isinstance(value, str):
        token = re.fullmatch(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}", value)
        if token and token.group(1) in variables:
            return variables[token.group(1)]
        for name, replacement in variables.items():
            if isinstance(replacement, (str, int, float, bool)):
                value = value.replace("{{" + name + "}}", str(replacement))
        return value
    return value


def _build_template(template_text: str, variables: dict[str, Any]) -> dict[str, Any]:
    if not template_text:
        raise RuntimeError("Режим custom_json требует JSON-шаблон тела запроса.")
    try:
        template = json.loads(template_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Некорректный JSON-шаблон API: {exc}") from exc
    if not isinstance(template, dict):
        raise RuntimeError("Шаблон API должен быть JSON-объектом.")
    result = _template_value(template, variables)
    if not isinstance(result, dict):
        raise RuntimeError("После подстановки шаблона тело API должно быть объектом.")
    return result


def _response_path(value: Any, path: str) -> Any:
    current = value
    if not path:
        return current
    for part in path.split("."):
        if isinstance(current, list) and part.isdigit():
            idx = int(part)
            if idx >= len(current):
                raise RuntimeError(f"Путь ответа '{path}' выходит за пределы массива.")
            current = current[idx]
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise RuntimeError(f"В JSON-ответе API отсутствует поле '{part}' (путь: {path}).")
    return current


async def call_ai(prompt: str, history: list[dict], image: tuple[str, str] | None):
    if not AI_KEY:
        raise RuntimeError("Не задан AI_API_KEY.")
    if not AI_BASE:
        raise RuntimeError("Не задан AI_API_BASE_URL.")
    if not AI_MODEL:
        raise RuntimeError("Не задан AI_MODEL — укажи точное имя модели провайдера.")
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
    conversation = [*history, {"role": "user", "content": user_content}]
    provider_conversation = [*history]
    if image:
        mime, encoded = image
        provider_content: Any = [
            {"type": "text", "text": prompt or "Реши задачу на изображении."},
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": encoded}},
        ]
    else:
        provider_content = prompt
    provider_conversation.append({"role": "user", "content": provider_content})
    last_error = "неизвестная ошибка"
    endpoint = _endpoint(AI_BASE, AI_PATH)
    headers = _api_headers(AI_KEY, AI_AUTH_HEADER, AI_AUTH_PREFIX, AI_EXTRA_HEADERS)

    async with ai_lock:
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            for model in [AI_MODEL, *AI_MODELS]:
                variables = {
                    "model": model,
                    "messages": messages,
                    "conversation": conversation,
                    "provider_conversation": provider_conversation,
                    "history": history,
                    "prompt": prompt,
                    "user_content": user_content,
                    "system_prompt": SYSTEM_PROMPT,
                    "max_tokens": AI_TOKENS,
                    "temperature": AI_TEMPERATURE,
                    "image": ({"mime_type": image[0], "base64": image[1]} if image else None),
                    "image_mime_type": image[0] if image else "",
                    "image_base64": image[1] if image else "",
                }
                if AI_MODE == "custom_json":
                    body = _build_template(AI_REQUEST_TEMPLATE, variables)
                elif AI_MODE == "openai_chat":
                    body = {
                        "model": model,
                        "messages": messages,
                        "temperature": AI_TEMPERATURE,
                        "max_tokens": AI_TOKENS,
                        "response_format": {"type": "json_object"},
                    }
                else:
                    raise RuntimeError(
                        f"Неизвестный AI_API_MODE='{AI_MODE}'. Используй openai_chat или custom_json."
                    )
                try:
                    response = await client.post(endpoint, headers=headers, json=body)
                    if response.status_code in {400, 422} and AI_MODE == "openai_chat":
                        # Some compatible APIs do not support the JSON response_format.
                        retry_body = dict(body)
                        retry_body.pop("response_format", None)
                        response = await client.post(endpoint, headers=headers, json=retry_body)
                    response.raise_for_status()
                    payload = response.json()
                    raw = _response_path(payload, AI_RESPONSE_PATH)
                    if isinstance(raw, list):
                        raw = "".join(str(item.get("text", "")) for item in raw if isinstance(item, dict))
                    if isinstance(raw, dict):
                        raw = json.dumps(raw, ensure_ascii=False)
                    if not str(raw).strip():
                        raise RuntimeError("Модель вернула пустой ответ.")
                    answer, speech = parse_ai_result(str(raw))
                    return answer, speech, model
                except Exception as exc:
                    last_error = f"{model}: {exc}"
                    log.warning("AI request failed (%s): %s", model, exc)
    raise RuntimeError(f"Все модели ИИ завершились ошибкой. Последняя ошибка: {last_error[:700]}")


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
        raise RuntimeError("Не задан TTS_API_KEY (или AI_API_KEY для общего ключа).")
    if not TTS_BASE:
        raise RuntimeError("Не задан TTS_API_BASE_URL.")
    endpoint = _endpoint(TTS_BASE, TTS_PATH)
    headers = _api_headers(TTS_KEY, TTS_AUTH_HEADER, TTS_AUTH_PREFIX, TTS_EXTRA_HEADERS)
    last_error = "неизвестная ошибка TTS"

    async with httpx.AsyncClient(timeout=TTS_TIMEOUT, follow_redirects=True) as client:
        for model in [TTS_MODEL, *TTS_MODELS]:
            variables = {
                "model": model,
                "input": text,
                "text": text,
                "voice": voice,
                "speed": speed,
                "response_format": TTS_AUDIO_FORMAT,
                "output_format": TTS_AUDIO_FORMAT,
            }
            try:
                if TTS_MODE == "custom_json":
                    body = _build_template(TTS_REQUEST_TEMPLATE, variables)
                elif TTS_MODE == "openai_speech":
                    body: dict[str, Any] = {"model": model, "input": text}
                    if TTS_VOICE_FIELD and voice:
                        body[TTS_VOICE_FIELD] = voice
                    if TTS_SPEED_FIELD:
                        body[TTS_SPEED_FIELD] = speed
                    if TTS_FORMAT_FIELD:
                        body[TTS_FORMAT_FIELD] = TTS_AUDIO_FORMAT
                else:
                    raise RuntimeError(
                        f"Неизвестный TTS_API_MODE='{TTS_MODE}'. Используй openai_speech или custom_json."
                    )

                response = await client.post(endpoint, headers=headers, json=body)
                if response.status_code >= 400:
                    try:
                        detail = response.json()
                    except Exception:
                        detail = response.text[:700]
                    raise RuntimeError(f"HTTP {response.status_code}: {str(detail)[:700]}")

                response_type = response.headers.get("content-type", "").lower()
                if TTS_RESPONSE_MODE == "raw_audio":
                    if "json" in response_type:
                        try:
                            detail = response.json()
                        except Exception:
                            detail = response.text[:500]
                        raise RuntimeError(
                            "API вернул JSON вместо бинарного аудио. "
                            f"Проверь TTS_RESPONSE_MODE. Ответ: {str(detail)[:500]}"
                        )
                    if len(response.content) < 100:
                        raise RuntimeError("API вернул слишком короткий аудиофайл.")
                    return response.content, model

                if TTS_RESPONSE_MODE == "json_base64":
                    payload = response.json()
                    encoded = _response_path(payload, TTS_RESPONSE_AUDIO_PATH)
                    if not isinstance(encoded, str):
                        raise RuntimeError("Поле с base64-аудио должно быть строкой.")
                    if encoded.startswith("data:") and "," in encoded:
                        encoded = encoded.split(",", 1)[1]
                    try:
                        audio = base64.b64decode(encoded, validate=True)
                    except Exception as exc:
                        raise RuntimeError(f"Не удалось декодировать base64-аудио: {exc}") from exc
                    if len(audio) < 100:
                        raise RuntimeError("API вернуло пустой или слишком короткий аудиофайл.")
                    return audio, model

                if TTS_RESPONSE_MODE == "json_url":
                    payload = response.json()
                    audio_url = _response_path(payload, TTS_RESPONSE_URL_PATH)
                    if not isinstance(audio_url, str) or not audio_url.startswith(("https://", "http://")):
                        raise RuntimeError("JSON-поле с аудио должно содержать абсолютный HTTP(S) URL.")
                    # Never forward the TTS API key to a possibly different download host.
                    file_response = await client.get(audio_url)
                    file_response.raise_for_status()
                    audio_type = file_response.headers.get("content-type", "").lower()
                    if "json" in audio_type or len(file_response.content) < 100:
                        raise RuntimeError("Ссылка не вернула бинарный аудиофайл.")
                    return file_response.content, model

                raise RuntimeError(
                    f"Неизвестный TTS_RESPONSE_MODE='{TTS_RESPONSE_MODE}'. "
                    "Допустимо raw_audio, json_base64 или json_url."
                )
            except Exception as exc:
                last_error = f"{model or '(model omitted)'}: {exc}"
                log.warning("TTS request failed (%s): %s", model, exc)
    raise RuntimeError(f"Не удалось создать аудио. Последняя ошибка: {last_error[:700]}")


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


def is_allowed(message: Message) -> bool:
    return (message.chat.type == ChatType.PRIVATE and message.from_user is not None
            and bool(ALLOWED) and message.from_user.id in ALLOWED)


async def require_access(message: Message) -> bool:
    if is_allowed(message):
        return True
    if message.chat.type == ChatType.PRIVATE:
        uid = message.from_user.id if message.from_user else "unknown"
        if not ALLOWED:
            await message.answer(
                f"Бот пока закрыт. Твой Telegram ID: {uid}\n"
                "Добавь этот ID в переменную ALLOWED_USER_IDS в настройках сервиса.")
        else:
            await message.answer(f"Доступ закрыт. Твой Telegram ID: {uid}")
    return False


@dp.message(Command("id"))
async def cmd_id(message: Message):
    uid = message.from_user.id if message.from_user else "unknown"
    await message.answer(f"Твой Telegram ID: {uid}")


@dp.message(CommandStart())
async def cmd_start(message: Message):
    if not await require_access(message):
        return
    await message.answer(
        "Привет! Я отдельный ИИ-ответчик с озвучкой.\n\n"
        "Пришли вопрос, задачу или фото. Я отправлю ответ текстом, голосовым сообщением "
        "и отдельным MP3, который можно сохранить на телефон.\n\n"
        "/new — новый диалог\n/voice alloy — выбрать голос\n/speed 1.1 — скорость\n"
        "/settings — настройки\n/help — справка")


@dp.message(Command("help"))
async def cmd_help(message: Message):
    if not await require_access(message):
        return
    await message.answer(
        "Отправляй текст вопроса или фотографию задания.\n\n"
        "/new — очистить историю диалога\n/voice alloy — выбрать голос TTS\n"
        "/speed 1.1 — скорость от 0.5 до 2.0\n/settings — текущие настройки\n/id — Telegram ID")


@dp.message(Command("new"))
async def cmd_new(message: Message):
    if not await require_access(message):
        return
    await reset_history(message.from_user.id)
    await message.answer("Готово. История диалога очищена.")


@dp.message(Command("settings"))
async def cmd_settings(message: Message):
    if not await require_access(message):
        return
    voice, speed = await get_prefs(message.from_user.id)
    shown_voice = voice or "голос провайдера по умолчанию"
    note = "\nУ текущей Fish Audio модели нет стандартных preset-голосов." if TTS_MODEL.startswith("fish-audio/") else ""
    await message.answer(f"Голос: {shown_voice}\nСкорость: {speed:g}x{note}\nИзменить: /voice nova, /speed 1.1")


@dp.message(Command("voice"))
async def cmd_voice(message: Message):
    if not await require_access(message):
        return
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        voice, _ = await get_prefs(message.from_user.id)
        await message.answer(f"Текущий голос: {voice}\nПример: /voice nova")
        return
    voice = re.sub(r"[^A-Za-z0-9_:\-]", "", parts[1])[:80]
    if not voice:
        await message.answer("Не удалось прочитать имя голоса.")
        return
    await set_prefs(message.from_user.id, voice=voice)
    if TTS_MODEL.startswith("fish-audio/"):
        await message.answer(
            f"Настройка сохранена: {voice}. Но текущая Fish Audio модель не использует стандартные имена голосов; "
            "будет выбран голос провайдера по умолчанию. Чтобы выбирать именованные голоса, укажи совместимую TTS_MODEL.")
    else:
        await message.answer(f"Голос установлен: {voice}. Поддержка зависит от TTS-модели.")


@dp.message(Command("speed"))
async def cmd_speed(message: Message):
    if not await require_access(message):
        return
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        _, speed = await get_prefs(message.from_user.id)
        await message.answer(f"Текущая скорость: {speed:g}x\nПример: /speed 1.1")
        return
    try:
        speed = float(parts[1].replace(",", "."))
    except ValueError:
        await message.answer("Укажи число от 0.5 до 2.0.")
        return
    if not 0.5 <= speed <= 2.0:
        await message.answer("Допустимая скорость: от 0.5 до 2.0.")
        return
    await set_prefs(message.from_user.id, speed=speed)
    if TTS_MODEL.startswith(("openai/", "azure/")):
        await message.answer(f"Скорость установлена: {speed:g}x.")
    else:
        await message.answer(
            f"Скорость сохранена: {speed:g}x, но текущая TTS-модель может игнорировать скорость. "
            "У Fish Audio этот параметр не отправляется, поскольку модель его не поддерживает.")


async def archive_result(message: Message, prompt: str, answer: str,
                         voice_message: Message | None, audio_message: Message | None):
    if not ARCHIVE:
        return
    answer_text = plain_html(answer)
    record = (f"Вопрос от Telegram ID {message.from_user.id if message.from_user else 'unknown'}\n\n"
              f"ВОПРОС:\n{prompt[:1200] or '[изображение без подписи]'}\n\n"
              f"ОТВЕТ:\n{answer_text[:2400]}")
    try:
        for chunk in split_text(record):
            await message.bot.send_message(ARCHIVE, chunk, disable_web_page_preview=True)
        if voice_message:
            await message.bot.copy_message(ARCHIVE, voice_message.chat.id, voice_message.message_id)
        if audio_message:
            await message.bot.copy_message(ARCHIVE, audio_message.chat.id, audio_message.message_id)
    except Exception:
        log.exception("Failed to save answer to archive channel")
        await message.answer(
            "Ответ готов, но копирование в канал не удалось. Проверь ARCHIVE_CHANNEL_ID "
            "и право бота на публикацию.")


@dp.message(F.text | F.photo | F.document)
async def handle_question(message: Message):
    if not await require_access(message) or not message.from_user:
        return
    prompt = (message.text or message.caption or "").strip()
    if message.document and not message.photo:
        mime = message.document.mime_type or ""
        if not mime.startswith("image/"):
            await message.answer("Сейчас поддерживаю текст, фотографии и изображения как документы.")
            return
    if not prompt and not message.photo and not message.document:
        await message.answer("Пришли текст вопроса или фотографию задания.")
        return
    if len(prompt) > MAX_PROMPT:
        await message.answer(f"Сообщение слишком длинное. Лимит: {MAX_PROMPT} символов.")
        return
    try:
        image = await get_image(message)
    except Exception as exc:
        await message.answer(f"Не получилось загрузить изображение: {str(exc)[:400]}")
        return
    if image and not prompt:
        prompt = "Реши задачу на изображении. Перепиши условие, объясни решение и проверь ответ."
    if not prompt:
        prompt = "Проанализируй изображение и объясни, что на нём изображено."

    status = await message.answer("Думаю над ответом…")
    try:
        history = await get_history(message.from_user.id)
        answer, speech, model = await call_ai(prompt, history, image)
        await save_turn(message.from_user.id, prompt, plain_html(answer))
        try:
            await status.delete()
        except Exception:
            pass
        await send_answer(message, answer)

        if not TTS_KEY:
            await message.answer("Текст готов. Для аудио добавь TTS_API_KEY или AI_API_KEY.")
            await archive_result(message, prompt, answer, None, None)
            return

        audio_status = await message.answer("Готовлю голосовое и MP3…")
        voice_msg = None
        audio_msg = None
        try:
            voice, speed = await get_prefs(message.from_user.id)
            with tempfile.TemporaryDirectory(prefix="chatai-") as folder:
                mp3, ogg, tts_models = await make_audio(speech, voice, speed, folder)
                if mp3.stat().st_size > 49 * 1024 * 1024:
                    raise RuntimeError("MP3 превышает лимит Telegram примерно в 50 МБ.")
                if ogg and ogg.stat().st_size <= 49 * 1024 * 1024:
                    voice_msg = await message.answer_voice(FSInputFile(ogg, filename="answer.ogg"))
                else:
                    try:
                        voice_msg = await message.answer_voice(FSInputFile(mp3, filename="answer.mp3"))
                    except Exception as exc:
                        log.warning("MP3 voice-note fallback failed: %s", exc)
                audio_msg = await message.answer_audio(
                    FSInputFile(mp3, filename="answer.mp3"),
                    title="Ответ ИИ", performer="chatai",
                    caption="MP3 — можно скачать и сохранить на телефон.")
                log.info("AI model=%s TTS=%s user=%s", model, tts_models, message.from_user.id)
        except Exception as exc:
            log.exception("TTS failed for user %s", message.from_user.id)
            await message.answer(
                "Текстовый ответ готов, но аудио создать не удалось. Проверь TTS-модель, ключ и лимиты.\n"
                f"Ошибка: {str(exc)[:600]}")
        finally:
            try:
                await audio_status.delete()
            except Exception:
                pass
        await archive_result(message, prompt, answer, voice_msg, audio_msg)
    except Exception as exc:
        log.exception("AI request failed for user %s", message.from_user.id)
        try:
            await status.edit_text(f"Не получилось завершить запрос: {str(exc)[:800]}")
        except Exception:
            await message.answer(f"Не получилось завершить запрос: {str(exc)[:800]}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot, polling_task
    await init_db()
    if not BOT_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN is missing; bot handlers are disabled.")
        yield
        return
    bot = Bot(token=BOT_TOKEN)
    try:
        if PUBLIC_URL:
            await bot.set_webhook(
                url=f"{PUBLIC_URL}{HOOK_PATH}",
                secret_token=HOOK_SECRET,
                allowed_updates=dp.resolve_used_update_types(),
                drop_pending_updates=False,
            )
            log.info("Webhook configured: %s%s", PUBLIC_URL, HOOK_PATH)
        else:
            await bot.delete_webhook(drop_pending_updates=False)
            polling_task = asyncio.create_task(dp.start_polling(bot, close_bot_session=False))
            log.info("Bot started in polling mode.")
        yield
    finally:
        if polling_task:
            polling_task.cancel()
            try:
                await polling_task
            except asyncio.CancelledError:
                pass
            polling_task = None
        if bot:
            await bot.session.close()
            bot = None


app = FastAPI(title="chatai Telegram bot", version="1.0.0", lifespan=lifespan)


@app.get("/")
async def root():
    return {"service": "chatai", "status": "ok", "health": "/healthz"}


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "bot": "configured" if BOT_TOKEN else "token_missing"}


@app.post(HOOK_PATH)
async def telegram_webhook(request: Request):
    if bot is None:
        raise HTTPException(status_code=503, detail="Bot is not initialised")
    if not PUBLIC_URL:
        raise HTTPException(status_code=404, detail="Webhook mode is disabled")
    received = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not received or not hmac.compare_digest(received, HOOK_SECRET):
        raise HTTPException(status_code=403, detail="Invalid webhook secret")
    try:
        update = Update.model_validate(await request.json(), context={"bot": bot})
        # ACK Telegram immediately; AI/TTS can take longer than Telegram's webhook timeout.
        task = asyncio.create_task(dp.feed_update(bot, update))
        webhook_tasks.add(task)
        def finished(completed: asyncio.Task):
            webhook_tasks.discard(completed)
            try:
                completed.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                log.error("Background update failed: %s", exc,
                          exc_info=(type(exc), exc, exc.__traceback__))
        task.add_done_callback(finished)
    except Exception:
        log.exception("Telegram webhook update failed")
        raise HTTPException(status_code=500, detail="Update processing failed")
    return JSONResponse({"ok": True})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
