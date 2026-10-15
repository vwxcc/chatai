"""Telegram archive and optional OpenAI-compatible TTS for ai-ch."""
import json
import logging
import os
import secrets
import urllib.request
from typing import Any

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHANNEL_ID = os.getenv("ARCHIVE_CHANNEL_ID", "").strip()
TELEGRAM_TIMEOUT = float(os.getenv("TELEGRAM_TIMEOUT", "25"))
TTS_BASE = os.getenv("TTS_API_BASE_URL", "").strip().rstrip("/")
TTS_PATH = os.getenv("TTS_API_PATH", "/audio/speech").strip()
TTS_KEY = os.getenv("TTS_API_KEY", "").strip()
TTS_MODEL = os.getenv("TTS_MODEL", "").strip()
TTS_VOICE = os.getenv("TTS_VOICE", "").strip()
TTS_FORMAT = os.getenv("TTS_AUDIO_FORMAT", "mp3").strip() or "mp3"
TTS_TIMEOUT = float(os.getenv("TTS_TIMEOUT", "120"))
TTS_CHUNK_CHARS = max(600, min(int(os.getenv("TTS_CHUNK_CHARS", "2600")), 4000))
TTS_MAX_CHARS = max(1000, int(os.getenv("TTS_MAX_CHARS", "24000")))


def split_text(text: str, limit: int) -> list[str]:
    remaining, result = (text or "").strip(), []
    while remaining:
        if len(remaining) <= limit:
            result.append(remaining)
            break
        cut = max(remaining.rfind("\n", 0, limit), remaining.rfind(" ", 0, limit))
        if cut < limit // 2:
            cut = limit
        result.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    return [part for part in result if part]


def _telegram_json(method: str, payload: dict[str, Any]):
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{BOT_TOKEN}/{method}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=TELEGRAM_TIMEOUT) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not result.get("ok"):
        raise RuntimeError(f"Telegram {method}: {str(result)[:500]}")
    return result.get("result")


def _telegram_audio(audio: bytes, caption: str):
    if len(audio) > 49 * 1024 * 1024:
        raise RuntimeError("TTS-файл превышает лимит Telegram")
    boundary = "----aich-" + secrets.token_hex(12)
    body = bytearray()

    def field(name: str, value: str):
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")

    field("chat_id", CHANNEL_ID)
    field("caption", caption[:900])
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(b'Content-Disposition: form-data; name="audio"; filename="answer.mp3"\r\n')
    body.extend(b"Content-Type: audio/mpeg\r\n\r\n")
    body.extend(audio)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendAudio",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=TELEGRAM_TIMEOUT) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not result.get("ok"):
        raise RuntimeError(f"Telegram sendAudio: {str(result)[:500]}")


def _tts_audio(text: str) -> bytes:
    if not all((TTS_BASE, TTS_KEY, TTS_MODEL)):
        raise RuntimeError("TTS_API_BASE_URL, TTS_API_KEY и TTS_MODEL должны быть заданы")
    path = TTS_PATH or "/audio/speech"
    url = path if path.startswith(("http://", "https://")) else TTS_BASE + "/" + path.lstrip("/")
    payload = {"model": TTS_MODEL, "input": text, "response_format": TTS_FORMAT}
    if TTS_VOICE:
        payload["voice"] = TTS_VOICE
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {TTS_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=TTS_TIMEOUT) as response:
        audio = response.read()
        content_type = response.headers.get("Content-Type", "").lower()
    if "json" in content_type or len(audio) < 100:
        raise RuntimeError("TTS API не вернул бинарное аудио; проверь endpoint и модель")
    return audio


def publish_answer(prompt: str, answer: str, user_id: int, model: str):
    """Archive every completed answer. Archive errors do not fail the website response."""
    if not BOT_TOKEN or not CHANNEL_ID:
        logging.error("Ответ не опубликован: задай TELEGRAM_BOT_TOKEN и ARCHIVE_CHANNEL_ID")
        return
    question = (prompt or "[вопрос с прикреплённым файлом]").strip()
    if len(question) > 1200:
        question = question[:1200] + "…"
    record = f"ai-ch · пользователь #{user_id} · модель {model}\n\nВОПРОС:\n{question}\n\nОТВЕТ:\n{(answer or '').strip()}"
    try:
        for part in split_text(record, 3700):
            _telegram_json("sendMessage", {"chat_id": CHANNEL_ID, "text": part, "disable_web_page_preview": True})
    except Exception:
        logging.exception("Не удалось опубликовать текст ответа в Telegram")
        return
    if not all((TTS_BASE, TTS_KEY, TTS_MODEL)):
        logging.info("TTS не настроен; в Telegram опубликован только текст")
        return
    speech = (answer or "").replace("```", "")
    if len(speech) > TTS_MAX_CHARS:
        logging.warning("TTS пропущен: ответ длиннее TTS_MAX_CHARS=%s", TTS_MAX_CHARS)
        return
    chunks = split_text(speech, TTS_CHUNK_CHARS)
    try:
        for index, chunk in enumerate(chunks, 1):
            _telegram_audio(_tts_audio(chunk), f"ai-ch · озвучка ответа · часть {index}/{len(chunks)}")
    except Exception:
        logging.exception("Не удалось создать/опубликовать TTS-аудио")
