import base64
import hashlib
import io
import os
import wave
from typing import Optional

import redis.asyncio as aioredis
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="tts-service")

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/1")
CACHE_TTL = 86400  # 24 hours for audio

_redis: aioredis.Redis | None = None

# Map OpenAI voice IDs to Gemini prebuilt voices
_GEMINI_VOICE_MAP = {
    "ash": "Kore",
    "alloy": "Zephyr",
    "echo": "Charon",
    "fable": "Puck",
    "onyx": "Fenrir",
    "nova": "Leda",
    "shimmer": "Aoede",
}


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=False)
    return _redis


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


class SpeakRequest(BaseModel):
    text: str
    voice_id: str
    voice_instructions: str
    user_id: Optional[str] = None


class SpeakResponse(BaseModel):
    audio_bytes: str  # base64 WAV
    format: str = "wav"
    duration_sec: float
    cached: bool


async def _openai_tts(text: str, voice_id: str, instructions: str) -> bytes:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    response = await client.audio.speech.create(
        model=model,
        voice=voice_id,
        input=text,
        instructions=instructions,
        response_format="wav",
    )
    return response.content


async def _gemini_tts(text: str, voice_id: str, instructions: str) -> bytes:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    gemini_voice = _GEMINI_VOICE_MAP.get(voice_id, "Kore")
    prompt = f"{instructions}\n\n{text}" if instructions else text
    response = await client.aio.models.generate_content(
        model="gemini-3.1-flash-tts-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=gemini_voice)
                )
            ),
        ),
    )
    pcm = response.candidates[0].content.parts[0].inline_data.data
    return _pcm_to_wav(pcm)


@app.post("/speak", response_model=SpeakResponse)
async def speak(req: SpeakRequest) -> SpeakResponse:
    cache_key = hashlib.sha256(
        f"{req.text}:{req.voice_id}:{req.voice_instructions}".encode()
    ).hexdigest()
    redis_key = f"tts:{cache_key}"

    cached = await get_redis().get(redis_key)
    if cached:
        duration = len(cached) / (24000 * 2)
        return SpeakResponse(
            audio_bytes=base64.b64encode(cached).decode(),
            duration_sec=duration,
            cached=True,
        )

    provider = os.environ.get("TTS_PROVIDER", "openai")
    if provider == "openai":
        audio_bytes = await _openai_tts(req.text, req.voice_id, req.voice_instructions)
    elif provider == "gemini":
        audio_bytes = await _gemini_tts(req.text, req.voice_id, req.voice_instructions)
    else:
        raise NotImplementedError(f"TTS provider {provider!r} not implemented")

    await get_redis().setex(redis_key, CACHE_TTL, audio_bytes)
    duration = len(audio_bytes) / (24000 * 2)
    return SpeakResponse(
        audio_bytes=base64.b64encode(audio_bytes).decode(),
        duration_sec=duration,
        cached=False,
    )


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "provider": os.environ.get("TTS_PROVIDER", "openai"),
    }
