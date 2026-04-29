import asyncio
import hashlib
import json
import os
from pathlib import Path

import httpx
import redis.asyncio as aioredis
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

TTS_SERVICE_URL = os.environ.get('TTS_SERVICE_URL', 'http://tts-service:9003')
MEDIA_ROOT = os.environ.get('MEDIA_ROOT', '/media')
REDIS_URL = os.environ.get('REDIS_URL', 'redis://redis:6379/1')
PARAMS_TTL = 3600  # 1 hour — only needed until first stream fetch caches the file

router = APIRouter()

_redis: aioredis.Redis | None = None
# Per-key events to deduplicate concurrent generation requests for the same audio
_generation_events: dict[str, asyncio.Event] = {}


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis


class SpeakIn(BaseModel):
    text: str
    voice_id: str
    voice_instructions: str


class SpeakOut(BaseModel):
    stream_path: str


@router.post('/tools/speak', response_model=SpeakOut)
async def speak(body: SpeakIn) -> SpeakOut:
    cache_key = hashlib.sha256(f'{body.text}:{body.voice_id}:{body.voice_instructions}'.encode()).hexdigest()

    # Store params so the stream endpoint can generate on demand
    await get_redis().setex(
        f'tts:params:{cache_key}',
        PARAMS_TTL,
        json.dumps({'text': body.text, 'voice_id': body.voice_id, 'voice_instructions': body.voice_instructions}),
    )

    return SpeakOut(stream_path=f'audio/stream/{cache_key}')


@router.get('/audio/stream/{cache_key}')
async def stream_audio(cache_key: str) -> StreamingResponse:
    dest = Path(MEDIA_ROOT) / 'audio' / f'{cache_key}.wav'

    if dest.exists():
        return StreamingResponse(
            _file_chunks(dest),
            media_type='audio/wav',
            headers={'Cache-Control': 'public, max-age=86400'},
        )

    params_json = await get_redis().get(f'tts:params:{cache_key}')
    if not params_json:
        raise HTTPException(status_code=404, detail='Audio not found')
    params = json.loads(params_json)

    # If another request is already generating this key, wait then serve from disk
    if cache_key in _generation_events:
        await _generation_events[cache_key].wait()
        if dest.exists():
            return StreamingResponse(_file_chunks(dest), media_type='audio/wav')
        raise HTTPException(status_code=500, detail='Audio generation failed')

    event = asyncio.Event()
    _generation_events[cache_key] = event

    async def _stream_and_save():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix('.tmp')
        f = open(tmp, 'wb')
        try:
            async with httpx.AsyncClient(timeout=90) as client:
                async with client.stream('POST', f'{TTS_SERVICE_URL}/speak/stream', json=params) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes(4096):
                        f.write(chunk)
                        yield chunk
            f.close()
            tmp.rename(dest)
        except Exception:
            f.close()
            tmp.unlink(missing_ok=True)
            raise
        finally:
            event.set()
            _generation_events.pop(cache_key, None)

    return StreamingResponse(_stream_and_save(), media_type='audio/wav')


async def _file_chunks(path: Path):
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            yield chunk
