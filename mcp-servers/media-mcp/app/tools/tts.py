import base64
import hashlib
import os
from pathlib import Path

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

TTS_SERVICE_URL = os.environ.get('TTS_SERVICE_URL', 'http://tts-service:9003')
MEDIA_ROOT = os.environ.get('MEDIA_ROOT', '/media')

router = APIRouter()


class SpeakIn(BaseModel):
    text: str
    voice_id: str
    voice_instructions: str


class SpeakOut(BaseModel):
    file_path: str


@router.post('/tools/speak', response_model=SpeakOut)
async def speak(body: SpeakIn) -> SpeakOut:
    cache_key = hashlib.sha256(f'{body.text}:{body.voice_id}:{body.voice_instructions}'.encode()).hexdigest()
    filename = f'{cache_key}.wav'
    dest = Path(MEDIA_ROOT) / 'audio' / filename

    if dest.exists():
        return SpeakOut(file_path=f'audio/{filename}')

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f'{TTS_SERVICE_URL}/speak',
            json={
                'text': body.text,
                'voice_id': body.voice_id,
                'voice_instructions': body.voice_instructions,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    audio_bytes = base64.b64decode(data['audio_bytes'])
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(audio_bytes)
    return SpeakOut(file_path=f'audio/{filename}')
