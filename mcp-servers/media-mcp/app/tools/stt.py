import os
from pathlib import Path

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

STT_SERVICE_URL = os.environ.get('STT_SERVICE_URL', 'http://stt-service:9004')
MEDIA_ROOT = os.environ.get('MEDIA_ROOT', '/media')

router = APIRouter()


class TranscribeIn(BaseModel):
    file_path: str  # relative to MEDIA_ROOT


class TranscribeOut(BaseModel):
    text: str


@router.post('/tools/transcribe', response_model=TranscribeOut)
async def transcribe(body: TranscribeIn):
    import base64

    abs_path = Path(MEDIA_ROOT) / body.file_path
    audio_bytes = abs_path.read_bytes()
    ext = abs_path.suffix.lstrip('.')

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f'{STT_SERVICE_URL}/transcribe',
            json={
                'audio_bytes': base64.b64encode(audio_bytes).decode(),
                'format': ext or 'wav',
            },
        )
        resp.raise_for_status()
        data = resp.json()

    return TranscribeOut(text=data['text'])
