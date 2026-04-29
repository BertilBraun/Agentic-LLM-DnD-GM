import base64
import os
import time
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

IMAGE_SERVICE_URL = os.environ.get('IMAGE_SERVICE_URL', 'http://image-service:9002')
MEDIA_ROOT = os.environ.get('MEDIA_ROOT', '/media')

NEGATIVE_PROMPT = 'blurry, low quality, distorted, text, watermark, ugly, deformed'
PORTRAIT_NEGATIVE = f'{NEGATIVE_PROMPT}, multiple people, crowd'

router = APIRouter()


class GenerateImageIn(BaseModel):
    prompt: str
    style: str
    type: str = 'scene'  # scene | portrait


class GenerateImageOut(BaseModel):
    file_path: str


@router.post('/tools/generate_image', response_model=GenerateImageOut)
async def generate_image(body: GenerateImageIn) -> GenerateImageOut:
    neg = PORTRAIT_NEGATIVE if body.type == 'portrait' else NEGATIVE_PROMPT
    full_prompt = f'{body.style}\n\n{body.prompt}\n\nNegative: {neg}'

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f'{IMAGE_SERVICE_URL}/generate',
            json={'prompt': full_prompt, 'style': body.style, 'type': body.type},
        )
        resp.raise_for_status()
        data = resp.json()

    image_bytes = base64.b64decode(data['image_bytes'])
    ext = data.get('format', 'jpeg')
    filename = f'{int(time.time())}_{uuid.uuid4().hex[:8]}.{ext}'
    dest = Path(MEDIA_ROOT) / 'images' / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(image_bytes)
    return GenerateImageOut(file_path=f'images/{filename}')
