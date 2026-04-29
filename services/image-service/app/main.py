import base64
import hashlib
import json
import os
from typing import Optional

import redis.asyncio as aioredis
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="image-service")

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
CACHE_TTL = 3600

_redis: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=False)
    return _redis


class GenerateRequest(BaseModel):
    prompt: str
    style: str = ""
    type: str = "scene"  # scene | portrait
    user_id: Optional[str] = None
    cache: Optional[bool] = True


class GenerateResponse(BaseModel):
    image_bytes: str  # base64
    format: str = "jpeg"
    cached: bool


async def _generate_gemini(prompt: str, img_type: str) -> bytes:
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
    aspect_hint = "square portrait format" if img_type == "portrait" else "wide landscape 16:9 format"
    full_prompt = f"{prompt} -- {aspect_hint}"
    response = await client.aio.models.generate_content(
        model=model,
        contents=full_prompt,
    )
    for part in response.parts:
        if part.inline_data is not None:
            return part.inline_data.data
    raise ValueError("No image in Gemini response")


async def _generate_dalle(prompt: str, img_type: str) -> bytes:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    size = "1024x1024" if img_type == "portrait" else "1792x1024"
    response = await client.images.generate(model="dall-e-3", prompt=prompt, size=size, response_format="b64_json")
    return base64.b64decode(response.data[0].b64_json)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    cache_key = None
    if req.cache:
        raw = f"{req.prompt}:{req.type}"
        cache_key = f"img:{hashlib.sha256(raw.encode()).hexdigest()}"
        cached = await get_redis().get(cache_key)
        if cached:
            return GenerateResponse(image_bytes=base64.b64encode(cached).decode(), cached=True)

    provider = os.environ.get("IMAGE_PROVIDER", "gemini")
    if provider == "dalle":
        image_bytes = await _generate_dalle(req.prompt, req.type)
    else:
        image_bytes = await _generate_gemini(req.prompt, req.type)

    if cache_key:
        await get_redis().setex(cache_key, CACHE_TTL, image_bytes)

    return GenerateResponse(image_bytes=base64.b64encode(image_bytes).decode(), cached=False)


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "provider": os.environ.get("IMAGE_PROVIDER", "gemini"),
        "model": os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image"),
    }
