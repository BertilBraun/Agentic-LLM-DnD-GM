import hashlib
import json
import os
from typing import Optional

import redis.asyncio as aioredis
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import text

from .providers import get_provider

import logging

logger = logging.getLogger(__name__)


def _langfuse_enabled() -> bool:
    pk = os.environ.get('LANGFUSE_PUBLIC_KEY')
    sk = os.environ.get('LANGFUSE_SECRET_KEY')
    return pk is not None and sk is not None and pk != 'pk-lf-...' and sk != 'sk-lf-...'


app = FastAPI(title='llm-service')

# Initialise Langfuse once at module load so the SDK registers its OTel exporter
# before any request arrives. Lazy get_client() per-request misses the startup window.
_langfuse = None
if _langfuse_enabled():
    from langfuse import get_client as _lf_get_client
    _langfuse = _lf_get_client()

REDIS_URL = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
DATABASE_URL = os.environ.get('DATABASE_URL', '')
CACHE_TTL = 3600

_redis: aioredis.Redis | None = None
_db_engine = None
_db_session = None


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis


def get_db_session() -> async_sessionmaker | None:
    global _db_engine, _db_session
    if _db_session is None and DATABASE_URL:
        _db_engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
        _db_session = async_sessionmaker(_db_engine, expire_on_commit=False)
    return _db_session


class GenerateRequest(BaseModel):
    messages: list[dict]
    response_format: Optional[str] = 'text'  # text | json
    response_json_schema: Optional[dict] = None
    user_id: Optional[str] = None
    cache: Optional[bool] = True


class GenerateResponse(BaseModel):
    text: str
    tokens_in: int
    tokens_out: int
    cached: bool


@app.post('/generate', response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    provider = get_provider()
    cache_key = None
    if req.cache:
        raw = json.dumps(req.messages, sort_keys=True) + (req.response_format or 'text')
        if req.response_json_schema:
            raw += json.dumps(req.response_json_schema, sort_keys=True)
        cache_key = f'llm:{hashlib.sha256(raw.encode()).hexdigest()}'
        cached = await get_redis().get(cache_key)
        if cached:
            data = json.loads(cached)
            return GenerateResponse(cached=True, **data)

    if _langfuse is not None:
        model_name = os.environ.get('GEMINI_MODEL', os.environ.get('OPENAI_MODEL', os.environ.get('ANTHROPIC_MODEL', 'unknown')))
        with _langfuse.start_as_current_generation(
            name='llm-generate',
            model=model_name,
            input=req.messages,
            metadata={'response_format': req.response_format, 'user_id': req.user_id},
        ) as generation:
            text_out, tokens_in, tokens_out = await provider.generate(
                req.messages, req.response_format or 'text', req.response_json_schema
            )
            generation.update(
                output=text_out,
                usage={'input': tokens_in, 'output': tokens_out, 'unit': 'TOKENS'},
            )
    else:
        text_out, tokens_in, tokens_out = await provider.generate(
            req.messages, req.response_format or 'text', req.response_json_schema
        )

    if cache_key:
        payload = json.dumps({'text': text_out, 'tokens_in': tokens_in, 'tokens_out': tokens_out})
        await get_redis().setex(cache_key, CACHE_TTL, payload)

    await _log_usage(req.user_id, tokens_in, tokens_out, cached=False)

    return GenerateResponse(text=text_out, tokens_in=tokens_in, tokens_out=tokens_out, cached=False)


async def _log_usage(user_id: Optional[str], tokens_in: int, tokens_out: int, cached: bool) -> None:
    session_factory = get_db_session()
    if session_factory is None:
        return
    provider_name = os.environ.get('LLM_PROVIDER', 'gemini')
    model_name = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')
    try:
        async with session_factory() as session:
            await session.execute(
                text(
                    'INSERT INTO llm_usage (user_id, provider, model, tokens_in, tokens_out, cached) '
                    'VALUES (:user_id, :provider, :model, :tokens_in, :tokens_out, :cached)'
                ),
                {
                    'user_id': user_id,
                    'provider': provider_name,
                    'model': model_name,
                    'tokens_in': tokens_in,
                    'tokens_out': tokens_out,
                    'cached': cached,
                },
            )
            await session.commit()
    except Exception:
        logger.warning('Usage logging failed (non-critical)', exc_info=True)


@app.get('/health')
async def health() -> dict:
    return {
        'ok': True,
        'provider': os.environ.get('LLM_PROVIDER', 'gemini'),
        'model': os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash'),
    }
