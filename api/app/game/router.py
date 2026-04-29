import asyncio
import json
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth.dependencies import get_current_user
from ..config import settings
from ..database import get_db
from ..dispatcher import dispatch
from ..redis_client import get_redis
from .schemas import PlayerMessage, AudioUploadResponse

router = APIRouter(tags=['game'])

LOCK_TTL = 60  # seconds


async def _assert_campaign_owner(campaign_id: str, user_id: str, db: AsyncSession) -> None:
    row = await db.execute(
        text('SELECT id FROM campaigns WHERE id = :id AND user_id = :uid'),
        {'id': campaign_id, 'uid': user_id},
    )
    if row.first() is None:
        raise HTTPException(status_code=403, detail='Access denied')


@router.post('/campaigns/{campaign_id}/audio', response_model=AudioUploadResponse)
async def upload_audio(
    campaign_id: str,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AudioUploadResponse:
    await _assert_campaign_owner(campaign_id, user['user_id'], db)

    ext = Path(file.filename or 'recording.webm').suffix or '.webm'
    filename = f'uploads/{uuid.uuid4().hex}{ext}'
    dest = Path(settings.media_root) / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await file.read())

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f'{settings.media_mcp_url}/tools/transcribe',
            json={'file_path': filename},
            headers={'X-Campaign-ID': campaign_id},
        )
        resp.raise_for_status()
        transcript = resp.json().get('text', '')

    return AudioUploadResponse(file_path=filename, transcript=transcript)


@router.post('/campaigns/{campaign_id}/message')
async def send_message(
    campaign_id: str,
    body: PlayerMessage,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, bool]:
    await _assert_campaign_owner(campaign_id, user['user_id'], db)

    redis = get_redis()
    lock_key = f'lock:campaign:{campaign_id}'
    acquired = await redis.set(lock_key, '1', ex=LOCK_TTL, nx=True)
    if not acquired:
        raise HTTPException(status_code=409, detail='Another turn is in progress')

    async def _run_and_release():
        try:
            await dispatch(campaign_id, body.content)
        except Exception as exc:
            await redis.publish(
                f'sse:campaign:{campaign_id}',
                json.dumps({'type': 'error', 'message': str(exc), 'code': 'AGENT_ERROR'}),
            )
        finally:
            await redis.delete(lock_key)

    asyncio.create_task(_run_and_release())
    return {'ok': True}


@router.get('/campaigns/{campaign_id}/stream')
async def stream(
    campaign_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    await _assert_campaign_owner(campaign_id, user['user_id'], db)
    redis = get_redis()

    async def event_generator():
        pubsub = redis.pubsub()
        await pubsub.subscribe(f'sse:campaign:{campaign_id}')
        try:
            while True:
                if await request.is_disconnected():
                    break
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=30)
                if msg:
                    yield f'data: {msg["data"]}\n\n'
                else:
                    yield 'data: {"type":"ping"}\n\n'
        finally:
            await pubsub.unsubscribe()
            await pubsub.aclose()

    return StreamingResponse(
        event_generator(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@router.get('/campaigns/{campaign_id}/turns')
async def get_turns(
    campaign_id: str,
    limit: int = 50,
    role: str | None = None,
    npc_name: str | None = None,
    play_only: bool = False,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    await _assert_campaign_owner(campaign_id, user['user_id'], db)
    async with httpx.AsyncClient(timeout=10) as client:
        body: dict[str, object] = {'limit': limit}
        if role:
            body['exclude_roles'] = [r for r in ['player', 'dm', 'npc', 'system'] if r != role]
        if npc_name:
            body['npc_name'] = npc_name
        if play_only:
            body['phase_filter'] = 'active'
        resp = await client.post(
            f'{settings.state_mcp_url}/tools/get_turns',
            json=body,
            headers={'X-Campaign-ID': campaign_id},
        )
        resp.raise_for_status()
    turns: list[dict] = resp.json().get('turns', [])
    return turns


@router.get('/media/audio/stream/{cache_key}')
async def stream_audio_proxy(
    cache_key: str,
    user: dict = Depends(get_current_user),
) -> StreamingResponse:
    async def _proxy():
        async with httpx.AsyncClient(timeout=90) as client:
            async with client.stream('GET', f'{settings.media_mcp_url}/audio/stream/{cache_key}') as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes(4096):
                    yield chunk

    return StreamingResponse(_proxy(), media_type='audio/wav')


@router.get('/media/{file_path:path}')
async def serve_media(
    file_path: str,
    user: dict = Depends(get_current_user),
) -> FileResponse:
    full_path = Path(settings.media_root) / file_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail='File not found')
    return FileResponse(str(full_path))
