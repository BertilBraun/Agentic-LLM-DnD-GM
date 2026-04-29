import json

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..schemas import GetMemoryOut, UpdateMemoryIn, OkOut

router = APIRouter()


@router.post('/tools/get_memory', response_model=GetMemoryOut)
async def get_memory(
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> GetMemoryOut:
    campaign_id: str = request.state.campaign_id
    row = await db.execute(
        text('SELECT short_term_memory, long_term_memory FROM campaigns WHERE id = :campaign_id'),
        {'campaign_id': campaign_id},
    )
    r = row.mappings().first()
    if r is None:
        raise HTTPException(status_code=404, detail='Campaign not found')
    stm = r['short_term_memory']
    if isinstance(stm, str):
        stm = json.loads(stm)
    return GetMemoryOut(short_term=stm or [], long_term=r['long_term_memory'] or '')


@router.post('/tools/update_memory', response_model=OkOut)
async def update_memory(
    body: UpdateMemoryIn,
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> OkOut:
    campaign_id: str = request.state.campaign_id
    await db.execute(
        text("""
            UPDATE campaigns SET
                short_term_memory = CAST(:short_term AS jsonb),
                long_term_memory = :long_term,
                updated_at = now()
            WHERE id = :campaign_id
        """),
        {
            'campaign_id': campaign_id,
            'short_term': json.dumps(body.short_term),
            'long_term': body.long_term,
        },
    )
    await db.commit()
    return OkOut()
