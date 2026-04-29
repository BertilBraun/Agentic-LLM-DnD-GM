import json
from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..schemas import (
    SaveNPCIn,
    SaveNPCOut,
    GetNPCIn,
    GetNPCOut,
    ListNPCsOut,
    SetActiveNPCIn,
    OkOut,
    ActiveNPCStateOut,
)

router = APIRouter()


@router.post('/tools/save_npc', response_model=SaveNPCOut)
async def save_npc(
    body: SaveNPCIn,
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> SaveNPCOut:
    campaign_id: str = request.state.campaign_id
    n = body.npc_json
    row = await db.execute(
        text("""
            INSERT INTO npcs
                (campaign_id, name, role, visual_description, voice_id, voice_instructions, portrait_path)
            VALUES
                (:campaign_id, :name, :role, :visual_description, :voice_id, :voice_instructions, :portrait_path)
            ON CONFLICT (campaign_id, name) DO UPDATE SET
                role = EXCLUDED.role,
                visual_description = EXCLUDED.visual_description,
                voice_id = EXCLUDED.voice_id,
                voice_instructions = EXCLUDED.voice_instructions,
                portrait_path = COALESCE(EXCLUDED.portrait_path, npcs.portrait_path)
            RETURNING id
        """),
        {
            'campaign_id': campaign_id,
            'name': n.get('name', ''),
            'role': n.get('role', ''),
            'visual_description': n.get('visual_description', ''),
            'voice_id': n.get('voice_id', 'ash'),
            'voice_instructions': n.get('voice_instructions', ''),
            'portrait_path': body.portrait_path,
        },
    )
    npc_id = str(row.scalar_one())
    await db.commit()
    return SaveNPCOut(npc_id=npc_id)


@router.post('/tools/get_npc', response_model=GetNPCOut)
async def get_npc(
    body: GetNPCIn,
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> GetNPCOut:
    campaign_id: str = request.state.campaign_id
    if body.npc_id:
        row = await db.execute(
            text('SELECT * FROM npcs WHERE id = :npc_id AND campaign_id = :campaign_id'),
            {'npc_id': body.npc_id, 'campaign_id': campaign_id},
        )
    elif body.name:
        row = await db.execute(
            text('SELECT * FROM npcs WHERE name = :name AND campaign_id = :campaign_id'),
            {'name': body.name, 'campaign_id': campaign_id},
        )
    else:
        return GetNPCOut(npc=None)

    r = row.mappings().first()
    if r is None:
        return GetNPCOut(npc=None)
    return GetNPCOut(npc=_npc_row(r))


@router.post('/tools/list_npcs', response_model=ListNPCsOut)
async def list_npcs(
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> ListNPCsOut:
    campaign_id: str = request.state.campaign_id
    rows = await db.execute(
        text('SELECT * FROM npcs WHERE campaign_id = :campaign_id ORDER BY created_at'),
        {'campaign_id': campaign_id},
    )
    return ListNPCsOut(npcs=[_npc_row(r) for r in rows.mappings()])


@router.post('/tools/set_active_npc', response_model=OkOut)
async def set_active_npc(
    body: SetActiveNPCIn,
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> OkOut:
    campaign_id: str = request.state.campaign_id
    await db.execute(
        text("""
            UPDATE campaigns SET
                active_npc_id = :npc_id,
                active_npc_briefing = CAST(:briefing AS jsonb),
                active_npc_conv_start = :conv_start_turn_id,
                updated_at = now()
            WHERE id = :campaign_id
        """),
        {
            'campaign_id': campaign_id,
            'npc_id': body.npc_id,
            'briefing': json.dumps(body.briefing),
            'conv_start_turn_id': body.conv_start_turn_id,
        },
    )
    await db.commit()
    return OkOut()


@router.post('/tools/get_active_npc_state', response_model=ActiveNPCStateOut)
async def get_active_npc_state(
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> ActiveNPCStateOut:
    campaign_id: str = request.state.campaign_id
    row = await db.execute(
        text('SELECT active_npc_id, active_npc_briefing, active_npc_conv_start FROM campaigns WHERE id = :campaign_id'),
        {'campaign_id': campaign_id},
    )
    r = row.mappings().first()
    if r is None or r['active_npc_id'] is None:
        return ActiveNPCStateOut(npc_id=None)
    return ActiveNPCStateOut(
        npc_id=str(r['active_npc_id']),
        briefing=r['active_npc_briefing'],
        conv_start_turn_id=str(r['active_npc_conv_start']) if r['active_npc_conv_start'] else None,
    )


@router.post('/tools/clear_active_npc', response_model=OkOut)
async def clear_active_npc(
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> OkOut:
    campaign_id: str = request.state.campaign_id
    await db.execute(
        text("""
            UPDATE campaigns SET
                active_npc_id = NULL,
                active_npc_briefing = NULL,
                active_npc_conv_start = NULL,
                updated_at = now()
            WHERE id = :campaign_id
        """),
        {'campaign_id': campaign_id},
    )
    await db.commit()
    return OkOut()


def _npc_row(r: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': str(r['id']),
        'campaign_id': str(r['campaign_id']),
        'name': r['name'],
        'role': r['role'],
        'visual_description': r['visual_description'],
        'voice_id': r['voice_id'],
        'voice_instructions': r['voice_instructions'],
        'portrait_path': r['portrait_path'],
        'created_at': r['created_at'].isoformat() if r['created_at'] else None,
    }
