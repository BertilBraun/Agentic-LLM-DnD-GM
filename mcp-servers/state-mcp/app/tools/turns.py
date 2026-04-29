import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..schemas import LogTurnIn, LogTurnOut, GetTurnsIn, GetTurnsOut, RoutingStateOut

router = APIRouter()


@router.post("/tools/log_turn", response_model=LogTurnOut)
async def log_turn(
    body: LogTurnIn,
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> LogTurnOut:
    campaign_id: str = request.state.campaign_id
    metadata = body.metadata or {}
    if body.audio_path:
        metadata["audio_path"] = body.audio_path
    if body.image_path:
        metadata["image_path"] = body.image_path

    row = await db.execute(
        text("""
            INSERT INTO turns (campaign_id, role, content, npc_name, audio_path, image_path, metadata, session_phase)
            SELECT :campaign_id, CAST(:role AS turn_role), :content, :npc_name, :audio_path, :image_path,
                   CAST(:metadata AS jsonb), phase
            FROM campaigns WHERE id = :campaign_id
            RETURNING id
        """),
        {
            "campaign_id": campaign_id,
            "role": body.role,
            "content": body.content,
            "npc_name": body.npc_name,
            "audio_path": body.audio_path,
            "image_path": body.image_path,
            "metadata": json.dumps(metadata),
        },
    )
    turn_id = str(row.scalar_one())
    await db.commit()
    return LogTurnOut(turn_id=turn_id)


@router.post("/tools/get_turns", response_model=GetTurnsOut)
async def get_turns(
    body: GetTurnsIn,
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> GetTurnsOut:
    campaign_id: str = request.state.campaign_id
    limit = body.limit or 20
    params: dict[str, Any] = {"campaign_id": campaign_id, "limit": limit}

    where_clauses = ["campaign_id = :campaign_id"]

    if body.exclude_roles:
        excl_params = {f"excl_{i}": r for i, r in enumerate(body.exclude_roles)}
        placeholders = ", ".join(f"CAST(:{k} AS turn_role)" for k in excl_params)
        where_clauses.append(f"role NOT IN ({placeholders})")
        params.update(excl_params)

    if body.npc_name:
        where_clauses.append("npc_name = :npc_name")
        params["npc_name"] = body.npc_name

    if body.phase_filter:
        where_clauses.append("session_phase = :phase_filter")
        params["phase_filter"] = body.phase_filter

    order = "DESC"
    reverse_result = True

    if body.since_turn_id:
        where_clauses.append(
            "created_at > (SELECT created_at FROM turns WHERE id = :since_turn_id)"
        )
        params["since_turn_id"] = body.since_turn_id
        order = "ASC"
        reverse_result = False

    if body.before_turn_id:
        where_clauses.append(
            "created_at < (SELECT created_at FROM turns WHERE id = :before_turn_id)"
        )
        params["before_turn_id"] = body.before_turn_id
        order = "DESC"
        reverse_result = True

    where_sql = " AND ".join(where_clauses)
    query = f"SELECT * FROM turns WHERE {where_sql} ORDER BY created_at {order} LIMIT :limit"

    rows = await db.execute(text(query), params)
    turns = [_turn_row(r) for r in rows.mappings()]

    if reverse_result:
        turns = list(reversed(turns))

    return GetTurnsOut(turns=turns)


@router.post("/tools/get_routing_state", response_model=RoutingStateOut)
async def get_routing_state(
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> RoutingStateOut:
    campaign_id: str = request.state.campaign_id
    row = await db.execute(
        text("SELECT phase, active_npc_id FROM campaigns WHERE id = :campaign_id"),
        {"campaign_id": campaign_id},
    )
    r = row.mappings().first()
    if r is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return RoutingStateOut(
        phase=r["phase"],
        active_npc_id=str(r["active_npc_id"]) if r["active_npc_id"] else None,
    )


def _turn_row(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(r["id"]),
        "campaign_id": str(r["campaign_id"]),
        "role": r["role"],
        "content": r["content"],
        "npc_name": r["npc_name"],
        "audio_path": r["audio_path"],
        "image_path": r["image_path"],
        "metadata": r["metadata"] or {},
        "created_at": r["created_at"].isoformat() if r["created_at"] else None,
    }
