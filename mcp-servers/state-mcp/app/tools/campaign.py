from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..schemas import (
    CreateCampaignIn, CreateCampaignOut,
    SaveCampaignPlanIn, OkOut,
    SetPhaseIn,
)

router = APIRouter()


@router.post("/tools/create_campaign", response_model=CreateCampaignOut)
async def create_campaign(
    body: CreateCampaignIn,
    request: Request,
    db: AsyncSession = Depends(get_session),
):
    user_id = request.state.campaign_id  # reused as user_id during setup
    row = await db.execute(
        text(
            "INSERT INTO campaigns (user_id, title, language) "
            "VALUES (:user_id, :title, :language) RETURNING id"
        ),
        {"user_id": user_id, "title": body.title or "Untitled Campaign", "language": body.language},
    )
    campaign_id = str(row.scalar_one())
    await db.commit()
    return CreateCampaignOut(campaign_id=campaign_id)


@router.post("/tools/save_campaign_plan", response_model=OkOut)
async def save_campaign_plan(
    body: SaveCampaignPlanIn,
    request: Request,
    db: AsyncSession = Depends(get_session),
):
    import json
    campaign_id = request.state.campaign_id
    params: dict = {"campaign_id": campaign_id, "plan_json": json.dumps(body.plan_json)}
    query = "UPDATE campaigns SET plan_json = CAST(:plan_json AS jsonb), updated_at = now()"
    if body.visual_style is not None:
        query += ", visual_style = :visual_style"
        params["visual_style"] = body.visual_style
    query += " WHERE id = :campaign_id"
    await db.execute(text(query), params)
    await db.commit()
    return OkOut()


@router.post("/tools/set_phase", response_model=OkOut)
async def set_phase(
    body: SetPhaseIn,
    request: Request,
    db: AsyncSession = Depends(get_session),
):
    campaign_id = request.state.campaign_id
    await db.execute(
        text("UPDATE campaigns SET phase = CAST(:phase AS campaign_phase), updated_at = now() WHERE id = :campaign_id"),
        {"phase": body.phase, "campaign_id": campaign_id},
    )
    await db.commit()
    return OkOut()
