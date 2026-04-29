import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth.dependencies import get_current_user
from ..database import get_db
from .schemas import CreateCampaignRequest, CampaignOut, CampaignDetailOut

router = APIRouter(prefix="/campaigns", tags=["campaigns"])


@router.get("", response_model=list[CampaignOut])
async def list_campaigns(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rows = await db.execute(
        text("SELECT id, title, phase, created_at FROM campaigns WHERE user_id = :uid ORDER BY created_at DESC"),
        {"uid": user["user_id"]},
    )
    return [
        CampaignOut(id=str(r["id"]), title=r["title"], phase=r["phase"], created_at=r["created_at"].isoformat())
        for r in rows.mappings()
    ]


@router.post("", response_model=CampaignOut, status_code=201)
async def create_campaign(
    body: CreateCampaignRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    campaign_id = str(uuid.uuid4())
    row = await db.execute(
        text(
            "INSERT INTO campaigns (id, user_id, title, language) "
            "VALUES (:id, :uid, :title, :lang) RETURNING id, title, phase, created_at"
        ),
        {"id": campaign_id, "uid": user["user_id"], "title": body.title or "Untitled Campaign", "lang": body.language},
    )
    await db.commit()
    r = row.mappings().first()
    return CampaignOut(id=str(r["id"]), title=r["title"], phase=r["phase"], created_at=r["created_at"].isoformat())


@router.get("/{campaign_id}", response_model=CampaignDetailOut)
async def get_campaign(
    campaign_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    row = await db.execute(
        text("SELECT * FROM campaigns WHERE id = :id AND user_id = :uid"),
        {"id": campaign_id, "uid": user["user_id"]},
    )
    r = row.mappings().first()
    if r is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return CampaignDetailOut(
        id=str(r["id"]),
        title=r["title"],
        language=r["language"],
        phase=r["phase"],
        plan_json=r["plan_json"],
        visual_style=r["visual_style"],
        created_at=r["created_at"].isoformat(),
        updated_at=r["updated_at"].isoformat(),
    )


@router.delete("/{campaign_id}")
async def delete_campaign(
    campaign_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        text("DELETE FROM campaigns WHERE id = :id AND user_id = :uid"),
        {"id": campaign_id, "uid": user["user_id"]},
    )
    await db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return {"ok": True}
