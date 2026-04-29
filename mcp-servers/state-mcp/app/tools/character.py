import json
from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..schemas import SaveCharacterIn, SaveCharacterOut, CampaignContextOut

router = APIRouter()


@router.post("/tools/save_character", response_model=SaveCharacterOut)
async def save_character(
    body: SaveCharacterIn,
    request: Request,
    db: AsyncSession = Depends(get_session),
):
    campaign_id = request.state.campaign_id
    c = body.character_json
    row = await db.execute(
        text("""
            INSERT INTO characters
                (campaign_id, name, background, class_and_level, abilities,
                 equipment, limitations, power_level, visual_description, portrait_path)
            VALUES
                (:campaign_id, :name, :background, :class_and_level, :abilities,
                 :equipment, :limitations, :power_level, :visual_description, :portrait_path)
            ON CONFLICT (campaign_id) DO UPDATE SET
                name = EXCLUDED.name,
                background = EXCLUDED.background,
                class_and_level = EXCLUDED.class_and_level,
                abilities = EXCLUDED.abilities,
                equipment = EXCLUDED.equipment,
                limitations = EXCLUDED.limitations,
                power_level = EXCLUDED.power_level,
                visual_description = EXCLUDED.visual_description,
                portrait_path = COALESCE(EXCLUDED.portrait_path, characters.portrait_path)
            RETURNING id
        """),
        {
            "campaign_id": campaign_id,
            "name": c.get("name", ""),
            "background": c.get("background", ""),
            "class_and_level": c.get("class_and_level", ""),
            "abilities": c.get("abilities", []),
            "equipment": c.get("equipment", []),
            "limitations": c.get("limitations", []),
            "power_level": c.get("power_level", "Novice"),
            "visual_description": c.get("visual_description", ""),
            "portrait_path": body.portrait_path,
        },
    )
    character_id = str(row.scalar_one())
    await db.commit()
    return SaveCharacterOut(character_id=character_id)


@router.post("/tools/get_campaign_context", response_model=CampaignContextOut)
async def get_campaign_context(
    request: Request,
    db: AsyncSession = Depends(get_session),
):
    campaign_id = request.state.campaign_id
    row = await db.execute(
        text("""
            SELECT c.id, c.title, c.language, c.phase, c.plan_json, c.visual_style,
                   ch.id as char_id, ch.name, ch.background, ch.class_and_level,
                   ch.abilities, ch.equipment, ch.limitations, ch.power_level,
                   ch.visual_description, ch.portrait_path
            FROM campaigns c
            LEFT JOIN characters ch ON ch.campaign_id = c.id
            WHERE c.id = :campaign_id
        """),
        {"campaign_id": campaign_id},
    )
    r = row.mappings().first()
    if r is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Campaign not found")

    campaign = {
        "id": str(r["id"]),
        "title": r["title"],
        "language": r["language"],
        "phase": r["phase"],
        "plan_json": r["plan_json"],
        "visual_style": r["visual_style"],
    }
    character = None
    if r["char_id"] is not None:
        character = {
            "id": str(r["char_id"]),
            "name": r["name"],
            "background": r["background"],
            "class_and_level": r["class_and_level"],
            "abilities": list(r["abilities"]) if r["abilities"] else [],
            "equipment": list(r["equipment"]) if r["equipment"] else [],
            "limitations": list(r["limitations"]) if r["limitations"] else [],
            "power_level": r["power_level"],
            "visual_description": r["visual_description"],
            "portrait_path": r["portrait_path"],
        }
    return CampaignContextOut(campaign=campaign, character=character)
