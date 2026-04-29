"""Character creator agent (Section 5.2)."""
from __future__ import annotations

import logging
import os

from pydantic import BaseModel, Field

from shared.helpers import call_mcp, call_llm, call_llm_structured, publish_event
from shared.mcp_models import (
    CampaignContextOut,
    GetTurnsOut,
    ImageOut,
    LogTurnOut,
    OkOut,
    SpeakOut,
)

STATE_MCP_URL = os.environ.get("STATE_MCP_URL", "http://state-mcp:8001")
MEDIA_MCP_URL = os.environ.get("MEDIA_MCP_URL", "http://media-mcp:8004")

DM_VOICE_ID = "ash"
DM_VOICE_INSTRUCTIONS = "Speak in a deep, authoritative voice with dramatic pauses and varied intonation to bring the fantasy world to life"

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a D&D character creation assistant. Ask 4-6 focused questions covering
background, class/level, abilities, equipment, limitations, and physical appearance.
When you have enough information for a complete character sheet, end your response with [DONE].
Keep questions conversational and help the player think through their choices."""

CHARACTER_PARSE_PROMPT = """Based on the conversation, create a complete PlayerCharacter JSON. Return ONLY valid JSON matching this schema exactly:
{
  "name": "string",
  "background": "string — background, origin story, personality",
  "class_and_level": "string — e.g. 'Level 3 Wizard'",
  "abilities": ["string", ...],
  "equipment": ["string", ...],
  "limitations": ["string", ...],
  "power_level": "Novice|Apprentice|Journeyman|Expert|Master|Legendary",
  "visual_description": "string — 300-500 chars, detailed physical appearance for portrait"
}
Ensure the character has meaningful limitations and a realistic power level."""


class PlayerCharacter(BaseModel):
    name: str = ""
    background: str = ""
    class_and_level: str = ""
    abilities: list[str] = Field(default_factory=list)
    equipment: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    power_level: str = "Novice"
    visual_description: str = ""


async def run(campaign_id: str, player_message: str) -> str:
    # 1. Load context
    ctx = await call_mcp(STATE_MCP_URL, "get_campaign_context", {}, campaign_id, CampaignContextOut)
    turns_resp = await call_mcp(STATE_MCP_URL, "get_turns", {"limit": 20}, campaign_id, GetTurnsOut)

    # 2. Build messages
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for t in turns_resp.turns:
        role = "assistant" if t.role in ("dm", "system") else "user"
        messages.append({"role": role, "content": t.content})
    if player_message:
        messages.append({"role": "user", "content": player_message})
        await call_mcp(STATE_MCP_URL, "log_turn", {"role": "player", "content": player_message}, campaign_id, LogTurnOut)

    # 3. LLM call
    llm_resp = await call_llm(messages)
    response_text: str = llm_resp["text"]

    done = "[DONE]" in response_text
    clean_text = response_text.replace("[DONE]", "").strip()

    # 4. Log DM response
    await call_mcp(STATE_MCP_URL, "log_turn", {"role": "dm", "content": clean_text}, campaign_id, LogTurnOut)

    # 5. Publish text + audio
    await publish_event(campaign_id, {"type": "dm_text", "content": clean_text})

    try:
        speak_resp = await call_mcp(
            MEDIA_MCP_URL, "speak",
            {"text": clean_text, "voice_id": DM_VOICE_ID, "voice_instructions": DM_VOICE_INSTRUCTIONS},
            campaign_id, SpeakOut,
        )
        if speak_resp.file_path:
            await publish_event(campaign_id, {"type": "audio_ready", "file_path": speak_resp.file_path})
    except Exception:
        logger.warning("TTS failed during character creation (non-critical)", exc_info=True)

    # 6. If done, parse character + generate portrait + transition
    if done:
        parse_messages = messages + [
            {"role": "assistant", "content": response_text},
            {"role": "user", "content": CHARACTER_PARSE_PROMPT},
        ]
        character = await call_llm_structured(parse_messages, PlayerCharacter)

        visual_style = ctx.campaign.visual_style or "fantasy digital art"

        portrait_path: str | None = None
        try:
            img_resp = await call_mcp(
                MEDIA_MCP_URL, "generate_image",
                {"prompt": character.visual_description, "style": visual_style, "type": "portrait"},
                campaign_id, ImageOut,
            )
            portrait_path = img_resp.file_path or None
        except Exception:
            logger.warning("Portrait generation failed during character creation (non-critical)", exc_info=True)

        await call_mcp(
            STATE_MCP_URL, "save_character",
            {"character_json": character.model_dump(), "portrait_path": portrait_path},
            campaign_id, OkOut,
        )
        await call_mcp(STATE_MCP_URL, "set_phase", {"phase": "campaign_design"}, campaign_id, OkOut)

        if portrait_path:
            await publish_event(campaign_id, {"type": "portrait_ready", "file_path": portrait_path})
        await publish_event(campaign_id, {"type": "phase_change", "phase": "campaign_design"})

    return clean_text
