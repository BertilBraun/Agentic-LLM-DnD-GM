"""Campaign designer agent (Section 5.3)."""
from __future__ import annotations

import json
import logging
import os

from pydantic import BaseModel, Field

from shared.helpers import call_mcp, call_llm, call_llm_structured, publish_event
from shared.mcp_models import (
    CampaignContextOut,
    GetTurnsOut,
    LogTurnOut,
    OkOut,
    UpdateWorldOut,
)

STATE_MCP_URL = os.environ.get("STATE_MCP_URL", "http://state-mcp:8001")
KNOWLEDGE_MCP_URL = os.environ.get("KNOWLEDGE_MCP_URL", "http://knowledge-mcp:8003")

logger = logging.getLogger(__name__)

SYSTEM_TEMPLATE = """You are an expert D&D campaign designer. You have the player's character sheet.
Ask 3-5 focused questions about genre, tone, themes, and how the character's background
should drive the story. End your final response with [DONE].

Character: {character_json}"""

PLAN_PARSE_PROMPT = """Based on the conversation, create a complete CampaignPlan JSON. Return ONLY valid JSON:
{{
  "title": "string — compelling campaign title",
  "synopsis": "string — 2-3 sentences covering central conflict, stakes, uniqueness",
  "acts": ["string", ...],
  "visual_style": "string — 400-600 chars, detailed visual style guide for image generation",
  "character_context": "string — how the campaign is balanced around the player character"
}}
Make the visual_style rich: art style, color palette, lighting, atmosphere, level of detail."""


class CampaignPlan(BaseModel):
    title: str = ""
    synopsis: str = ""
    acts: list[str] = Field(default_factory=list)
    visual_style: str = ""
    character_context: str = ""


async def run(campaign_id: str, player_message: str) -> str:
    # 1. Load context
    ctx = await call_mcp(STATE_MCP_URL, "get_campaign_context", {}, campaign_id, CampaignContextOut)
    turns_resp = await call_mcp(STATE_MCP_URL, "get_turns", {"limit": 20}, campaign_id, GetTurnsOut)

    # 2. Build messages
    character_json = json.dumps(ctx.character.model_dump() if ctx.character else {}, indent=2)
    system = SYSTEM_TEMPLATE.format(character_json=character_json)
    messages: list[dict] = [{"role": "system", "content": system}]
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

    # 4. Log + publish
    await call_mcp(STATE_MCP_URL, "log_turn", {"role": "dm", "content": clean_text}, campaign_id, LogTurnOut)
    await publish_event(campaign_id, {"type": "dm_text", "content": clean_text})

    # 5. If done, parse plan + seed world + transition
    if done:
        parse_messages = messages + [
            {"role": "assistant", "content": response_text},
            {"role": "user", "content": PLAN_PARSE_PROMPT},
        ]
        plan = await call_llm_structured(parse_messages, CampaignPlan)

        await call_mcp(
            STATE_MCP_URL, "save_campaign_plan",
            {"plan_json": plan.model_dump(), "visual_style": plan.visual_style},
            campaign_id, OkOut,
        )

        try:
            seed_text = plan.synopsis + " " + " ".join(plan.acts)
            await call_mcp(KNOWLEDGE_MCP_URL, "update_world", {"narrative_text": seed_text}, campaign_id, UpdateWorldOut)
        except Exception:
            logger.warning("Failed to seed knowledge base during campaign plan finalization", exc_info=True)

        await call_mcp(STATE_MCP_URL, "set_phase", {"phase": "active"}, campaign_id, OkOut)
        await publish_event(campaign_id, {"type": "campaign_plan_ready", "plan": plan.model_dump()})
        await publish_event(campaign_id, {"type": "phase_change", "phase": "active"})

    return clean_text
