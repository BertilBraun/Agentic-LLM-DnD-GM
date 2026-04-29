"""Campaign designer agent (Section 5.3)."""
from __future__ import annotations

import json
import os

import httpx
import redis.asyncio as aioredis

STATE_MCP_URL = os.environ.get("STATE_MCP_URL", "http://state-mcp:8001")
KNOWLEDGE_MCP_URL = os.environ.get("KNOWLEDGE_MCP_URL", "http://knowledge-mcp:8003")
LLM_SERVICE_URL = os.environ.get("LLM_SERVICE_URL", "http://llm-service:9001")
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

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


async def _mcp(base_url: str, tool: str, body: dict, campaign_id: str) -> dict:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{base_url}/tools/{tool}",
            json=body,
            headers={"X-Campaign-ID": campaign_id},
        )
        resp.raise_for_status()
        return resp.json()


async def _llm(messages: list[dict], response_format: str = "text") -> dict:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{LLM_SERVICE_URL}/generate",
            json={"messages": messages, "response_format": response_format},
        )
        resp.raise_for_status()
        return resp.json()


async def _publish(redis_client: aioredis.Redis, campaign_id: str, event: dict) -> None:
    await redis_client.publish(f"sse:campaign:{campaign_id}", json.dumps(event))


async def run(campaign_id: str, player_message: str) -> str:
    redis_client = aioredis.from_url(REDIS_URL)
    try:
        # 1. Load context
        ctx = await _mcp(STATE_MCP_URL, "get_campaign_context", {}, campaign_id)
        character = ctx.get("character") or {}
        turns_resp = await _mcp(STATE_MCP_URL, "get_turns", {"limit": 20}, campaign_id)
        turns = turns_resp.get("turns", [])

        # 2. Build messages
        system = SYSTEM_TEMPLATE.format(character_json=json.dumps(character, indent=2))
        messages = [{"role": "system", "content": system}]
        for t in turns:
            role = "assistant" if t["role"] in ("dm", "system") else "user"
            messages.append({"role": role, "content": t["content"]})
        if player_message:
            messages.append({"role": "user", "content": player_message})
            await _mcp(STATE_MCP_URL, "log_turn", {"role": "player", "content": player_message}, campaign_id)

        # 3. LLM call
        llm_resp = await _llm(messages)
        response_text: str = llm_resp["text"]

        done = "[DONE]" in response_text
        clean_text = response_text.replace("[DONE]", "").strip()

        # 4. Log + publish
        await _mcp(STATE_MCP_URL, "log_turn", {"role": "dm", "content": clean_text}, campaign_id)
        await _publish(redis_client, campaign_id, {"type": "dm_text", "content": clean_text})

        # 5. If done, parse plan + seed world + transition
        if done:
            parse_messages = messages + [
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": PLAN_PARSE_PROMPT},
            ]
            parse_resp = await _llm(parse_messages, "json")
            plan_json = json.loads(parse_resp["text"])

            await _mcp(STATE_MCP_URL, "save_campaign_plan", {"plan_json": plan_json, "visual_style": plan_json.get("visual_style")}, campaign_id)

            try:
                seed_text = plan_json.get("synopsis", "") + " " + " ".join(plan_json.get("acts", []))
                await _mcp(KNOWLEDGE_MCP_URL, "update_world", {"narrative_text": seed_text}, campaign_id)
            except Exception:
                pass

            await _mcp(STATE_MCP_URL, "set_phase", {"phase": "active"}, campaign_id)
            await _publish(redis_client, campaign_id, {"type": "campaign_plan_ready", "plan": plan_json})
            await _publish(redis_client, campaign_id, {"type": "phase_change", "phase": "active"})

        return clean_text
    finally:
        await redis_client.aclose()
