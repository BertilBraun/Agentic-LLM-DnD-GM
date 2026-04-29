"""Character creator agent (Section 5.2)."""
from __future__ import annotations

import json
import os

import httpx
import redis.asyncio as aioredis

STATE_MCP_URL = os.environ.get("STATE_MCP_URL", "http://state-mcp:8001")
MEDIA_MCP_URL = os.environ.get("MEDIA_MCP_URL", "http://media-mcp:8004")
LLM_SERVICE_URL = os.environ.get("LLM_SERVICE_URL", "http://llm-service:9001")
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

DM_VOICE_ID = "ash"
DM_VOICE_INSTRUCTIONS = "Speak in a deep, authoritative voice with dramatic pauses and varied intonation to bring the fantasy world to life"

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
        turns_resp = await _mcp(STATE_MCP_URL, "get_turns", {"limit": 20}, campaign_id)
        turns = turns_resp.get("turns", [])

        # 2. Build messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for t in turns:
            role = "assistant" if t["role"] in ("dm", "system") else "user"
            messages.append({"role": role, "content": t["content"]})
        if player_message:
            messages.append({"role": "user", "content": player_message})
            # Log player turn
            await _mcp(STATE_MCP_URL, "log_turn", {"role": "player", "content": player_message}, campaign_id)

        # 3. LLM call
        llm_resp = await _llm(messages)
        response_text: str = llm_resp["text"]

        done = "[DONE]" in response_text
        clean_text = response_text.replace("[DONE]", "").strip()

        # 4. Log DM response
        await _mcp(STATE_MCP_URL, "log_turn", {"role": "dm", "content": clean_text}, campaign_id)

        # 5. Publish text + audio
        await _publish(redis_client, campaign_id, {"type": "dm_text", "content": clean_text})

        try:
            speak_resp = await _mcp(
                MEDIA_MCP_URL,
                "speak",
                {"text": clean_text, "voice_id": DM_VOICE_ID, "voice_instructions": DM_VOICE_INSTRUCTIONS},
                campaign_id,
            )
            await _publish(redis_client, campaign_id, {"type": "audio_ready", "file_path": speak_resp["file_path"]})
        except Exception:
            pass  # audio is non-critical

        # 6. If done, parse character + generate portrait + transition
        if done:
            parse_messages = messages + [
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": CHARACTER_PARSE_PROMPT},
            ]
            parse_resp = await _llm(parse_messages, "json")
            character_json = json.loads(parse_resp["text"])

            visual_style = (ctx.get("campaign") or {}).get("visual_style", "fantasy digital art")

            try:
                img_resp = await _mcp(
                    MEDIA_MCP_URL,
                    "generate_image",
                    {
                        "prompt": character_json.get("visual_description", ""),
                        "style": visual_style,
                        "type": "portrait",
                    },
                    campaign_id,
                )
                portrait_path = img_resp.get("file_path")
            except Exception:
                portrait_path = None

            await _mcp(
                STATE_MCP_URL,
                "save_character",
                {"character_json": character_json, "portrait_path": portrait_path},
                campaign_id,
            )
            await _mcp(STATE_MCP_URL, "set_phase", {"phase": "campaign_design"}, campaign_id)

            if portrait_path:
                await _publish(redis_client, campaign_id, {"type": "portrait_ready", "file_path": portrait_path})
            await _publish(redis_client, campaign_id, {"type": "phase_change", "phase": "campaign_design"})

        return clean_text
    finally:
        await redis_client.aclose()
