"""DM agent — core gameplay agent (Section 5.4)."""
from __future__ import annotations

import asyncio
import json
import os

import httpx
import redis.asyncio as aioredis

STATE_MCP_URL = os.environ.get("STATE_MCP_URL", "http://state-mcp:8001")
KNOWLEDGE_MCP_URL = os.environ.get("KNOWLEDGE_MCP_URL", "http://knowledge-mcp:8003")
MEDIA_MCP_URL = os.environ.get("MEDIA_MCP_URL", "http://media-mcp:8004")
MEMORY_AGENT_URL = os.environ.get("MEMORY_AGENT_URL", "http://memory-agent:8014")
LLM_SERVICE_URL = os.environ.get("LLM_SERVICE_URL", "http://llm-service:9001")
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

DM_VOICE_ID = "ash"
DM_VOICE_INSTRUCTIONS = "Speak in a deep, authoritative voice with dramatic pauses and varied intonation to bring the fantasy world to life"

DM_SYSTEM = """You are an expert Dungeon Master for a voice-driven D&D campaign.
Respond with JSON matching this schema exactly:
{{
  "gm_speech": "string — narration (2-5 sentences, vivid, immersive)",
  "scene_description": "string — visual description for image gen (400-800 chars)",
  "memory_note": "string — one-sentence summary of what changed in the world",
  "invoke_npc": null | {{
    "name": "string",
    "role": "string",
    "visual_description": "string",
    "voice_id": "alloy|ash|ballad|coral|echo|fable|onyx|nova|sage|shimmer|verse",
    "voice_instructions": "string",
    "briefing": {{
      "goals": "string",
      "knows": "string",
      "mood": "string",
      "reveal_if": "string"
    }},
    "opening_line": "string"
  }}
}}

Campaign: {campaign_json}
Character: {character_json}
Long-term memory: {long_term_summary}
Recent events: {recent_events}
Recalled context: {recalled_context}
World context: {world_context}
Recent turns (last 10, NPC excluded): {recent_turns}"""


async def _mcp(base_url: str, tool: str, body: dict, campaign_id: str, timeout: float = 60) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{base_url}/tools/{tool}",
            json=body,
            headers={"X-Campaign-ID": campaign_id},
        )
        resp.raise_for_status()
        return resp.json()


async def _llm(messages: list[dict], response_format: str = "text") -> dict:
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            f"{LLM_SERVICE_URL}/generate",
            json={"messages": messages, "response_format": response_format},
        )
        resp.raise_for_status()
        return resp.json()


async def _a2a_memory(campaign_id: str, query: str, new_event: str) -> dict:
    payload = json.dumps({"query": query, "new_event": new_event})
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            MEMORY_AGENT_URL + "/",
            json={"jsonrpc": "2.0", "method": "tasks/send",
                  "params": {"task_id": "mem", "campaign_id": campaign_id, "message": payload},
                  "id": 1},
        )
        resp.raise_for_status()
        result_output = resp.json().get("result", {}).get("output", "{}")
    try:
        return json.loads(result_output)
    except Exception:
        return {}


async def _publish(redis_client: aioredis.Redis, campaign_id: str, event: dict) -> None:
    await redis_client.publish(f"sse:campaign:{campaign_id}", json.dumps(event))


async def run(campaign_id: str, player_message: str) -> str:
    redis_client = aioredis.from_url(REDIS_URL)
    try:
        # 1. Gather all context in parallel
        ctx, turns_resp, world_resp, mem_resp = await asyncio.gather(
            _mcp(STATE_MCP_URL, "get_campaign_context", {}, campaign_id),
            _mcp(STATE_MCP_URL, "get_turns", {"limit": 10, "exclude_roles": ["npc"]}, campaign_id),
            _mcp(KNOWLEDGE_MCP_URL, "get_world_context", {"focus_text": player_message}, campaign_id),
            _a2a_memory(campaign_id, player_message, ""),
            return_exceptions=True,
        )

        campaign = (ctx.get("campaign") or {}) if isinstance(ctx, dict) else {}
        character = (ctx.get("character") or {}) if isinstance(ctx, dict) else {}
        turns = (turns_resp.get("turns") or []) if isinstance(turns_resp, dict) else []
        world_context = (world_resp.get("context") or "") if isinstance(world_resp, dict) else ""
        long_term_summary = (mem_resp.get("long_term_summary") or "") if isinstance(mem_resp, dict) else ""
        recalled_context = (mem_resp.get("recalled_context") or "") if isinstance(mem_resp, dict) else ""
        recent_events = (mem_resp.get("recent_events") or []) if isinstance(mem_resp, dict) else []

        # 2. Log player turn
        await _mcp(STATE_MCP_URL, "log_turn", {"role": "player", "content": player_message}, campaign_id)

        # 3. LLM call
        recent_turns_text = "\n".join(f"[{t['role'].upper()}] {t['content']}" for t in turns)
        system = DM_SYSTEM.format(
            campaign_json=json.dumps(campaign, indent=2),
            character_json=json.dumps(character, indent=2),
            long_term_summary=long_term_summary or "None yet.",
            recent_events="\n".join(f"- {e}" for e in recent_events) or "None yet.",
            recalled_context=recalled_context or "None.",
            world_context=world_context or "No world data yet.",
            recent_turns=recent_turns_text or "None.",
        )
        llm_resp = await _llm(
            [{"role": "system", "content": system}, {"role": "user", "content": player_message}],
            "json",
        )
        dm_response = json.loads(llm_resp["text"])

        gm_speech: str = dm_response.get("gm_speech", "")
        scene_description: str = dm_response.get("scene_description", "")
        memory_note: str = dm_response.get("memory_note", "")
        invoke_npc = dm_response.get("invoke_npc")
        visual_style = campaign.get("visual_style", "fantasy digital art, detailed")

        # 4. Log DM turn (text only — media paths added by background task)
        log_resp = await _mcp(STATE_MCP_URL, "log_turn", {
            "role": "dm", "content": gm_speech,
        }, campaign_id)
        dm_turn_id = log_resp.get("turn_id")

        # 5. Publish dm_text — frontend unlocks input immediately
        await _publish(redis_client, campaign_id, {"type": "dm_text", "content": gm_speech})

        # 6. TTS + image + memory + world as background (user can type again now)
        async def _background(rc: aioredis.Redis):
            async def _speak():
                try:
                    r = await _mcp(MEDIA_MCP_URL, "speak",
                                   {"text": gm_speech, "voice_id": DM_VOICE_ID,
                                    "voice_instructions": DM_VOICE_INSTRUCTIONS},
                                   campaign_id, timeout=90)
                    if r.get("file_path"):
                        await _publish(rc, campaign_id, {"type": "audio_ready", "file_path": r["file_path"]})
                except Exception:
                    pass

            async def _image():
                try:
                    r = await _mcp(MEDIA_MCP_URL, "generate_image",
                                   {"prompt": scene_description, "style": visual_style, "type": "scene"},
                                   campaign_id, timeout=120)
                    if r.get("file_path"):
                        await _publish(rc, campaign_id, {"type": "scene_ready", "file_path": r["file_path"]})
                except Exception:
                    pass

            async def _memory_and_world():
                try:
                    await _a2a_memory(campaign_id, player_message, memory_note)
                except Exception:
                    pass
                try:
                    await _mcp(KNOWLEDGE_MCP_URL, "update_world",
                               {"narrative_text": gm_speech}, campaign_id)
                except Exception:
                    pass

            await asyncio.gather(_speak(), _image(), _memory_and_world())

            if invoke_npc:
                await _handle_npc(rc, campaign_id, invoke_npc, visual_style)

            await rc.aclose()

        bg_redis = aioredis.from_url(REDIS_URL)
        asyncio.create_task(_background(bg_redis))

        return gm_speech
    finally:
        await redis_client.aclose()


async def _handle_npc(redis_client: aioredis.Redis, campaign_id: str, invoke_npc: dict, visual_style: str) -> None:
    save_resp = await _mcp(STATE_MCP_URL, "save_npc", {"npc_json": invoke_npc}, campaign_id)
    npc_id = save_resp.get("npc_id", "")

    portrait_resp, npc_audio_resp = await asyncio.gather(
        _mcp(MEDIA_MCP_URL, "generate_image",
             {"prompt": invoke_npc.get("visual_description", ""), "style": visual_style, "type": "portrait"},
             campaign_id, timeout=120),
        _mcp(MEDIA_MCP_URL, "speak",
             {"text": invoke_npc.get("opening_line", ""),
              "voice_id": invoke_npc.get("voice_id", "ash"),
              "voice_instructions": invoke_npc.get("voice_instructions", "")},
             campaign_id, timeout=90),
        return_exceptions=True,
    )
    portrait_path = portrait_resp.get("file_path") if isinstance(portrait_resp, dict) else None
    opening_audio_path = npc_audio_resp.get("file_path") if isinstance(npc_audio_resp, dict) else None

    if portrait_path:
        await _mcp(STATE_MCP_URL, "save_npc",
                   {"npc_json": invoke_npc, "portrait_path": portrait_path}, campaign_id)

    log_resp = await _mcp(STATE_MCP_URL, "log_turn", {
        "role": "npc",
        "content": invoke_npc.get("opening_line", ""),
        "npc_name": invoke_npc.get("name"),
        "audio_path": opening_audio_path,
    }, campaign_id)
    conv_start_turn_id = log_resp.get("turn_id", "")

    await _mcp(STATE_MCP_URL, "set_active_npc", {
        "npc_id": npc_id,
        "briefing": invoke_npc.get("briefing", {}),
        "conv_start_turn_id": conv_start_turn_id,
    }, campaign_id)

    await _publish(redis_client, campaign_id, {
        "type": "npc_introduced",
        "npc_name": invoke_npc.get("name"),
        "portrait_path": portrait_path,
    })
    await _publish(redis_client, campaign_id, {
        "type": "npc_speech",
        "npc_name": invoke_npc.get("name"),
        "content": invoke_npc.get("opening_line", ""),
    })
    if opening_audio_path:
        await _publish(redis_client, campaign_id,
                       {"type": "audio_ready", "file_path": opening_audio_path})
