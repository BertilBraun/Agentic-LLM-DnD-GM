"""NPC agent — handles one turn of multi-turn NPC conversation (Section 5.5)."""
from __future__ import annotations

import json
import os

import httpx
import redis.asyncio as aioredis

STATE_MCP_URL = os.environ.get("STATE_MCP_URL", "http://state-mcp:8001")
KNOWLEDGE_MCP_URL = os.environ.get("KNOWLEDGE_MCP_URL", "http://knowledge-mcp:8003")
MEDIA_MCP_URL = os.environ.get("MEDIA_MCP_URL", "http://media-mcp:8004")
LLM_SERVICE_URL = os.environ.get("LLM_SERVICE_URL", "http://llm-service:9001")
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

NPC_SYSTEM = """You are {npc_name} in a live D&D session. Stay in character at all times.

STATIC PROFILE:
  Role: {npc_role}
  Voice/personality: {voice_instructions}

BRIEFING FOR THIS CONVERSATION:
  Goals:      {goals}
  You know:   {knows}
  Your mood:  {mood}
  Reveal if:  {reveal_if}

CONTEXT (what led to this conversation):
{preamble_turns}

CONVERSATION SO FAR:
{conv_turns}

Respond with JSON:
{{
  "npc_speech": "string — your next line, in character, natural dialogue length",
  "done": boolean
}}

Never set done=true mid-conversation. End only when the conversation genuinely feels over
(goodbye said, topic fully resolved, or continuing would feel forced)."""

SUMMARY_SYSTEM = """Summarise this NPC conversation in 1-2 sentences from the DM's perspective,
noting only what the player learned or agreed to. Be concise.
Return JSON: {{"summary": "string"}}"""


async def _mcp(base_url: str, tool: str, body: dict, campaign_id: str, timeout: float = 30) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
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


def _format_turn(t: dict) -> str:
    role = t["role"]
    content = t["content"]
    if role == "player":
        return f"[Player] {content}"
    elif role == "npc":
        return f"[You] {content}"
    else:
        return f"[DM] {content}"


async def run(campaign_id: str, player_message: str) -> str:
    redis_client = aioredis.from_url(REDIS_URL)
    try:
        # 1. Get active NPC state
        npc_state = await _mcp(STATE_MCP_URL, "get_active_npc_state", {}, campaign_id)
        npc_id = npc_state.get("npc_id")
        if not npc_id:
            return ""

        briefing = npc_state.get("briefing") or {}
        conv_start_turn_id = npc_state.get("conv_start_turn_id")

        # 2. Get NPC record
        npc_resp = await _mcp(STATE_MCP_URL, "get_npc", {"npc_id": npc_id}, campaign_id)
        npc = npc_resp.get("npc") or {}

        # 3. Preamble: 3 turns before conversation started
        preamble_turns = []
        if conv_start_turn_id:
            pre_resp = await _mcp(STATE_MCP_URL, "get_turns",
                                  {"before_turn_id": conv_start_turn_id, "limit": 3}, campaign_id)
            preamble_turns = pre_resp.get("turns", [])

        # 4. Conversation turns since start
        conv_turns = []
        if conv_start_turn_id:
            conv_resp = await _mcp(STATE_MCP_URL, "get_turns",
                                   {"since_turn_id": conv_start_turn_id}, campaign_id)
            conv_turns = conv_resp.get("turns", [])

        # Add current player message to conv turns for context (not yet logged)
        preamble_text = "\n".join(_format_turn(t) for t in preamble_turns) or "(start of session)"
        conv_text = "\n".join(_format_turn(t) for t in conv_turns)
        if player_message:
            conv_text += f"\n[Player] {player_message}"

        # 5. Build system prompt
        system = NPC_SYSTEM.format(
            npc_name=npc.get("name", "Unknown"),
            npc_role=npc.get("role", ""),
            voice_instructions=npc.get("voice_instructions", ""),
            goals=briefing.get("goals", ""),
            knows=briefing.get("knows", ""),
            mood=briefing.get("mood", ""),
            reveal_if=briefing.get("reveal_if", ""),
            preamble_turns=preamble_text,
            conv_turns=conv_text,
        )

        # 6. LLM call
        llm_resp = await _llm(
            [{"role": "system", "content": system}, {"role": "user", "content": player_message}],
            "json",
        )
        npc_data = json.loads(llm_resp["text"])
        npc_speech: str = npc_data.get("npc_speech", "")
        done: bool = npc_data.get("done", False)

        # 7. Generate audio
        audio_path = None
        try:
            speak_resp = await _mcp(MEDIA_MCP_URL, "speak", {
                "text": npc_speech,
                "voice_id": npc.get("voice_id", "ash"),
                "voice_instructions": npc.get("voice_instructions", ""),
            }, campaign_id, timeout=90)
            audio_path = speak_resp.get("file_path")
        except Exception:
            pass

        # 8. Log player + NPC turns
        await _mcp(STATE_MCP_URL, "log_turn", {"role": "player", "content": player_message}, campaign_id)
        await _mcp(STATE_MCP_URL, "log_turn", {
            "role": "npc",
            "content": npc_speech,
            "npc_name": npc.get("name"),
            "audio_path": audio_path,
            "metadata": {"audio_path": audio_path},
        }, campaign_id)

        # 9. Publish events
        await _publish(redis_client, campaign_id, {
            "type": "npc_speech",
            "npc_name": npc.get("name"),
            "content": npc_speech,
        })
        if audio_path:
            await _publish(redis_client, campaign_id, {"type": "audio_ready", "file_path": audio_path})

        # 10. If done, summarise and clear
        if done:
            all_conv = conv_turns + [
                {"role": "player", "content": player_message},
                {"role": "npc", "content": npc_speech},
            ]
            conv_transcript = "\n".join(_format_turn(t) for t in all_conv)
            try:
                summary_resp = await _llm([
                    {"role": "system", "content": SUMMARY_SYSTEM},
                    {"role": "user", "content": f"Conversation with {npc.get('name')}:\n\n{conv_transcript}"},
                ], "json")
                summary_data = json.loads(summary_resp["text"])
                summary_text: str = summary_data.get("summary", "")
            except Exception:
                summary_text = f"Conversation with {npc.get('name')} concluded."

            await _mcp(STATE_MCP_URL, "log_turn", {
                "role": "system",
                "content": summary_text,
                "metadata": {"type": "npc_conv_summary", "npc_name": npc.get("name")},
            }, campaign_id)
            await _mcp(STATE_MCP_URL, "clear_active_npc", {}, campaign_id)

            try:
                await _mcp(KNOWLEDGE_MCP_URL, "update_world", {"narrative_text": conv_transcript}, campaign_id)
            except Exception:
                pass

            await _publish(redis_client, campaign_id, {
                "type": "npc_conversation_ended",
                "npc_name": npc.get("name"),
                "summary": summary_text,
            })

        return npc_speech
    finally:
        await redis_client.aclose()
