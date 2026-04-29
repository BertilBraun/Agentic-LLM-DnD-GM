"""DM agent — core gameplay agent (Section 5.4)."""

from __future__ import annotations

import asyncio
import json
import logging
import httpx
from pydantic import BaseModel

from shared.helpers import (
    call_mcp,
    call_llm_structured,
    publish_event,
    STATE_MCP_URL,
    KNOWLEDGE_MCP_URL,
    MEDIA_MCP_URL,
    MEMORY_AGENT_URL,
    DM_VOICE_ID,
    DM_VOICE_INSTRUCTIONS,
)
from shared.mcp_models import (
    CampaignContextOut,
    GetTurnsOut,
    ImageOut,
    LogTurnOut,
    NpcBriefing,
    OkOut,
    SaveNpcOut,
    SpeakOut,
    UpdateWorldOut,
    WorldContextOut,
)

logger = logging.getLogger(__name__)

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

PACING RULES — follow strictly:
- Set invoke_npc to null for the vast majority of turns. Most turns are exploration: describe the environment, present choices, reveal details, react to player actions.
- Only set invoke_npc when the player explicitly addresses or approaches a specific named or visible person in the scene. Never invent a surprise NPC encounter unprompted.
- Do not invoke an NPC two turns in a row. After a conversation ends, give at least one or two pure exploration turns before allowing another NPC.
- Good DM turns without NPCs: vivid scene descriptions, consequences of player actions, ambient world details, forks in the road, sounds/smells/atmosphere, discovered objects or clues.

Campaign: {campaign_json}
Character: {character_json}
Long-term memory: {long_term_summary}
Recent events: {recent_events}
Recalled context: {recalled_context}
World context: {world_context}
Recent turns (last 10, NPC excluded): {recent_turns}"""


class InvokeNpc(BaseModel):
    name: str = ''
    role: str = ''
    visual_description: str = ''
    voice_id: str = 'ash'
    voice_instructions: str = ''
    briefing: NpcBriefing = NpcBriefing()
    opening_line: str = ''


class DmResponse(BaseModel):
    gm_speech: str = ''
    scene_description: str = ''
    memory_note: str = ''
    invoke_npc: InvokeNpc | None = None


class MemoryAgentResult(BaseModel):
    recalled_context: str = ''
    long_term_summary: str = ''
    recent_events: list[str] = []


async def _call_memory_agent(campaign_id: str, query: str, new_event: str) -> MemoryAgentResult:
    payload = json.dumps({'query': query, 'new_event': new_event})
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            MEMORY_AGENT_URL + '/',
            json={
                'jsonrpc': '2.0',
                'method': 'tasks/send',
                'params': {'task_id': 'mem', 'campaign_id': campaign_id, 'message': payload},
                'id': 1,
            },
        )
        resp.raise_for_status()
        result_output = resp.json().get('result', {}).get('output', '{}')
    try:
        return MemoryAgentResult.model_validate_json(result_output)
    except Exception:
        logger.warning('Failed to parse memory agent response')
        return MemoryAgentResult()


async def run(campaign_id: str, player_message: str) -> str:
    # 1. Gather all context in parallel
    ctx, turns_resp, world_resp, mem_resp = await asyncio.gather(
        call_mcp(STATE_MCP_URL, 'get_campaign_context', {}, campaign_id, CampaignContextOut),
        call_mcp(STATE_MCP_URL, 'get_turns', {'limit': 10, 'exclude_roles': ['npc']}, campaign_id, GetTurnsOut),
        call_mcp(KNOWLEDGE_MCP_URL, 'get_world_context', {'focus_text': player_message}, campaign_id, WorldContextOut),
        _call_memory_agent(campaign_id, player_message, ''),
        return_exceptions=True,
    )

    campaign = ctx.campaign if isinstance(ctx, CampaignContextOut) else CampaignContextOut().campaign
    character = ctx.character if isinstance(ctx, CampaignContextOut) else None
    turns = turns_resp.turns if isinstance(turns_resp, GetTurnsOut) else []
    world_context = world_resp.context if isinstance(world_resp, WorldContextOut) else ''
    mem = mem_resp if isinstance(mem_resp, MemoryAgentResult) else MemoryAgentResult()
    long_term_summary = mem.long_term_summary
    recalled_context = mem.recalled_context
    recent_events = mem.recent_events

    # 2. Log player turn (system-tagged messages are logged as system, not player)
    is_system_trigger = player_message.startswith('[SYSTEM]')
    log_role = 'system' if is_system_trigger else 'player'
    await call_mcp(STATE_MCP_URL, 'log_turn', {'role': log_role, 'content': player_message}, campaign_id, LogTurnOut)

    # 3. LLM call
    recent_turns_text = '\n'.join(f'[{t.role.upper()}] {t.content}' for t in turns)
    system = DM_SYSTEM.format(
        campaign_json=json.dumps(campaign.model_dump(), indent=2),
        character_json=json.dumps(character.model_dump() if character else {}, indent=2),
        long_term_summary=long_term_summary or 'None yet.',
        recent_events='\n'.join(f'- {e}' for e in recent_events) or 'None yet.',
        recalled_context=recalled_context or 'None.',
        world_context=world_context or 'No world data yet.',
        recent_turns=recent_turns_text or 'None.',
    )
    dm = await call_llm_structured(
        [{'role': 'system', 'content': system}, {'role': 'user', 'content': player_message}],
        DmResponse,
        timeout=90,
    )

    visual_style: str = campaign.visual_style or 'fantasy digital art, detailed'

    # 4. Log DM turn
    await call_mcp(STATE_MCP_URL, 'log_turn', {'role': 'dm', 'content': dm.gm_speech}, campaign_id, LogTurnOut)

    # 5. Publish dm_text — frontend unlocks input immediately
    await publish_event(campaign_id, {'type': 'dm_text', 'content': dm.gm_speech})

    # 6. TTS + image + memory + world as background (user can type again now)
    async def _background() -> None:
        async def _speak() -> None:
            try:
                r = await call_mcp(
                    MEDIA_MCP_URL,
                    'speak',
                    {'text': dm.gm_speech, 'voice_id': DM_VOICE_ID, 'voice_instructions': DM_VOICE_INSTRUCTIONS},
                    campaign_id,
                    SpeakOut,
                    timeout=90,
                )
                if r.file_path:
                    await publish_event(campaign_id, {'type': 'audio_ready', 'file_path': r.file_path})
            except Exception:
                logger.warning('DM TTS failed (non-critical)', exc_info=True)

        async def _image() -> None:
            try:
                r = await call_mcp(
                    MEDIA_MCP_URL,
                    'generate_image',
                    {'prompt': dm.scene_description, 'style': visual_style, 'type': 'scene'},
                    campaign_id,
                    ImageOut,
                    timeout=120,
                )
                if r.file_path:
                    await publish_event(campaign_id, {'type': 'scene_ready', 'file_path': r.file_path})
            except Exception:
                logger.warning('Scene image generation failed (non-critical)', exc_info=True)

        async def _memory_and_world() -> None:
            try:
                await _call_memory_agent(campaign_id, player_message, dm.memory_note)
            except Exception:
                logger.warning('Memory agent update failed (non-critical)', exc_info=True)
            try:
                await call_mcp(
                    KNOWLEDGE_MCP_URL, 'update_world', {'narrative_text': dm.gm_speech}, campaign_id, UpdateWorldOut
                )
            except Exception:
                logger.warning('World update after DM turn failed (non-critical)', exc_info=True)

        await asyncio.gather(_speak(), _image(), _memory_and_world())

        if dm.invoke_npc:
            await _handle_npc(campaign_id, dm.invoke_npc, visual_style)

    asyncio.create_task(_background())

    return dm.gm_speech


async def _handle_npc(campaign_id: str, invoke_npc: InvokeNpc, visual_style: str) -> None:
    save_resp = await call_mcp(
        STATE_MCP_URL, 'save_npc', {'npc_json': invoke_npc.model_dump()}, campaign_id, SaveNpcOut
    )
    npc_id: str = save_resp.npc_id

    portrait_resp, npc_audio_resp = await asyncio.gather(
        call_mcp(
            MEDIA_MCP_URL,
            'generate_image',
            {'prompt': invoke_npc.visual_description, 'style': visual_style, 'type': 'portrait'},
            campaign_id,
            ImageOut,
            timeout=120,
        ),
        call_mcp(
            MEDIA_MCP_URL,
            'speak',
            {
                'text': invoke_npc.opening_line,
                'voice_id': invoke_npc.voice_id,
                'voice_instructions': invoke_npc.voice_instructions,
            },
            campaign_id,
            SpeakOut,
            timeout=90,
        ),
        return_exceptions=True,
    )
    portrait_path: str | None = portrait_resp.file_path if isinstance(portrait_resp, ImageOut) else None
    opening_audio_path: str | None = npc_audio_resp.file_path if isinstance(npc_audio_resp, SpeakOut) else None

    if portrait_path:
        await call_mcp(
            STATE_MCP_URL,
            'save_npc',
            {'npc_json': invoke_npc.model_dump(), 'portrait_path': portrait_path},
            campaign_id,
            SaveNpcOut,
        )

    log_resp = await call_mcp(
        STATE_MCP_URL,
        'log_turn',
        {
            'role': 'npc',
            'content': invoke_npc.opening_line,
            'npc_name': invoke_npc.name,
            'audio_path': opening_audio_path,
        },
        campaign_id,
        LogTurnOut,
    )
    conv_start_turn_id: str = log_resp.turn_id

    await call_mcp(
        STATE_MCP_URL,
        'set_active_npc',
        {
            'npc_id': npc_id,
            'briefing': invoke_npc.briefing.model_dump(),
            'conv_start_turn_id': conv_start_turn_id,
        },
        campaign_id,
        OkOut,
    )

    await publish_event(
        campaign_id,
        {
            'type': 'npc_introduced',
            'npc_name': invoke_npc.name,
            'portrait_path': portrait_path,
        },
    )
    await publish_event(
        campaign_id,
        {
            'type': 'npc_speech',
            'npc_name': invoke_npc.name,
            'content': invoke_npc.opening_line,
        },
    )
    if opening_audio_path:
        await publish_event(campaign_id, {'type': 'audio_ready', 'file_path': opening_audio_path})
