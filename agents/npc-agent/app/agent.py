"""NPC agent — handles one turn of multi-turn NPC conversation (Section 5.5)."""

from __future__ import annotations

import logging
from pydantic import BaseModel

from shared.helpers import (
    call_mcp,
    call_llm_structured,
    publish_event,
    STATE_MCP_URL,
    KNOWLEDGE_MCP_URL,
    MEDIA_MCP_URL,
)
from shared.mcp_models import (
    ActiveNpcStateOut,
    GetTurnsOut,
    GetNpcOut,
    LogTurnOut,
    NpcBriefing,
    OkOut,
    SpeakOut,
    UpdateWorldOut,
)

logger = logging.getLogger(__name__)

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


class NpcTurnResponse(BaseModel):
    npc_speech: str = ''
    done: bool = False


class NpcConversationSummary(BaseModel):
    summary: str = ''


def _format_turn(role: str, content: str) -> str:
    if role == 'player':
        return f'[Player] {content}'
    elif role == 'npc':
        return f'[You] {content}'
    else:
        return f'[DM] {content}'


async def run(campaign_id: str, player_message: str) -> str:
    # 1. Get active NPC state
    npc_state = await call_mcp(STATE_MCP_URL, 'get_active_npc_state', {}, campaign_id, ActiveNpcStateOut)
    if not npc_state.npc_id:
        return ''

    briefing: NpcBriefing = npc_state.briefing or NpcBriefing()
    conv_start_turn_id = npc_state.conv_start_turn_id

    # 2. Get NPC record
    npc_resp = await call_mcp(STATE_MCP_URL, 'get_npc', {'npc_id': npc_state.npc_id}, campaign_id, GetNpcOut)
    npc = npc_resp.npc

    if not npc:
        return ''

    # 3. Preamble: 3 turns before conversation started
    preamble_turns: list = []
    if conv_start_turn_id:
        pre_resp = await call_mcp(
            STATE_MCP_URL, 'get_turns', {'before_turn_id': conv_start_turn_id, 'limit': 3}, campaign_id, GetTurnsOut
        )
        preamble_turns = pre_resp.turns

    # 4. Conversation turns since start
    conv_turns: list = []
    if conv_start_turn_id:
        conv_resp = await call_mcp(
            STATE_MCP_URL, 'get_turns', {'since_turn_id': conv_start_turn_id}, campaign_id, GetTurnsOut
        )
        conv_turns = conv_resp.turns

    preamble_text = '\n'.join(_format_turn(t.role, t.content) for t in preamble_turns) or '(start of session)'
    conv_text = '\n'.join(_format_turn(t.role, t.content) for t in conv_turns)
    if player_message:
        conv_text += f'\n[Player] {player_message}'

    # 5. Build system prompt + LLM call
    system = NPC_SYSTEM.format(
        npc_name=npc.name or 'Unknown',
        npc_role=npc.role,
        voice_instructions=npc.voice_instructions,
        goals=briefing.goals,
        knows=briefing.knows,
        mood=briefing.mood,
        reveal_if=briefing.reveal_if,
        preamble_turns=preamble_text,
        conv_turns=conv_text,
    )
    npc_turn = await call_llm_structured(
        [{'role': 'system', 'content': system}, {'role': 'user', 'content': player_message}],
        NpcTurnResponse,
    )

    # 6. Generate audio
    audio_path: str | None = None
    try:
        speak_resp = await call_mcp(
            MEDIA_MCP_URL,
            'speak',
            {
                'text': npc_turn.npc_speech,
                'voice_id': npc.voice_id,
                'voice_instructions': npc.voice_instructions,
            },
            campaign_id,
            SpeakOut,
            timeout=90,
        )
        audio_path = speak_resp.stream_path or None
    except Exception:
        logger.warning('TTS failed for NPC speech (non-critical)', exc_info=True)

    # 7. Log player + NPC turns
    await call_mcp(STATE_MCP_URL, 'log_turn', {'role': 'player', 'content': player_message}, campaign_id, LogTurnOut)
    await call_mcp(
        STATE_MCP_URL,
        'log_turn',
        {
            'role': 'npc',
            'content': npc_turn.npc_speech,
            'npc_name': npc.name,
            'audio_path': audio_path,
            'metadata': {'audio_path': audio_path},
        },
        campaign_id,
        LogTurnOut,
    )

    # 8. Publish events
    await publish_event(
        campaign_id,
        {
            'type': 'npc_speech',
            'npc_name': npc.name,
            'content': npc_turn.npc_speech,
        },
    )
    if audio_path:
        await publish_event(campaign_id, {'type': 'audio_ready', 'stream_path': audio_path})

    # 9. If done, summarise and clear
    if npc_turn.done:
        conv_transcript = '\n'.join(_format_turn(t.role, t.content) for t in conv_turns)
        conv_transcript += f'\n{_format_turn("player", player_message)}'
        conv_transcript += f'\n{_format_turn("npc", npc_turn.npc_speech)}'
        summary_text: str
        try:
            summary = await call_llm_structured(
                [
                    {'role': 'system', 'content': SUMMARY_SYSTEM},
                    {'role': 'user', 'content': f'Conversation with {npc.name}:\n\n{conv_transcript}'},
                ],
                NpcConversationSummary,
            )
            summary_text = summary.summary
        except Exception:
            logger.warning('NPC conversation summary failed', exc_info=True)
            summary_text = f'Conversation with {npc.name} concluded.'

        await call_mcp(
            STATE_MCP_URL,
            'log_turn',
            {
                'role': 'system',
                'content': summary_text,
                'metadata': {'type': 'npc_conv_summary', 'npc_name': npc.name},
            },
            campaign_id,
            LogTurnOut,
        )
        await call_mcp(STATE_MCP_URL, 'clear_active_npc', {}, campaign_id, OkOut)

        try:
            await call_mcp(
                KNOWLEDGE_MCP_URL, 'update_world', {'narrative_text': conv_transcript}, campaign_id, UpdateWorldOut
            )
        except Exception:
            logger.warning('World update after NPC conversation failed (non-critical)', exc_info=True)

        await publish_event(
            campaign_id,
            {
                'type': 'npc_conversation_ended',
                'npc_name': npc.name,
                'summary': summary_text,
            },
        )

        return summary_text

    return ''
