"""Voice-driven D&D framework using Gemini-2.5-Flash"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from audio.stt import WhisperSTT, stt
from audio.tts import BaseTTS, get_tts

from llm import llm_parse, llm_chat

from config import OPENAI_API_KEY

PLAN_FILE = Path('plan.json')

DM_TTS_INSTRUCTIONS = 'Speak in a deep, authoritative voice'
OPENAI_VOICE_ID_DESCRIPTIONS = """Alloy: Confident, warm, and energetic with a friendly tone. Versatile for expressive, conversational tasks.
Ash: Calm, neutral, with a soothing delivery. Suitable for professional or instructional content.
Ballad: Melodic, clear, slightly artistic tone. Good for storytelling or engaging narratives.
Coral: Soft, gentle, approachable voice. Works well for empathetic or supportive dialogue.
Echo: Bright, crisp, and articulate. Great for clear explanations or lively interactions.
Fable: Playful, imaginative, with storytelling flair. Ideal for creative content or children's stories.
Onyx: Deep, rich, and authoritative. Best for formal, confident presentations or strong delivery.
Nova: Youthful, energetic, with a friendly demeanor. Suitable for casual, modern conversations.
Sage: Wise, thoughtful, and composed. Good for advisory, reflective, or informative tones.
Shimmer: Sparkling, enthusiastic, expressive tone. Ideal for dynamic, animated content.
Verse: Rhythmic, poetic, and smooth. Excellent for narration or creative, flowing dialogue."""


def create_tts(voice_id: str, instructions: str) -> BaseTTS:
    return get_tts(
        engine='openai',
        voice_model='gpt-4o-mini-tts',
        voice_id=voice_id,
        api_key=OPENAI_API_KEY,
        instructions=instructions,
    )


# ───────────────────────────────────────────────────────────────
# Response models
# ───────────────────────────────────────────────────────────────
class CampaignPlan(BaseModel):
    title: str = Field(description='the title of the campaign')
    synopsis: str = Field(description='a short synopsis of the campaign')
    acts: List[str] = Field(description='a list of acts in the campaign')


class NPC(BaseModel):
    name: str = Field(description='the name of the NPC')
    role: str = Field(
        description='full character description including personality, knowledge, task to convey to the players, etc.'
    )
    voice_id: Literal[
        'alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 'onyx', 'nova', 'sage', 'shimmer', 'verse'
    ] = Field(description=f'the voice id of the NPC.\n{OPENAI_VOICE_ID_DESCRIPTIONS}')
    voice_instructions: str = Field(description='the voice instructions for how the NPC should speak')


class DMResponse(BaseModel):
    gm_speech: str = Field(
        description='What the Dungeon Master says to the players'
    )  # could include voice instructions like "speak in a deep, authoritative voice"
    memory_append: str = Field(description='plain-text chunk to tack onto DM memory')
    npc: Optional[NPC] = Field(description='if this is present, spawn that NPC')


class NPCMessage(BaseModel):
    npc_speech: str = Field(description='What the NPC says to the player')
    done: bool = Field(description='True → conversation over')


class ConversationSummary(BaseModel):
    summary: str = Field(description='what the GM needs to remember')


# ───────────────────────────────────────────────────────────────
# Memory helpers (plain-text file)
# ───────────────────────────────────────────────────────────────
MEM_FILE = Path('memory.txt')
if not MEM_FILE.exists():
    MEM_FILE.write_text('=== Campaign Memory ===\n')


def read_memory() -> str:
    with open(MEM_FILE, 'r', encoding='utf-8') as f:
        return f.read()


def append_memory(chunk: str):
    with open(MEM_FILE, 'a', encoding='utf-8') as f:
        f.write(chunk.strip() + '\n\n')


# ───────────────────────────────────────────────────────────────
# Agents
# ───────────────────────────────────────────────────────────────
def interactive_planner() -> CampaignPlan:
    """Ask questions until the user hits Ctrl-C or says 'done'."""
    campaign_title = input('Enter the campaign title: ')
    messages = [
        {
            'role': 'system',
            'content': 'You are a campaign-planning assistant. Ask concise questions until told to stop.',
        },
        {
            'role': 'user',
            'content': f'The campaign title is "{campaign_title}".',
        },
    ]
    while True:
        q = llm_chat(messages)
        messages.append({'role': 'assistant', 'content': q})
        print(f'[Planner] {q}')
        a = input('("done" to finish planning) >> ').strip()
        if a.lower() == 'done':
            break
        messages.append({'role': 'user', 'content': a})

    print('\nGenerating full campaign …')

    messages.append({'role': 'system', 'content': 'Using everything above, output the CampaignPlan.'})
    return llm_parse(messages, CampaignPlan)


def dm_turn(player_text: str, plan: CampaignPlan) -> DMResponse:
    prompt = (
        'You are the Dungeon Master.\n'
        '===== CAMPAIGN PLAN =====\n'
        f'{plan.model_dump_json(indent=2)}\n\n'
        '===== MEMORY =====\n'
        f'{read_memory()}\n\n'
        f'===== PLAYER SAYS =====\n{player_text}'
    )
    return llm_parse([{'role': 'user', 'content': prompt}], DMResponse)


def npc_loop(npc: NPC, stt_model: WhisperSTT) -> None:
    tts = create_tts(voice_id=npc.voice_id, instructions=npc.voice_instructions)

    messages: list[dict[str, str]] = [
        {'role': 'system', 'content': f'You are {npc.name}, {npc.role}. Stay in character and end naturally.'}
    ]
    transcript: list[str] = []

    while True:
        print(f'🎙️  Say something to {npc.name}: ')
        player_input = stt(stt_model)
        print(f'Player: {player_input}')
        messages.append({'role': 'user', 'content': player_input})
        npc_msg = llm_parse(messages, NPCMessage)

        print(f'NPC {npc.name}: {npc_msg.npc_speech}')
        tts.play(npc_msg.npc_speech)
        transcript.append(f'Player: {player_input}\n\n{npc.name}: {npc_msg.npc_speech}')
        messages.append({'role': 'assistant', 'content': npc_msg.npc_speech})

        if npc_msg.done:
            break

    # Summarise the finished conversation
    summary_prompt = [
        {'role': 'system', 'content': 'Summarise the salient information for the GM.'},
        {'role': 'user', 'content': '\n'.join(transcript)},
    ]
    summary = llm_parse(summary_prompt, ConversationSummary)
    append_memory(f'*Conversation with {npc.name}:*\n{summary.summary}')


# ───────────────────────────────────────────────────────────────
# Main loop
# ───────────────────────────────────────────────────────────────
def main():
    # Draft a campaign the first time we run
    if not PLAN_FILE.exists():
        plan = interactive_planner()
        print(f'Created campaign plan: {plan}')
        PLAN_FILE.write_text(plan.model_dump_json(indent=2))
        append_memory(f'# Campaign: {plan.title}\n## Synopsis\n{plan.synopsis}\n')
    else:
        plan = CampaignPlan.model_validate_json(PLAN_FILE.read_text())

    stt_model = WhisperSTT(model_name='base')
    dm_tts = create_tts(voice_id='alloy', instructions=DM_TTS_INSTRUCTIONS)

    while True:
        # Collect player speech
        print('🎙️  Your move …')
        player_text = stt(stt_model)
        print(f'Player: {player_text}')

        # DM step
        dm_out = dm_turn(player_text, plan)
        print(f'DM: {dm_out.gm_speech}')
        append_memory(dm_out.memory_append)
        dm_tts.play(dm_out.gm_speech)

        # Spawn NPC if ordered
        if dm_out.npc:
            npc_loop(dm_out.npc, stt_model)


if __name__ == '__main__':
    main()
