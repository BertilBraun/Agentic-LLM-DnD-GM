"""Voice-driven D&D framework using Gemini-2.5-Flash"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Generator, List, Literal, Optional

from pydantic import BaseModel, Field

from stt import WhisperSTT, stt
from tts import BaseTTS, get_tts

from llm import llm_parse, llm_chat

from image import generate_image

from config import OPENAI_API_KEY

# TODO allow translation to German
# TODO allow stt from German
# TODO compress the conversation history once it gets too long or a clear cutoff point is reached -> after f.e. going somewhere, where they can't go back and cannot interact with the history anymore (i.e. no other characters are around), then the history can be compressed

STATE_SAVE_FILE = Path('save_state.json')

DM_TTS_INSTRUCTIONS = (
    'Speak in a deep, authoritative voice with dramatic pauses and varied intonation to bring the fantasy world to life'
)


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
# Response models with enhanced field descriptions
# ───────────────────────────────────────────────────────────────
class CampaignPlan(BaseModel):
    title: str = Field(
        description='A compelling campaign title that captures the theme and tone (e.g., "The Shattered Crown of Eldoria")'
    )
    synopsis: str = Field(
        description='A 2-3 sentence synopsis covering the central conflict, stakes, and what makes this campaign unique. Should hook players immediately.'
    )
    acts: List[str] = Field(
        description='3-5 major story acts/chapters, each 1-2 sentences describing key events, locations, or revelations that drive the narrative forward'
    )
    visual_style: str = Field(
        description='Detailed visual style guide (400-600 characters) defining the artistic direction for all generated images: art style (digital painting, fantasy art, etc.), color palette, lighting preferences, atmosphere, level of detail, and any specific visual themes that should be consistent across all scenes'
    )


class NPC(BaseModel):
    name: str = Field(
        description='A memorable NPC name that fits the fantasy setting and hints at their role or personality'
    )
    role: str = Field(
        description='Complete character profile including: personality traits, motivations, knowledge they possess, relationship to the story, specific information or quest they need to convey to players, and any secrets or hidden agendas'
    )
    visual_description: str = Field(
        description='Extremely detailed physical description (300-500 characters) for image generation: race, age, build, facial features, hair, eyes, clothing, armor, weapons, accessories, scars, tattoos, posture, and any distinctive visual elements. Be specific about colors, materials, and styles.'
    )
    voice_id: Literal[
        'alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 'onyx', 'nova', 'sage', 'shimmer', 'verse'
    ] = Field(
        description=f"Select the voice that best matches the NPC's personality and role in the story:\n{OPENAI_VOICE_ID_DESCRIPTIONS}"
    )
    voice_instructions: str = Field(
        description='Detailed voice direction including accent, speech patterns, emotional tone, and mannerisms (e.g., "Speak with a gravelly dwarf accent, slow and deliberate, often pausing to stroke beard")'
    )


class DMResponse(BaseModel):
    gm_speech: str = Field(
        description='Your response as the Dungeon Master. Use vivid descriptions, engage the senses, create atmosphere, and respond directly to player actions. Include narrative exposition, NPC dialogue, environmental details, or consequences of actions as appropriate.'
    )
    scene_description: str = Field(
        description='Extremely detailed visual description (400-800 characters) of the current scene for image generation: location, environment, lighting, weather, objects, creatures, architecture, vegetation, atmosphere, colors, textures, and spatial relationships. Focus on visual elements that would make a compelling image.'
    )
    memory_append: str = Field(
        description='Concise summary of new developments to remember: player decisions, story progression, world changes, or important revelations. Write in past tense as historical facts.'
    )
    npc: Optional[NPC] = Field(
        description='Only include if the scene introduces a new NPC that players will interact with directly. Leave empty for background characters or if no new NPCs appear.'
    )


class NPCMessage(BaseModel):
    npc_speech: str = Field(
        description="What the NPC says in response to the player. Stay in character, advance the conversation naturally, and include relevant information or reactions based on the NPC's personality and goals."
    )
    done: bool = Field(
        description='Set to true when the conversation has reached a natural conclusion, the NPC has shared their key information, or they would realistically end the interaction.'
    )


class ConversationSummary(BaseModel):
    summary: str = Field(
        description='Bullet-point summary of key information the GM needs to track: what the player learned, decisions made, relationships formed/damaged, quest progress, and any promises or commitments made by either party.'
    )


# ───────────────────────────────────────────────────────────────
# Agents with enhanced prompts
# ───────────────────────────────────────────────────────────────
def interactive_planner() -> CampaignPlan:
    """Ask questions until the user hits Ctrl-C or says 'done'."""
    campaign_title = input('Enter the campaign title: ')
    messages = [
        {
            'role': 'system',
            'content': """You are an expert D&D campaign designer. Your goal is to create an engaging, well-structured campaign through targeted questions.

Ask 3-5 focused questions to understand:
1. Genre/theme preferences (high fantasy, dark fantasy, political intrigue, etc.)
2. Tone (heroic, gritty, comedic, mystery, etc.)
3. Key story elements they want to include
4. Campaign length/scope preferences
5. Any specific settings, conflicts, or character types they're excited about

Keep questions concise and build on previous answers. Help them create something they'll be excited to run.""",
        },
        {
            'role': 'user',
            'content': f'I want to create a D&D campaign titled "{campaign_title}". Help me develop this into a full campaign plan.',
        },
    ]

    while True:
        q = llm_chat(messages)
        messages.append({'role': 'assistant', 'content': q})
        print(f'[Campaign Designer] {q}')
        a = input('("done" to finish planning) >> ').strip()
        if a.lower() == 'done':
            break
        messages.append({'role': 'user', 'content': a})

    print('\nGenerating full campaign plan…')

    messages.append(
        {
            'role': 'system',
            'content': """Now create a complete CampaignPlan based on our discussion. Ensure:

- The title captures the campaign's essence
- The synopsis establishes clear stakes and hooks
- Each act builds logically toward a satisfying conclusion
- The plan provides enough structure for improvisation
- Story elements incorporate the user's preferences

Focus on creating memorable moments and meaningful player choices.""",
        }
    )

    return llm_parse(messages, CampaignPlan)


def dm_turn(player_text: str, plan: CampaignPlan, memory: str) -> DMResponse:
    prompt = f"""You are an expert Dungeon Master running a D&D campaign. Your role is to create immersive, engaging experiences that respond meaningfully to player actions.

===== CAMPAIGN CONTEXT =====
{plan.model_dump_json(indent=2)}

===== CURRENT GAME STATE =====
{memory}

===== PLAYER ACTION =====
{player_text}

===== INSTRUCTIONS =====
Respond as the DM by:

1. **Acknowledge the player's action** - Show how their choice affects the world
2. **Paint the scene** - Use vivid, sensory descriptions to bring the world to life
3. **Advance the story** - Move the narrative forward while staying true to the campaign plan
4. **Create meaningful choices** - Present opportunities for player agency and decision-making
5. **Maintain consistency** - Respect established world rules and previous events
6. **Describe the scene visually** - Provide extremely detailed visual description for image generation

**Guidelines:**
- Use "you" to address the player directly
- Describe consequences of actions clearly
- Include environmental details and atmosphere
- Introduce complications or new information as appropriate
- If introducing an NPC for direct interaction, include them in the response
- Keep responses engaging but not overwhelming (2-4 sentences typically)
- For scene_description: Focus on visual elements that would create a compelling, detailed image

**Visual Description Requirements:**
- Include specific details about location, lighting, objects, creatures, architecture
- Mention colors, textures, spatial relationships, weather, atmosphere
- Be extremely specific and detailed (400-800 characters)
- Always incorporate the campaign's visual_style: {plan.visual_style}

**Tone:** Match the campaign's established tone while maintaining dramatic tension and player engagement."""

    response = llm_parse([{'role': 'user', 'content': prompt}], DMResponse)
    print(f'🎭 DM: {response.gm_speech}')
    return response


def image_prompt(description: str, visual_style: str) -> str:
    return f"""VISUAL STYLE:
{visual_style}

DESCRIPTION:
{description}"""


def image(description: str, visual_style: str) -> Path:
    prompt = image_prompt(description, visual_style)
    print('🎨 Generating image...')
    image = generate_image(prompt)[0]
    print(f'🖼️  Image saved: {image}')
    return image


def npc_loop(npc: NPC, stt_model: WhisperSTT, plan: CampaignPlan) -> Generator[UiUpdate, None, str]:
    # Generate NPC portrait at start of conversation
    npc_image = image(npc.visual_description, f'{plan.visual_style}, character portrait')
    yield UiUpdate(history='', image=npc_image)

    tts = create_tts(voice_id=npc.voice_id, instructions=npc.voice_instructions)

    messages: list[dict[str, str]] = [
        {
            'role': 'system',
            'content': f"""You are {npc.name}, a character in a D&D campaign. 

CHARACTER PROFILE:
{npc.role}

ROLEPLAY INSTRUCTIONS:
- Stay completely in character at all times
- Respond naturally to what the player says
- Share information gradually and believably
- React according to your personality and motivations
- The conversation should feel organic, not forced
- End the conversation when it reaches a natural conclusion or when you've accomplished your story purpose
- Don't break character to ask out-of-character questions

Remember: You are having a real conversation with an adventurer in your world. Respond as {npc.name} would, with their voice, mannerisms, and agenda.""",
        }
    ]
    transcript: list[str] = []

    while True:
        print(f'🎙️  Speak with {npc.name}: ')
        player_input = stt(stt_model)
        print(f'Player: {player_input}')
        messages.append({'role': 'user', 'content': player_input})

        npc_msg = llm_parse(messages, NPCMessage)

        print(f'{npc.name}: {npc_msg.npc_speech}')
        tts.play(npc_msg.npc_speech)
        transcript.append(f'Player: {player_input}\n{npc.name}: {npc_msg.npc_speech}')
        messages.append({'role': 'assistant', 'content': npc_msg.npc_speech})

        yield UiUpdate(history='\n\n'.join(transcript), image=npc_image)

        if npc_msg.done:
            break

    # Summarize the conversation for the GM
    summary_prompt = [
        {
            'role': 'system',
            'content': """Analyze this NPC conversation and extract the key information the GM needs to track for future sessions.

Focus on:
- Important information revealed or learned
- Relationship changes or developments  
- Commitments, promises, or agreements made
- Quest updates or new objectives
- Character motivations or secrets revealed
- Any consequences that should affect future gameplay

Format as clear, actionable bullet points the GM can reference later.""",
        },
        {'role': 'user', 'content': f'Conversation with {npc.name}:\n\n' + '\n\n'.join(transcript)},
    ]

    summary = llm_parse(summary_prompt, ConversationSummary)
    return f'\n**Conversation with {npc.name}:**\n{summary.summary}\n'


# ───────────────────────────────────────────────────────────────
# Main loop
# ───────────────────────────────────────────────────────────────


class AppState(BaseModel):
    plan: Optional[CampaignPlan]
    current_scene_image: Optional[Path]
    current_npc: Optional[NPC]
    memory: str

    def save(self):
        STATE_SAVE_FILE.write_text(self.model_dump_json(indent=2))

    @staticmethod
    def load():
        if not STATE_SAVE_FILE.exists():
            return AppState(plan=None, current_scene_image=None, current_npc=None, memory='')
        return AppState.model_validate_json(STATE_SAVE_FILE.read_text())

    def append_memory(self, chunk: str):
        self.memory += chunk.strip() + '\n\n'
        self.save()

    def init_campaign(self):
        self.plan = interactive_planner()
        print(f'\n✨ Campaign created: {self.plan.title}')
        print(f'📖 Synopsis: {self.plan.synopsis}')
        self.append_memory(
            f'# Campaign: {self.plan.title}\n\n## Synopsis\n{self.plan.synopsis}\n\n## Story Structure\n'
            + '\n'.join([f'**Act {i + 1}:** {act}' for i, act in enumerate(self.plan.acts)])
            + '\n'
        )
        self.save()


class UiUpdate(BaseModel):
    history: str
    image: Optional[Path]


def main(player_input: Callable[[], str]) -> Generator[UiUpdate, None, None]:
    app_state = AppState.load()

    # Draft a campaign the first time we run
    if app_state.plan is None:
        app_state.init_campaign()

    assert app_state.plan is not None

    stt_model = WhisperSTT(model_name='base')
    dm_tts = create_tts(voice_id='ash', instructions=DM_TTS_INSTRUCTIONS)

    print('\n🎲 Starting your D&D adventure! Speak your actions and the DM will respond.\n')

    dm_out = dm_turn(
        "Describe the current situation, since the players are just arriving in the world and don't know what's going on",
        app_state.plan,
        app_state.memory,
    )
    if app_state.current_scene_image is None:
        app_state.current_scene_image = image(dm_out.scene_description, app_state.plan.visual_style)

    history = f'**DM:** {dm_out.gm_speech}'
    yield UiUpdate(history=history, image=app_state.current_scene_image)

    dm_tts.play(dm_out.gm_speech)

    try:
        while True:
            # Collect player speech
            player_text = player_input()
            # DM step
            dm_out = dm_turn(player_text, app_state.plan, app_state.memory)

            # Generate scene image
            app_state.current_scene_image = image(dm_out.scene_description, app_state.plan.visual_style)

            history += f'\n**Player:** {player_text}\n**DM:** {dm_out.gm_speech}'
            yield UiUpdate(history=history, image=app_state.current_scene_image)

            dm_tts.play(dm_out.gm_speech)

            app_state.append_memory(
                f"""
**Player:** {player_text}
**DM:** {dm_out.gm_speech}
**Update:** {dm_out.memory_append}
"""
            )
            # Spawn NPC if ordered
            if dm_out.npc:
                print(f'\n💬 {dm_out.npc.name} wants to talk with you...')
                interaction_summary = yield from npc_loop(dm_out.npc, stt_model, app_state.plan)
                app_state.append_memory(interaction_summary)
                print('\n🎭 Back to the main adventure...\n')
    finally:
        app_state.save()


if __name__ == '__main__':
    stt_model = WhisperSTT(model_name='base')

    def player_input():
        return stt(stt_model)

    main(player_input)
