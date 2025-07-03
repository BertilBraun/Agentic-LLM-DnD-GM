"""Voice-driven D&D framework using Gemini-2.5-Flash"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Generator, List, Literal, Optional

from pydantic import BaseModel, Field

from stt import WhisperSTT, stt
from tts import BaseTTS, get_tts

from llm import llm_parse, llm_chat

from image import generate_image

from memory import SimpleMemorySystem

from config import LANGUAGE, OPENAI_API_KEY, WHISPER_MODEL

STATE_SAVE_FILE = Path('save/state.json')
STATE_SAVE_FILE.parent.mkdir(parents=True, exist_ok=True)

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
# Character Creation System
# ───────────────────────────────────────────────────────────────
class PlayerCharacter(BaseModel):
    name: str = Field(description='Character name')
    background: str = Field(description='Character background, origin story, and personality')
    class_and_level: str = Field(
        description='Character class, level, and progression (e.g., "Level 3 Wizard", "Novice Fighter")'
    )
    abilities: List[str] = Field(
        description='List of specific abilities, spells, skills, and powers the character possesses'
    )
    equipment: List[str] = Field(description='Weapons, armor, magical items, tools, and other possessions')
    limitations: List[str] = Field(
        description='Character weaknesses, restrictions, moral codes, or things they cannot do'
    )
    power_level: Literal['Novice', 'Apprentice', 'Journeyman', 'Expert', 'Master', 'Legendary'] = Field(
        description='Overall power level to help DM balance encounters and challenges appropriately'
    )
    visual_description: str = Field(
        description='Detailed physical appearance for character portrait generation (300-500 characters)'
    )


def create_character() -> PlayerCharacter:
    """Interactive character creation process"""
    print('=== CHARACTER CREATION ===')
    print("Let's create your D&D character! This will help the DM tailor the adventure to your abilities.\n")

    messages = [
        {
            'role': 'system',
            'content': """You are a helpful D&D character creation assistant. Guide the player through creating a balanced, interesting character by asking focused questions about:

1. Character concept and background
2. Class/profession and experience level  
3. Key abilities and skills they want
4. Equipment and possessions
5. Character limitations and weaknesses
6. Physical appearance

Ask 4-6 questions total. Build on their answers and help them create a character that's both powerful enough to be interesting but balanced enough for good gameplay. Encourage them to include both strengths AND meaningful limitations.

Keep questions conversational and help them think through the implications of their choices.""",
        },
        {
            'role': 'user',
            'content': 'I want to create a character for a D&D campaign. Help me build someone interesting and balanced.',
        },
    ]

    while True:
        q = llm_chat(messages)
        messages.append({'role': 'assistant', 'content': q})
        print(f'[Character Creator] {q}')
        a = input('("done" to finish character) >> ').strip()
        if a.lower() == 'done':
            break
        messages.append({'role': 'user', 'content': a})

    print('\nGenerating character sheet...')

    messages.append(
        {
            'role': 'user',
            'content': """Now create a complete PlayerCharacter based on our discussion. Ensure:

- The character has clear strengths AND meaningful limitations
- Power level matches their described abilities realistically
- Equipment list is appropriate for their background and level
- Abilities are specific enough for the DM to reference during gameplay
- Visual description is detailed enough for portrait generation

Balance is key - even powerful characters should have interesting limitations or challenges.""",
        }
    )

    return llm_parse(messages, PlayerCharacter)


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
    character_context: str = Field(
        description="Summary of how the campaign will be balanced around the player character's abilities and power level"
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
def interactive_planner(character: PlayerCharacter) -> CampaignPlan:
    """Ask questions until the user hits Ctrl-C or says 'done'."""
    campaign_title = input('Enter the campaign title: ')
    messages = [
        {
            'role': 'system',
            'content': f"""You are an expert D&D campaign designer. Your goal is to create an engaging, well-structured campaign tailored to the player character.

PLAYER CHARACTER:
{character.model_dump_json(indent=2)}

The campaign should be balanced around this character's power level ({character.power_level}) and incorporate their abilities, background, and limitations into the story.

Ask 3-5 focused questions to understand:
1. Genre/theme preferences that would complement their character
2. Tone (heroic, gritty, comedic, mystery, etc.)
3. How they want their character's abilities to be challenged
4. Story elements that would engage their character's background
5. Any specific conflicts or character growth they want to explore

Keep questions concise and build on previous answers. Help them create something that showcases their character.""",
        },
        {
            'role': 'user',
            'content': f'I want to create a D&D campaign titled "{campaign_title}" for my character {character.name}. Help me develop this into a full campaign plan.',
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
            'role': 'user',
            'content': f"""Now create a complete CampaignPlan based on our discussion and the character profile. Ensure:

- The campaign is appropriately balanced for a {character.power_level} level character
- Challenges will test their abilities without being impossible
- Story elements connect to their background and motivations
- The character_context explains how encounters/challenges will be scaled
- NPCs and conflicts are appropriate for their power level

Character Power Level: {character.power_level}
Character Abilities: {', '.join(character.abilities)}
Character Limitations: {', '.join(character.limitations)}""",
        }
    )

    return llm_parse(messages, CampaignPlan)


def dm_turn(player_text: str, app_state: AppState) -> DMResponse:
    """DM turn with character-aware validation"""

    prompt = f"""You are an expert Dungeon Master running a D&D campaign. Your role is to create immersive, engaging experiences that respond meaningfully to player actions while respecting character limitations.

===== PLAYER CHARACTER =====
{app_state.character.model_dump_json(indent=2)}

===== CAMPAIGN CONTEXT =====
{app_state.plan.model_dump_json(indent=2)}

===== CURRENT GAME STATE =====
{app_state.get_memory_for_dm()}

===== PLAYER ACTION =====
{player_text}

===== INSTRUCTIONS =====
Before responding, consider:
1. **Can the character actually do this?** Check their abilities, equipment, and limitations
2. **Is this appropriate for their power level?** Scale challenges accordingly
3. **How do their abilities affect the outcome?** Use their specific skills and equipment

Then respond as the DM by:
1. **Validate the action** - If impossible for their character, explain why gently and offer alternatives
2. **Acknowledge the action** - Show how their abilities and choices affect the world
3. **Paint the scene** - Use vivid descriptions appropriate to their power level
4. **Advance the story** - Create challenges that test their specific abilities
5. **Maintain balance** - Ensure encounters match their character's capabilities

**Character Validation Examples:**
- If they try to cast a spell not in their abilities: "Your character doesn't know that spell, but you could try [alternative]"
- If they attempt something beyond their power level: "That would be challenging for a {app_state.character.power_level} character, but you could [scaled approach]"
- If they use a listed ability: Describe it working effectively and having meaningful impact

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
- Always incorporate the campaign's visual_style: {app_state.plan.visual_style}

**Tone:** Match the campaign's tone while respecting character limitations and celebrating their unique abilities."""

    response = llm_parse([{'role': 'user', 'content': prompt}], DMResponse)

    print(f'🎭 DM: {response.gm_speech}')
    return response


def image(description: str, visual_style: str) -> Path:
    prompt = f"""VISUAL STYLE:
{visual_style}

DESCRIPTION:
{description}"""

    print('🎨 Generating image...')
    image = generate_image(prompt)[0]
    print(f'🖼️  Image saved: {image}')
    return image


def npc_loop(npc: NPC, player_input: Callable[[], str], plan: CampaignPlan) -> Generator[UiUpdate, None, str]:
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
        player_text = player_input()
        print(f'Player: {player_text}')
        messages.append({'role': 'user', 'content': player_text})

        npc_msg = llm_parse(messages, NPCMessage)

        print(f'{npc.name}: {npc_msg.npc_speech}')
        tts.play(npc_msg.npc_speech)
        transcript.append(f'**Player:** {player_text}\n**Conversation with {npc.name}:** {npc_msg.npc_speech}')
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
    character: PlayerCharacter
    plan: CampaignPlan
    current_scene_image: Optional[Path]
    current_npc: Optional[NPC]
    memory: SimpleMemorySystem = Field(default_factory=SimpleMemorySystem)

    def save(self):
        STATE_SAVE_FILE.write_text(self.model_dump_json(indent=2), encoding='utf-8')

    @staticmethod
    def load():
        if STATE_SAVE_FILE.exists():
            return AppState.model_validate_json(STATE_SAVE_FILE.read_text(encoding='utf-8'))

        # Create new character and campaign
        character = create_character()
        print(f'\n✨ Character created: {character.name}')
        print(f'🎭 Class: {character.class_and_level}')
        print(f'⚡ Power Level: {character.power_level}')

        plan = interactive_planner(character)
        print(f'\n🏰 Campaign created: {plan.title}')
        print(f'📖 Synopsis: {plan.synopsis}')

        # Generate character portrait
        character_image = image(character.visual_description, f'{plan.visual_style}, character portrait')
        print(f'🖼️ Character portrait: {character_image}')

        app_state = AppState(
            character=character,
            plan=plan,
            current_scene_image=None,
            current_npc=None,
            memory=SimpleMemorySystem(),
        )
        app_state.save()
        return app_state

    def append_memory(self, event: str):
        """Add event and check if compression is needed"""
        self.memory.add_event(event)
        self.save()

    def get_memory_for_dm(self) -> str:
        """Get formatted memory for DM context including character info"""
        acts = '\n'.join([f'**Act {i + 1}:** {act}' for i, act in enumerate(self.plan.acts)])

        character_summary = f"""
## Player Character: {self.character.name}
- **Class/Level:** {self.character.class_and_level}
- **Power Level:** {self.character.power_level}
- **Key Abilities:** {', '.join(self.character.abilities)}
- **Equipment:** {', '.join(self.character.equipment)}
- **Limitations:** {', '.join(self.character.limitations)}
"""

        base_context = f"""# Campaign: {self.plan.title}

## Synopsis
{self.plan.synopsis}

## Character Balance Notes
{self.plan.character_context}

## Story Structure
{acts}

{character_summary}
"""
        return base_context + self.memory.get_full_context()


# ───────────────────────────────────────────────────────────────
# Main Loop
# ───────────────────────────────────────────────────────────────


class UiUpdate(BaseModel):
    history: str
    image: Optional[Path]


def main(player_input: Callable[[], str]) -> Generator[UiUpdate, None, None]:
    """Main game loop with character-aware gameplay"""
    app_state = AppState.load()

    dm_tts = create_tts(voice_id='ash', instructions=DM_TTS_INSTRUCTIONS)

    print(f'\n🎲 Starting your D&D adventure as {app_state.character.name}!')
    print(f'⚡ Power Level: {app_state.character.power_level}')
    print(f'🎭 {app_state.character.class_and_level}')
    print("Speak your actions and the DM will respond based on your character's abilities.\n")

    # Initial scene with character context
    dm_out = dm_turn(
        f'Describe the opening scene for {app_state.character.name}. Set up the initial situation that draws them into the adventure, taking into account their background and current circumstances.',
        app_state,
    )

    if app_state.current_scene_image is None or not app_state.current_scene_image.exists():
        app_state.current_scene_image = image(dm_out.scene_description, app_state.plan.visual_style)

    history = f'**DM:** {dm_out.gm_speech}'
    yield UiUpdate(history=history, image=app_state.current_scene_image)

    dm_tts.play(dm_out.gm_speech)

    try:
        while True:
            # Get player input
            player_text = player_input()

            # DM response with automatic memory management
            dm_out = dm_turn(player_text, app_state)

            # Generate scene image
            # TODO async image?
            app_state.current_scene_image = image(dm_out.scene_description, app_state.plan.visual_style)

            history += f'\n**Player:** {player_text}\n**DM:** {dm_out.gm_speech}'
            yield UiUpdate(history=history, image=app_state.current_scene_image)

            dm_tts.play(dm_out.gm_speech)

            # Add to memory
            memory_entry = f"""**Player:** {player_text}
**DM:** {dm_out.gm_speech}
**Memory Note:** {dm_out.memory_append}"""

            app_state.append_memory(memory_entry)

            # Handle NPC interactions
            if dm_out.npc:
                print(f'\n💬 {dm_out.npc.name} wants to talk with you...')
                interaction_summary = yield from npc_loop(dm_out.npc, player_input, app_state.plan)
                app_state.append_memory(f'**NPC Interaction:** {interaction_summary}')
                print('\n🎭 Back to the main adventure...\n')

                history += f'\nInteraction with {dm_out.npc.name}: {interaction_summary}'

                yield UiUpdate(history=history, image=app_state.current_scene_image)

    finally:
        app_state.save()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', action='store_true')
    args = parser.parse_args()

    if args.cli:
        stt_model = WhisperSTT(model_name=WHISPER_MODEL, language=LANGUAGE)  # Expensive - therefore only load once

        def player_input():
            return stt(stt_model)

        for _ in main(player_input):
            pass

    else:
        from qt import gui_main

        gui_main()
