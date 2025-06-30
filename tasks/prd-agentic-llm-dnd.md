# Agent-Based LLM Dungeon Master Application PRD

## Introduction / Overview
This project aims to deliver an immersive, fully AI-driven tabletop role-playing experience inspired by Dungeons & Dragons.  An agent-based Large Language Model (LLM) system will assume the role of Game Master (GM), structuring narratives, portraying non-player characters (NPCs), and dynamically describing environments.  Players interact primarily through voice; the platform converts speech to text input, generates AI responses, and reads them aloud.  Optional scene illustrations are created on demand by an image-generation model to heighten immersion.

The solution will be exposed through a simple CLI and a web dashboard (via python fasthtml).

## Goals
1. Eliminate human GM preparation time by auto-generating campaigns, NPCs, and scenes.
2. Provide a frictionless **voice-only** play mode (hands-free).
3. Persist campaign state so sessions can be paused and resumed seamlessly.
4. Offer lightweight scene visualization through on-demand AI imagery.

## User Stories
* **US-01** – As a *player*, I can speak to the game and hear AI narration so that I can play without reading text or looking at a screen.
* **US-02** – As a *player*, I can converse with NPCs and receive coherent, character-appropriate replies - and long form conversations with multiple round trips keep conversation context.
* **US-03** – As a *player*, I can view an automatically generated illustration of the current scene to help me visualize the environment.
* **US-04** – As a *returning player*, I can resume a campaign from the exact point we left off, retaining world state and story continuity.
* **US-05** – As a *game facilitator*, I can provide high-level plot hooks or themes at campaign creation so the AI aligns the overarching story with my group's interests.

## Functional Requirements
1. **STT Input** – The system **must** capture player speech and transcribe it via Whisper.
2. **TTS Output** – The system **must** vocalize AI responses using an open-source TTS engine (e.g., Piper/Coqui) and support swappable TTS adapters.
3. **Master Agent** – A persistent "Master Agent" **must** track the global world state, overarching story plan, puzzles, and player history.
4. **Scene Agents** – For each interaction segment (e.g., tavern dialogue, combat), the system **must** spin up a "Scene Agent" that references the Master Agent's state and handles localized conversation.
5. **Context Compression** – The system **must** summarize lengthy histories into compact memory objects (aka. Markdown) to stay within model context windows while preserving continuity.
6. **Campaign Persistence** – The system **must** write world and agent state to human-readable text files (markdown) after each scene and reload them on campaign resume.
7. **Image Generation** – The system **must** request a scene illustration from an image model (Flux[schnell]) when prompted by players or triggered by story beats.
8. **Configurability** – The system **should** allow developers to plug in alternative LLMs, TTS engines, or image models via configuration files or dependency injection.
9.  **Cross-Platform Runtime** – The application **must** run on Windows and Linux.

## Design Considerations
* **Voice-First UX** – Default interaction loop is voice → text → LLM → text → voice.
* **Minimal Latency** – Aim for sub-2-second round-trip response time to maintain conversational flow.
* **Illustration Delivery** – Images can be returned as base64 strings or hosted URLs for external clients.
* **Session UI** – Provide a reference CLI and a lightweight web dashboard for debugging and live text transcript display.

## Technical Considerations
* **Language & Frameworks** – Python 3.11+, leveraging LangChain or similar for agent orchestration.
* **LLM Backends** – Default to:
```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI(
    api_key="GEMINI_API_KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

completion = client.beta.chat.completions.parse(
    model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "John and Susan are going to an AI conference on Friday."},
    ],
    response_format=CalendarEvent,
)

print(completion.choices[0].message.parsed)
```

* **Storage** – Text files per campaign; path configurable - keep plaintext to make integration with LLMs easier and better performant for the LLM completions - I.e. they probably handle plaintext which they generate in any way they want to structure better, than complex JSON or YAML. Maybe add some markdown formatting to structure it slightly better.
* **Audio Pipeline** – Whisper or gemini for STT and some open source API for TTS.
* **Image Generation** - Flux[schnell] via Runware - ensure really detailed prompts and one concrete style shared across all images.
```python
runware = Runware(api_key=RUNWARE_API_KEY)

await runware.connect()

request_image = IImageInference(
  positivePrompt='Concept design for a futuristic motorbike',
  height=1024,
  width=1024,
  model='runware:100@1',
  steps=25,
  CFGScale=4.0,
  numberResults=4,
  outputType='base64Data',
  outputFormat='JPG',
)

images = await runware.imageInference(requestImage=request_image)
output = [image.imageBase64Data for image in images if image.imageBase64Data is not None]
```

## Open Questions
1. Which specific open-source LLM(s) best balance quality vs. hardware cost? - for now Gemini 2.0 flash as it's completely free for now.
2. Do we need multi-player voice diarization to distinguish speakers? - No, but a button on the web UI to select, who is currently speaking.
3. Should we include an optional text chat fallback UI in v1? - No, logs in the console and a web UI (python fasthtml) to play the game.
4. How will content safety / moderation be handled for generated text and images? - not at all.
