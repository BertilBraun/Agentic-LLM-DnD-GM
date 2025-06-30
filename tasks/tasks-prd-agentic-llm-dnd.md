## Relevant Files

- `src/audio/stt.py` – Real-time speech-to-text wrapper using Whisper.
- `src/audio/tts.py` – Text-to-speech adapters supporting multiple engines (Coqui implemented, Piper stub).
- `src/agents/base_agent.py` – Abstract base class defining common agent interfaces.
- `src/agents/master_agent.py` – Persistent Master Agent managing global world state and story plan.
- `src/agents/scene_agent.py` – Per-scene agent handling localized interactions, spun up on demand.
- `src/agents/langchain_adapter.py` – LangChain orchestration utilities linking LLMs and agent memory.
- `src/context/compression.py` – Utilities for summarising long histories into compact memory objects (implemented).
- `src/persistence/storage.py` – Serialize/deserialize campaign state to markdown files and reload on resume (implemented).
- `src/image/generator.py` – Flux[schnell] Runware synchronous client (implemented).
- `src/image/prompt_builder.py` – Enforces consistent art style and builds detailed prompts.
- `src/ui/cli.py` – CLI interface driving the voice-first gameplay loop.
- `src/ui/dashboard.py` – fasthtml-based web dashboard for voice input, speaker selection, live transcripts, and TTS playback.
- `src/ui/logger_cli.py` – Lightweight CLI that streams logs for debugging purposes (implemented).
- `docs/whisper.md` – Reference notes & examples for Whisper STT.
- `docs/tts.md` – Research notes for chosen TTS engine(s).
- `docs/openai.md` – Guidelines and examples for OpenAI LLM usage.
- `docs/fastapi.md` – Quick reference for FastAPI (internal tooling, if required).
- `docs/fasthtml.md` – Up-to-date documentation and snippets for fasthtml UI.
- `docs/runware.md` – Notes and examples for Runware Flux[schnell] synchronous API.
- `docs/campaign_save_schema.md` – Markdown schema for campaign save files.
- `requirements.txt` – Python dependencies including Whisper STT and TTS engine libraries.
- `src/main.py` – Application entrypoint with resume logic.

### Notes

- Documentation markdown files under `docs/` will collect key usage patterns, code snippets, and configuration tips discovered during web research.
- Testing tasks and directories are intentionally deferred for now and may be added in a later phase.

## Tasks

- [x] 1.0 Voice Input & Output Pipeline (STT & TTS)
  - [x] 1.1 Select and install Whisper and preferred TTS engine libraries.
  - [x] 1.2 Implement `src/audio/stt.py` to stream microphone input to Whisper and return text.
  - [x] 1.3 Implement `src/audio/tts.py` with a pluggable adapter pattern (start with Piper/Coqui implementation).
  - [x] 1.4 Add configuration options to switch TTS engines at runtime.

- [x] 2.0 Core Agent Architecture (Master & Scene Agents, Context Compression)
  - [x] 2.1 Define `BaseAgent` interface with common methods (`prompt()`, `update_memory()`, etc.).
  - [x] 2.2 Implement `MasterAgent` to maintain world state and overarching story plan.
  - [x] 2.3 Implement `SceneAgent` factory that instantiates per interaction segment referencing `MasterAgent` state.
  - [x] 2.4 Implement context summarisation utilities in `context/compression.py` to create compact memory objects.
  - [x] 2.5 Integrate LangChain (or alternative) to orchestrate agent prompting and memory retrieval.

- [x] 3.0 Campaign State Persistence & Resume
  - [x] 3.1 Design markdown schema for campaign save files (world state, agents, scene history).
  - [x] 3.2 Implement `persistence/storage.py` to serialize state after each scene and load on resume.
  - [x] 3.3 Wire persistence hooks into the `MasterAgent` lifecycle to auto-save after each scene.
  - [x] 3.4 Implement resume flow in `src/main.py` to detect existing save and restore agents/state at startup.

- [x] 4.0 Image Generation Service Integration
  - [x] 4.1 Build `image/generator.py` wrapper for the Runware Flux[schnell] API with a synchronous interface.
  - [x] 4.2 Create prompt builder enforcing a consistent art style and detailed scene descriptions.
  - [x] 4.3 Expose configuration options for image parameters (model, resolution, steps, CFGScale).
  - [x] 4.4 Return generated images as base64 strings to the UI, with robust error handling and retries.

- [x] 5.0 Web User Interface (Primary) & CLI Logging
  - [x] 5.1 Build fasthtml `ui/dashboard.py` with:
    - Browser-based microphone capture and streaming to STT backend.
    - Current speaker selection (button or dropdown).
    - Display of live transcripts and scene images.
    - Playback of TTS audio responses.
  - [x] 5.2 Implement `ui/logger_cli.py` to stream real-time logs for debugging (no interaction loop).
  - [ ] 5.3 Update documentation (`README.md`) with setup instructions and usage guide.

- [ ] 6.0 Documentation & Web Research
  - [ ] 6.0.0 Perform comprehensive web searches to locate the latest official documentation and community examples for each technology listed below.
  - [ ] 6.1 Review latest Whisper STT documentation; summarize key usage patterns in `docs/whisper.md`.
  - [ ] 6.2 Research current TTS engines (Piper, Coqui, etc.); capture install and usage notes in `docs/tts.md`.
  - [ ] 6.3 Gather up-to-date OpenAI Python API references and examples; document in `docs/openai.md`.
  - [ ] 6.4 Collect FastAPI quick-start guidelines (for potential internal tooling) in `docs/fastapi.md`.
  - [ ] 6.5 Research fasthtml usage, especially handling of media input/output; record findings in `docs/fasthtml.md`.
  - [ ] 6.6 Study Runware Flux[schnell] synchronous API details; add notes and example calls to `docs/runware.md`.  