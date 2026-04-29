# Agentic D&D System — Architecture Plan

**Version:** 2.0
**Date:** 2026-04-27
**Status:** Approved for implementation

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Structure](#2-repository-structure)
3. [Data Models](#3-data-models)
4. [MCP Servers](#4-mcp-servers)
5. [Agent Specifications](#5-agent-specifications)
6. [API Layer](#6-api-layer)
7. [Frontend](#7-frontend)
8. [Turn Flows](#8-turn-flows)
9. [Docker Compose](#9-docker-compose)
10. [Environment Variables](#10-environment-variables)
11. [Resolved Decisions](#11-resolved-decisions)

---

## 1. Overview

### What the System Does

The system is a voice-driven, multi-session AI Dungeon Master. Authenticated users create campaigns, design player characters through a guided setup wizard, then play a voice-interactive D&D game. The DM responds to spoken player input with narration audio, a generated scene image, and optional NPC sub-conversations. All campaigns are isolated by a `campaign_id` that flows transparently through every layer.

### Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| LLM | Google Gemini 2.5 Flash (`google-genai` SDK) | High quality, low cost, native structured output |
| Image generation | Google Gemini (`gemini-2.5-flash-preview-image-generation`) | Same API key as LLM; swap to Runware/DALL-E via `image-service` |
| TTS | OpenAI TTS (`gpt-4o-mini-tts`) | Per-character voice instructions; swap via `tts-service` |
| STT | OpenAI Whisper (local model, `small`) | File-based, no latency dependency; swap via `stt-service` |
| Provider services | `llm-service`, `image-service`, `tts-service`, `stt-service` (FastAPI) | Adapter layer: token counting, caching, logging, swappable backends — one service per concern |
| Agent runtime | Google A2A protocol — five separate FastAPI microservices (`character-creator :8010`, `campaign-designer :8011`, `dm-agent :8012`, `npc-agent :8013`, `memory-agent :8014`) | Independent scalability, clear separation of concerns; `api` dispatches via A2A JSON-RPC |
| Agent-tool protocol | MCP (Model Context Protocol) over HTTP | Standardised tool calling, easy to add servers later |
| Embeddings | Google `gemini-embedding-2` (`google-genai` SDK, 1536-dim, auto-normalized) | Same API key as LLM; no local model to download or maintain |
| API | FastAPI (Python 3.12) | Async-native, SSE support, fast iteration |
| Frontend | React 18 SPA (TypeScript, Vite) | Modern, straightforward SSE + audio APIs |
| Primary DB | PostgreSQL 16 | Relational structured state |
| Vector store | Qdrant | Semantic memory / RAG |
| Graph store | Neo4j 5 | World knowledge: NPC/Location/Faction graph |
| Pub/sub bridge | Redis 7 | SSE fan-out between FastAPI and clients |
| Auth | JWT (HS256, access token only) | Stateless, simple; httpOnly cookie for SSE/media auth |
| Containerisation | Docker + Docker Compose | All services including MCP servers and provider services |
| Media storage | Docker named volume `media_data` | Shared between `api`, `tts-service`, and `stt-service` via `media-mcp` |

### High-Level Architecture Diagram

```
Browser (React SPA)
        | HTTPS REST + SSE
        v
+----------------------------------------------------------+
|                   FastAPI  (api)  :8000                   |
|  JWT Auth      SSE Publisher --> Redis :6379              |
|  A2A Dispatcher (reads routing state, sends A2A tasks)    |
+----------------------------------------------------------+
        | A2A protocol (JSON-RPC 2.0 over HTTP)
        v
+------------------+ +-------------------+ +----------+ +----------+ +--------------+
| character-creator| | campaign-designer | | dm-agent | | npc-agent| | memory-agent |
|      :8010       | |      :8011        | |  :8012   | |  :8013   | |    :8014     |
+------------------+ +-------------------+ +----------+ +----------+ +--------------+
        |         (all agents call MCP servers with X-Campaign-ID header)
        v
+------------+  +------------+  +---------------+  +------------+
| state-mcp  |  | memory-mcp |  | knowledge-mcp |  | media-mcp  |
|   :8001    |  |   :8002    |  |    :8003      |  |   :8004    |
+-----+------+  +-----+------+  +------+--------+  +-----+------+
      |               |                |
      v               v                v
 Postgres          Qdrant           Neo4j
 (+ STM/LTM        (gemini-          :7687
  memory cols)      embedding-2,
                    1536-dim)
        | HTTP  (internal provider calls)
        v
+---------------+  +---------------+  +---------------+  +---------------+
|  llm-service  |  | image-service |  |  tts-service  |  |  stt-service  |
|    :9001      |  |    :9002      |  |    :9003      |  |    :9004      |
|  (adapter +   |  |  (adapter +   |  |  (adapter +   |  |  (adapter +   |
|   caching +   |  |   caching +   |  |   caching +   |  |   loading +   |
|   logging)    |  |   logging)    |  |   logging)    |  |   logging)    |
+-------+-------+  +-------+-------+  +-------+-------+  +-------+-------+
        |                  |                  |                  |
        v                  v                  v                  v
  Gemini API          Gemini Image       OpenAI TTS         Whisper STT
  (text gen)          API (image gen)    [swap: ElevenLabs, (local model)
  [swap: Anthropic]   [swap: Runware,     Gemini Audio]     [swap: OpenAI,
                       DALL-E]                               Gemini Audio]
```

---

## 2. Repository Structure

```
llm-dnd/
├── .env.example                          # All required env vars with descriptions
├── .gitignore
├── docker-compose.yml                    # Full service orchestration
├── README.md
│
├── docs/
│   └── architecture.md                   # This document
│
├── api/                                  # FastAPI: auth, SSE, A2A dispatcher (NO agent logic)
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── app/
│       ├── __init__.py
│       ├── main.py                       # FastAPI app factory, lifespan, router mounts
│       ├── config.py                     # Settings loaded from env (pydantic-settings)
│       ├── database.py                   # Async SQLAlchemy engine + session factory
│       ├── redis_client.py               # Redis connection singleton
│       ├── a2a_client.py                 # Thin HTTP client: sends JSON-RPC tasks/send to agent services
│       ├── dispatcher.py                 # Reads routing state, dispatches to correct A2A agent URL
│       │
│       ├── auth/
│       │   ├── __init__.py
│       │   ├── models.py                 # User SQLAlchemy ORM model
│       │   ├── schemas.py                # Pydantic schemas: RegisterRequest, LoginRequest, TokenResponse
│       │   ├── router.py                 # POST /auth/register, POST /auth/login
│       │   ├── service.py                # Password hashing (bcrypt), JWT encode/decode
│       │   └── dependencies.py           # get_current_user FastAPI dependency
│       │
│       ├── campaigns/
│       │   ├── __init__.py
│       │   ├── models.py                 # Campaign, Character, NPC ORM models
│       │   ├── schemas.py                # Pydantic schemas for request/response
│       │   ├── router.py                 # Campaign CRUD endpoints
│       │   └── service.py                # Campaign business logic
│       │
│       ├── game/
│       │   ├── __init__.py
│       │   ├── router.py                 # POST /game/message, GET /game/stream, POST /game/audio
│       │   ├── service.py                # Resolves campaign_id, calls dispatcher.py
│       │   └── schemas.py                # PlayerMessage, GameEvent schemas
│       │
│       └── sse/
│           ├── __init__.py
│           └── publisher.py              # publish_event(campaign_id, event) -> Redis PUBLISH
│
├── agents/                               # Five A2A microservices
│   ├── shared/
│   │   ├── a2a.py                        # AgentCard model, JSON-RPC task schemas, base A2A router
│   │   └── mcp_client.py                 # Thin HTTP client: POST /tools/{name} + X-Campaign-ID header
│   │
│   ├── character-creator/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── app/
│   │       ├── main.py                   # FastAPI: GET /.well-known/agent.json, POST / (JSON-RPC)
│   │       └── agent.py                  # CharacterCreatorAgent step-by-step logic
│   │
│   ├── campaign-designer/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── app/
│   │       ├── main.py
│   │       └── agent.py                  # CampaignDesignerAgent step-by-step logic
│   │
│   ├── dm-agent/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── app/
│   │       ├── main.py
│   │       └── agent.py                  # DMAgent — calls memory-agent (A2A), not memory-mcp directly
│   │
│   ├── npc-agent/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── app/
│   │       ├── main.py
│   │       └── agent.py                  # NPCAgent step-by-step logic
│   │
│   └── memory-agent/
│       ├── Dockerfile
│       ├── pyproject.toml
│       └── app/
│           ├── main.py
│           └── agent.py                  # SimpleMemorySystem logic (CutoffDecision + MemoryCompression)
│
├── mcp-servers/
│   ├── shared/
│   │   ├── middleware.py                 # CampaignIDMiddleware: reads X-Campaign-ID header
│   │   └── db.py                         # Shared async DB helpers used by MCP servers
│   │
│   ├── state-mcp/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── app/
│   │       ├── __init__.py
│   │       ├── main.py                   # FastAPI app, mounts all tool routes
│   │       ├── config.py                 # DB URL, port
│   │       ├── tools/
│   │       │   ├── campaign.py           # create_campaign, save_campaign_plan, set_phase
│   │       │   ├── character.py          # save_character, get_campaign_context
│   │       │   ├── memory.py             # get_memory, update_memory
│   │       │   ├── npc.py                # save_npc, get_npc, list_npcs, set_active_npc, clear_active_npc
│   │       │   └── turns.py              # log_turn, get_turns, get_routing_state
│   │       └── schemas.py                # All input/output Pydantic models for state-mcp
│   │
│   ├── memory-mcp/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── app/
│   │       ├── __init__.py
│   │       ├── main.py                   # FastAPI app
│   │       ├── config.py                 # Qdrant URL, Gemini API key
│   │       ├── embedder.py               # Text -> vector (gemini-embedding-2 via google-genai)
│   │       └── tools/
│   │           └── memory.py             # store, recall
│   │
│   ├── knowledge-mcp/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── app/
│   │       ├── __init__.py
│   │       ├── main.py
│   │       ├── config.py                 # Neo4j URI, LLM service URL
│   │       ├── extractor.py              # LLM call to extract entities/relationships from text
│   │       └── tools/
│   │           └── knowledge.py          # update_world, get_world_context
│   │
│   └── media-mcp/
│       ├── Dockerfile
│       ├── pyproject.toml
│       └── app/
│           ├── __init__.py
│           ├── main.py
│           ├── config.py                 # API keys, media volume path
│           └── tools/
│               ├── image.py              # generate_image (calls image-service)
│               ├── tts.py                # speak (calls tts-service)
│               └── stt.py                # transcribe (calls stt-service)
│
└── frontend/
    ├── Dockerfile
    ├── package.json
    ├── vite.config.ts
    ├── tsconfig.json
    ├── index.html
    └── src/
        ├── main.tsx                      # React root, router setup
        ├── api.ts                        # Typed fetch wrappers for all API endpoints
        ├── auth.ts                       # JWT storage, cookie handling
        ├── sse.ts                        # SSE hook: useSSE(campaignId)
        ├── audioRecorder.ts              # MediaRecorder wrapper -> WAV blob -> upload
        ├── store/
        │   ├── index.ts                  # Zustand store root
        │   ├── authStore.ts              # user, tokens, login/logout actions
        │   └── gameStore.ts              # messages, currentImage, currentAudio, phase
        ├── pages/
        │   ├── LoginPage.tsx             # /login
        │   ├── DashboardPage.tsx         # /dashboard
        │   ├── SetupPage.tsx             # /campaigns/:id/setup
        │   └── PlayPage.tsx              # /campaigns/:id/play
        └── components/
            ├── ProtectedRoute.tsx        # Redirects to /login if no token
            ├── CampaignCard.tsx          # Dashboard campaign list item
            ├── SetupWizard/
            │   ├── SetupWizard.tsx       # Orchestrates character + campaign steps
            │   ├── CharacterStep.tsx     # Chat UI for character creation
            │   └── CampaignStep.tsx      # Chat UI for campaign design
            ├── GameView/
            │   ├── GameView.tsx          # Root play layout
            │   ├── SceneImage.tsx        # Displays current scene/portrait image
            │   ├── TranscriptPanel.tsx   # Scrolling conversation history
            │   ├── AudioControls.tsx     # Record button, upload progress
            │   └── NPCBadge.tsx          # Shows active NPC name/voice indicator
            └── shared/
                ├── Spinner.tsx
                └── ErrorBanner.tsx
```

---

## 3. Data Models

### 3.1 PostgreSQL Schema (Full DDL)

```sql
-- ─────────────────────────────────────────────────────
-- Extensions
-- ─────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ─────────────────────────────────────────────────────
-- Users
-- ─────────────────────────────────────────────────────
CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_users_email ON users (email);

-- ─────────────────────────────────────────────────────
-- LLM Usage (per-user token tracking from llm-service)
-- ─────────────────────────────────────────────────────
CREATE TABLE llm_usage (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id) ON DELETE SET NULL,
    provider    TEXT NOT NULL,
    model       TEXT NOT NULL,
    tokens_in   INT  NOT NULL,
    tokens_out  INT  NOT NULL,
    cached      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_llm_usage_user ON llm_usage (user_id, created_at DESC);

-- ─────────────────────────────────────────────────────
-- Campaigns
-- ─────────────────────────────────────────────────────
CREATE TYPE campaign_phase AS ENUM (
    'character_creation',
    'campaign_design',
    'active',
    'completed'
);

CREATE TABLE campaigns (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id               UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title                 TEXT NOT NULL DEFAULT 'Untitled Campaign',
    language              TEXT NOT NULL DEFAULT 'en',
    phase                 campaign_phase NOT NULL DEFAULT 'character_creation',
    plan_json             JSONB,
    visual_style          TEXT,
    active_npc_id         UUID,
    active_npc_briefing   JSONB,    -- DM-generated instructions for the current NPC conv (goals, mood, what to reveal)
    active_npc_conv_start UUID,     -- turn_id marking the start of the current NPC conv (for history slicing)
    short_term_memory     JSONB NOT NULL DEFAULT '[]',   -- list[str] of recent event strings (SimpleMemorySystem)
    long_term_memory      TEXT  NOT NULL DEFAULT '',      -- compressed markdown history (SimpleMemorySystem)
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_campaigns_user  ON campaigns (user_id);
CREATE INDEX idx_campaigns_phase ON campaigns (phase);

-- ─────────────────────────────────────────────────────
-- Characters
-- ─────────────────────────────────────────────────────
CREATE TABLE characters (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id         UUID NOT NULL UNIQUE REFERENCES campaigns(id) ON DELETE CASCADE,
    name                TEXT NOT NULL,
    background          TEXT NOT NULL,
    class_and_level     TEXT NOT NULL,
    abilities           TEXT[] NOT NULL DEFAULT '{}',
    equipment           TEXT[] NOT NULL DEFAULT '{}',
    limitations         TEXT[] NOT NULL DEFAULT '{}',
    power_level         TEXT NOT NULL,
    visual_description  TEXT NOT NULL,
    portrait_path       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ─────────────────────────────────────────────────────
-- NPCs
-- ─────────────────────────────────────────────────────
CREATE TABLE npcs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id         UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    name                TEXT NOT NULL,
    role                TEXT NOT NULL,
    visual_description  TEXT NOT NULL,
    voice_id            TEXT NOT NULL,
    voice_instructions  TEXT NOT NULL,
    portrait_path       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (campaign_id, name)
);

CREATE INDEX idx_npcs_campaign ON npcs (campaign_id);
CREATE INDEX idx_npcs_name     ON npcs (campaign_id, name);

ALTER TABLE campaigns
    ADD CONSTRAINT fk_campaigns_active_npc
    FOREIGN KEY (active_npc_id) REFERENCES npcs(id)
    ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED;

-- ─────────────────────────────────────────────────────
-- Turns
-- ─────────────────────────────────────────────────────
CREATE TYPE turn_role AS ENUM ('player', 'dm', 'npc', 'system');

CREATE TABLE turns (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    role        turn_role NOT NULL,
    content     TEXT NOT NULL,
    npc_name    TEXT,
    audio_path  TEXT,
    image_path  TEXT,
    metadata    JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_turns_campaign_created ON turns (campaign_id, created_at DESC);
CREATE INDEX idx_turns_campaign_role    ON turns (campaign_id, role);
CREATE INDEX idx_turns_campaign_npc     ON turns (campaign_id, npc_name)
    WHERE npc_name IS NOT NULL;
```

### 3.2 Qdrant Collection Specification

**Collection name:** `campaign_turns`

| Field | Value |
|---|---|
| Vector size | 1536 |
| Distance | Cosine |
| Embedding model | `gemini-embedding-2` via `google-genai` SDK (`client.models.embed_content`, auto-normalized) |
| On-disk payload | Yes |

**Point schema:**

```json
{
  "id": "<uuid — matches turns.id>",
  "vector": [0.023, ...],
  "payload": {
    "campaign_id": "<uuid>",
    "turn_id": "<uuid>",
    "role": "player|dm|npc|system",
    "npc_name": "<string or null>",
    "text_snippet": "<first 512 chars of content>",
    "created_at": "<ISO-8601>"
  }
}
```

**Filtering:** All recall queries filter by `campaign_id` before scoring.

**Index:** Create a payload index on `campaign_id` using Qdrant's `keyword` type at collection creation time.

### 3.3 Neo4j Node and Edge Schema

All nodes carry a `campaign_id` property. Every Cypher query appends `WHERE n.campaign_id = $campaign_id`.

**Node labels:**

| Label | Properties |
|---|---|
| `Location` | `name`, `campaign_id`, `description`, `type` |
| `NPC` | `name`, `campaign_id`, `role`, `faction`, `status` |
| `Faction` | `name`, `campaign_id`, `description`, `alignment` |
| `Item` | `name`, `campaign_id`, `description`, `magical` |
| `Event` | `name`, `campaign_id`, `description`, `turn_id` |

**Relationship types:**

| Type | From -> To | Properties |
|---|---|---|
| `LIVES_IN` | NPC -> Location | `since`, `status` |
| `MEMBER_OF` | NPC -> Faction | `rank` |
| `ALLIED_WITH` | Faction -> Faction | `strength` |
| `HOSTILE_TO` | Faction -> Faction | `since` |
| `CONTROLS` | Faction -> Location | — |
| `VISITED` | Event -> Location | — |
| `INVOLVES` | Event -> NPC | `role` |
| `OWNS` | NPC -> Item | — |
| `LOCATED_IN` | Item -> Location | — |

**Indexes:**

```cypher
CREATE INDEX npc_campaign      FOR (n:NPC)      ON (n.campaign_id, n.name);
CREATE INDEX location_campaign FOR (n:Location) ON (n.campaign_id, n.name);
CREATE INDEX faction_campaign  FOR (n:Faction)  ON (n.campaign_id, n.name);
```

---

## 4. MCP Servers

All MCP servers are FastAPI applications communicating over plain HTTP. Each server has `CampaignIDMiddleware` that reads the `X-Campaign-ID` request header and stores it in `request.state.campaign_id`. Tool endpoints follow the convention `POST /tools/{tool_name}`.

---

### 4.1 state-mcp (port 8001)

**Purpose:** Single source of truth for all structured campaign state in PostgreSQL.

| Env var | Description |
|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://...` |
| `PORT` | `8001` |

**Tool schemas:**

```
create_campaign
  Input:  { title?: string, language: string }
  Output: { campaign_id: string }

save_campaign_plan
  Input:  { plan_json: object }
  Output: { ok: true }

save_character
  Input:  { character_json: object, portrait_path?: string }
  Output: { character_id: string }

set_phase
  Input:  { phase: "character_creation"|"campaign_design"|"active"|"completed" }
  Output: { ok: true }

set_active_npc
  Input:  { npc_id: string, briefing: object, conv_start_turn_id: string }
  Output: { ok: true }
  Description: Begins an NPC conversation. Stores the DM-generated briefing (goals,
               mood, what the NPC knows/should reveal) and the turn_id that marks the
               start of this conversation so npc-agent can slice history correctly.

get_active_npc_state
  Input:  {}
  Output: { npc_id: string, briefing: object, conv_start_turn_id: string } | { npc_id: null }
  Description: Returns full NPC conversation state. Called by npc-agent at the start
               of every turn. Returns npc_id: null when no conversation is active.

clear_active_npc
  Input:  {}
  Output: { ok: true }

get_routing_state
  Input:  {}
  Output: { phase: string, active_npc_id: string|null }
  Description: Single fast SELECT used exclusively by the api dispatcher to determine
               which agent to route to. Returns only what is needed for routing.

get_campaign_context
  Input:  {}
  Output: {
    campaign: { id, title, language, phase, plan_json, visual_style },
    character: { id, name, background, class_and_level, abilities,
                 equipment, limitations, power_level, visual_description,
                 portrait_path } | null
  }

save_npc
  Input:  { npc_json: object, portrait_path?: string }
  Output: { npc_id: string }

get_npc
  Input:  { npc_id?: string, name?: string }
  Output: { npc: object } | { npc: null }
  Description: Lookup by npc_id (preferred, used by npc-agent) or by name.

list_npcs
  Input:  {}
  Output: { npcs: object[] }

log_turn
  Input:  { role: "player"|"dm"|"npc"|"system", content: string, npc_name?: string, metadata?: object }
  Output: { turn_id: string }

get_turns
  Input:  {
    limit?:          number,      -- max turns to return (default 20)
    exclude_roles?:  string[],    -- e.g. ["npc"] — DM uses this to keep NPC chatter out of its context
    since_turn_id?:  string,      -- return only turns created after this turn_id (exclusive); npc-agent uses this to slice to current conv
    before_turn_id?: string,      -- return only turns created before this turn_id (exclusive); used to fetch preamble
    npc_name?:       string       -- filter to turns from a specific NPC
  }
  Output: { turns: [{ id, role, content, npc_name, created_at, metadata }] }
  Description: Always returns turns in ascending created_at order.
               since_turn_id and before_turn_id use a sub-select on created_at for efficiency.

get_memory
  Input:  {}
  Output: { short_term: list[string], long_term: string }
  Description: Returns campaigns.short_term_memory and campaigns.long_term_memory.
               Called by memory-agent at the start of every memory consolidation cycle.

update_memory
  Input:  { short_term: list[string], long_term: string }
  Output: { ok: true }
  Description: Atomically overwrites both memory fields. Called by memory-agent
               after compression logic runs.
```

**Implementation notes:**

- All DB access is async (asyncpg via SQLAlchemy 2.x async session).
- `get_routing_state` executes a single `SELECT phase, active_npc_id FROM campaigns WHERE id = $1`.
- `get_campaign_context` is a single JOIN query on campaigns + characters.
- `save_npc` uses `INSERT ... ON CONFLICT (campaign_id, name) DO UPDATE`.
- `set_active_npc` atomically writes `active_npc_id`, `active_npc_briefing`, and `active_npc_conv_start` in one UPDATE.
- `clear_active_npc` sets all three NPC conv columns to NULL in one UPDATE.
- `get_turns` with `since_turn_id`: `WHERE created_at > (SELECT created_at FROM turns WHERE id = $since_turn_id)`.
- `get_turns` with `before_turn_id`: `WHERE created_at < (SELECT created_at FROM turns WHERE id = $before_turn_id) ORDER BY created_at DESC LIMIT $limit` — results re-sorted ascending before return.

---

### 4.2 memory-mcp (port 8002)

**Purpose:** Semantic memory via Qdrant. Stores turns as vector embeddings (gemini-embedding-2) and retrieves the top-k most relevant past turns for a given query.

| Env var | Description |
|---|---|
| `QDRANT_URL` | `http://qdrant:6333` |
| `GEMINI_API_KEY` | Google AI Studio key (used for `gemini-embedding-2` calls) |
| `PORT` | `8002` |

**Tool schemas:**

```
store
  Input:  { turn_id: string, text: string, role: string }
  Output: { ok: true }
  Description: Embeds text via gemini-embedding-2 and upserts a point into Qdrant.
               Point ID equals the turn_id UUID.

recall
  Input:  { query: string, top_k?: number (default 8) }
  Output: { context: string }
  Description: Embeds query, runs filtered vector search, returns formatted markdown
               of top_k hits: "[<role>] <text_snippet>" per line.
```

**Implementation notes:**

- Embedding per-request via `google-genai`: `client.models.embed_content(model="gemini-embedding-2", contents=text, config=types.EmbedContentConfig(output_dimensionality=1536))`. Auto-normalized.
- Qdrant collection `campaign_turns` created at startup if absent: `VectorParams(size=1536, distance=Distance.COSINE)`.
- `store` uses `qdrant_client.upsert` (idempotent).
- `recall` filters by `campaign_id` payload field before scoring.

---

### 4.3 knowledge-mcp (port 8003)

**Purpose:** World knowledge graph in Neo4j. Two tools: ingest narrative text and extract graph updates; retrieve structured world context for agent prompts.

| Env var | Description |
|---|---|
| `NEO4J_URI` | `bolt://neo4j:7687` |
| `NEO4J_USER` | `neo4j` |
| `NEO4J_PASSWORD` | set in `.env` |
| `LLM_SERVICE_URL` | `http://llm-service:9001` |
| `PORT` | `8003` |

**Tool schemas:**

```
update_world
  Input:  { narrative_text: string }
  Output: { entities_added: number, relationships_added: number }
  Description: LLM call to extract entities/relationships, then MERGE into Neo4j.
               LLM returns JSON: { nodes: [...], relationships: [...] }

get_world_context
  Input:  { focus_text?: string }
  Output: { context: string }
  Description: Full-text search for focus_text, returns matching nodes + 1-hop
               relationships as human-readable markdown. Without focus_text,
               returns summary of all campaign nodes (max 50).
```

**Implementation notes:**

- `update_world` stamps `campaign_id` on every merged node.
- `get_world_context` output format:
  ```
  ## World Context
  ### Entities
  - [NPC] Gareth — member of Iron Hand faction
  ### Relationships
  - Gareth MEMBER_OF Iron Hand (rank: sergeant)
  ```
- Neo4j async driver; session opened per request.
- Full-text index created at startup: `CREATE FULLTEXT INDEX entity_search IF NOT EXISTS FOR (n:NPC|Location|Faction|Item) ON EACH [n.name, n.description]`.

---

### 4.4 media-mcp (port 8004)

**Purpose:** All media generation and processing. Delegates to `image-service`, `tts-service`, and `stt-service`. Writes files to the shared `media_data` volume.

| Env var | Description |
|---|---|
| `IMAGE_SERVICE_URL` | `http://image-service:9002` |
| `TTS_SERVICE_URL` | `http://tts-service:9003` |
| `STT_SERVICE_URL` | `http://stt-service:9004` |
| `MEDIA_ROOT` | `/media` |
| `PORT` | `8004` |

**Tool schemas:**

```
generate_image
  Input:  { prompt: string, style: string, type: "scene"|"portrait" }
  Output: { file_path: string }
  Description: Calls image-service. Scene=16:9 (1344x768), Portrait=1:1 (1024x1024).
               Saves JPEG to $MEDIA_ROOT/images/<timestamp>_<uuid>.jpg.
               Returns relative path.

speak
  Input:  { text: string, voice_id: string, voice_instructions: string }
  Output: { file_path: string }
  Description: Calls tts-service POST /speak. SHA-256 caching inside tts-service.
               Saves WAV to $MEDIA_ROOT/audio/<hash>.wav. Returns relative path.

transcribe
  Input:  { file_path: string }
  Output: { text: string }
  Description: Calls stt-service POST /transcribe with the audio file path.
               Returns transcription string.
```

**Implementation notes:**

- No API keys in media-mcp; all credentials live in image-service, tts-service, and stt-service.
- Image negative prompt: `"blurry, low quality, distorted, text, watermark, ugly, deformed"`. Portraits add `"multiple people, crowd"`.
- TTS output is WAV (PCM 24kHz mono).

---

### 4.5 llm-service (port 9001)

**Purpose:** Single choke-point for all LLM text generation. Provider routing, Redis response caching, per-user token logging to Postgres. Switching providers requires only changing `LLM_PROVIDER`.

| Env var | Description |
|---|---|
| `LLM_PROVIDER` | `gemini` \| `openai` \| `anthropic` |
| `GEMINI_API_KEY` | Google AI Studio key |
| `GEMINI_MODEL` | e.g. `gemini-2.5-flash` |
| `OPENAI_API_KEY` | For OpenAI or Anthropic via compat |
| `OPENAI_MODEL` | e.g. `gpt-4o` |
| `ANTHROPIC_API_KEY` | For Anthropic provider |
| `ANTHROPIC_MODEL` | e.g. `claude-sonnet-4-6` |
| `REDIS_URL` | Response caching |
| `DATABASE_URL` | Token usage logging |
| `PORT` | `9001` |

**API:**

```
POST /generate
  Input:  {
    messages: [{role: "system"|"user"|"assistant", content: string}],
    response_format?: "text" | "json",
    user_id?: string,
    cache?: boolean   # default true
  }
  Output: { text: string, tokens_in: int, tokens_out: int, cached: boolean }

GET /health
  Output: { ok: true, provider: string, model: string }
```

**Adapter pattern:**

```python
class GeminiProvider(LLMProvider):
    async def generate(self, messages, response_format):
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=messages_to_gemini(messages),
            config=types.GenerateContentConfig(
                response_mime_type="application/json" if response_format == "json" else "text/plain"
            )
        )
        return response.text, response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count
```

**Cross-cutting concerns:**

1. **Caching:** `cache_key = sha256(json(messages) + provider + model)`. TTL: 1 hour.
2. **Token logging:** `INSERT INTO llm_usage (user_id, provider, model, tokens_in, tokens_out, cached, created_at)`.
3. **Request logging:** Structured log line per request.

---

### 4.6 image-service (port 9002)

**Purpose:** Single choke-point for image generation. Returns image bytes; caller (media-mcp) writes to the shared volume. Switching providers requires only changing `IMAGE_PROVIDER`.

| Env var | Description |
|---|---|
| `IMAGE_PROVIDER` | `gemini` \| `runware` \| `dalle` |
| `GEMINI_API_KEY` | For Gemini image generation |
| `GEMINI_IMAGE_MODEL` | e.g. `gemini-2.5-flash-preview-image-generation` |
| `RUNWARE_API_KEY` | For Runware provider |
| `OPENAI_API_KEY` | For DALL-E provider |
| `REDIS_URL` | Response caching |
| `DATABASE_URL` | Cost logging |
| `PORT` | `9002` |

**API:**

```
POST /generate
  Input:  { prompt: string, style: string, type: "scene"|"portrait", user_id?: string, cache?: boolean }
  Output: { image_bytes: string (base64), format: "png"|"jpeg", cached: boolean }

GET /health
  Output: { ok: true, provider: string, model: string }
```

**Gemini adapter:**

```python
class GeminiImageProvider(ImageProvider):
    async def generate(self, prompt: str, style: str, type: str) -> bytes:
        full_prompt = f"{style}\n\n{prompt}"
        aspect = "1:1" if type == "portrait" else "16:9"
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_generation_config=types.ImageGenerationConfig(aspect_ratio=aspect)
            )
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                return part.inline_data.data
```

---

### 4.7 tts-service (port 9003)

**Purpose:** Single choke-point for Text-to-Speech. Caches responses keyed on `sha256(text + voice_id + instructions)` so repeated NPC lines don't re-synthesise.

| Env var | Description |
|---|---|
| `TTS_PROVIDER` | `openai` \| `elevenlabs` \| `gemini` |
| `OPENAI_API_KEY` | For OpenAI TTS |
| `OPENAI_TTS_MODEL` | e.g. `gpt-4o-mini-tts` |
| `ELEVENLABS_API_KEY` | For ElevenLabs provider |
| `REDIS_URL` | TTS audio caching |
| `DATABASE_URL` | Usage logging |
| `PORT` | `9003` |

**API:**

```
POST /speak
  Input:  { text: string, voice_id: string, voice_instructions: string, user_id?: string }
  Output: { audio_bytes: string (base64), format: "wav", duration_sec: float, cached: boolean }

GET /health
  Output: { ok: true, provider: string, model: string }
```

**Notes:**

- Output is WAV (PCM 24kHz mono). Callers (media-mcp) write bytes to the shared volume.
- Cache hit returns audio_bytes immediately without calling the external API.
- Swapping `TTS_PROVIDER` without restarting other services is the intended upgrade path.

---

### 4.8 stt-service (port 9004)

**Purpose:** Single choke-point for Speech-to-Text. Loads the Whisper model once at startup and holds it in memory. Separated from tts-service because Whisper startup is slow (~10-30s depending on model size) and it needs its own healthcheck.

| Env var | Description |
|---|---|
| `STT_PROVIDER` | `whisper_local` \| `openai` \| `gemini` |
| `WHISPER_MODEL` | `tiny` \| `base` \| `small` \| `medium` \| `large` |
| `OPENAI_API_KEY` | For OpenAI Whisper API provider |
| `DATABASE_URL` | Usage logging |
| `PORT` | `9004` |

**API:**

```
POST /transcribe
  Input:  { audio_bytes: string (base64), format: "webm"|"wav"|"mp3" }
  Output: { text: string }

GET /health
  Output: { ok: true, provider: string, model_loaded: boolean }
```

**Notes:**

- Accepts `audio/webm` (native browser MediaRecorder output). Local Whisper provider runs `ffmpeg` internally to convert to WAV before transcription.
- Docker Compose healthcheck polls `/health` and waits for `model_loaded: true` before marking the container healthy.
- `whisper_local` is the default (zero marginal cost); swap to `openai` for faster cold starts in production.

---

## 5. Agent Specifications

All five agents are **independent FastAPI microservices** communicating via the **Google A2A protocol**. The `api` dispatcher sends A2A tasks; agents call MCP servers for data access and `llm-service` for LLM inference.

### A2A Protocol Contract (all five agents implement)

```
GET  /.well-known/agent.json  -> AgentCard { name, description, version, skills[] }
POST /                         -> JSON-RPC 2.0
                                  method: "tasks/send"
                                  params: { task_id, campaign_id, message }
                                  response: { jsonrpc: "2.0", result: { output, done } }
GET  /tasks/{task_id}          -> { status: "pending"|"running"|"done"|"error", result? }
```

Each agent receives `campaign_id` in the task params and injects it as `X-Campaign-ID` header on all MCP calls. Agents call `llm-service POST /generate` for all LLM inference — never the LLM API directly. SSE events are published to Redis by each agent; the `api` SSE endpoint fans them out to the browser.

---

### 5.1 api/dispatcher.py — A2A Dispatcher (not an agent service)

**Purpose:** Pure router inside the `api` process. Reads `get_routing_state()` and sends one A2A task to the correct agent service. No LLM call.

**Step-by-step flow:**

1. Call `state-mcp:get_routing_state()` -> `{ phase, active_npc_id }`.
2. Decision tree:
   - `phase == "character_creation"` -> send A2A task to `character-creator:8010`.
   - `phase == "campaign_design"` -> send A2A task to `campaign-designer:8011`.
   - `phase == "active"` AND `active_npc_id != null` -> send A2A task to `npc-agent:8013`.
   - `phase == "active"` AND `active_npc_id == null` -> send A2A task to `dm-agent:8012`.
   - `phase == "completed"` -> publish SSE event `{ type: "campaign_completed" }`, return.
3. A2A responses are a stream of SSE-compatible events published to Redis as they arrive.

---

### 5.2 character-creator (port 8010) — A2A Agent

**Purpose:** Guides the player through interactive character creation via multi-turn conversation, then writes the finalised character and transitions phase to `campaign_design`.

**MCP tools used:** `state-mcp`, `media-mcp`

**System prompt:**
```
You are a D&D character creation assistant. Ask 4-6 focused questions covering
background, class/level, abilities, equipment, limitations, and physical appearance.
When you have enough information for a complete character sheet, end with [DONE].
```

**Step-by-step flow:**

1. Call `state-mcp:get_turns(limit=20)` to load prior exchanges.
2. Call `state-mcp:get_campaign_context()` for language setting.
3. Build LLM messages from prior turns + player message. POST to `llm-service /generate`.
4. Call `state-mcp:log_turn(role="dm", content=<question>)`.
5. Publish SSE `{ type: "dm_text", content: <question> }`.
6. Call `media-mcp:speak(text=<question>, voice_id="ash", voice_instructions="...")` -> `audio_path`.
7. Publish SSE `{ type: "audio_ready", file_path: audio_path }`.
8. If response contains `[DONE]`:
   - POST to `llm-service /generate` with structured prompt -> parse `PlayerCharacter` JSON.
   - Call `media-mcp:generate_image(prompt=character.visual_description, style=campaign.visual_style, type="portrait")` -> `portrait_path`.
   - Call `state-mcp:save_character(character_json=<character>, portrait_path=portrait_path)`.
   - Call `state-mcp:set_phase(phase="campaign_design")`.
   - Publish SSE `{ type: "portrait_ready", file_path: portrait_path }`.
   - Publish SSE `{ type: "phase_change", phase: "campaign_design" }`.

**SSE events emitted:** `dm_text`, `audio_ready`, `portrait_ready`, `phase_change`.

---

### 5.3 campaign-designer (port 8011) — A2A Agent

**Purpose:** Guides the player through campaign design via multi-turn conversation, then writes the finalised campaign plan and transitions phase to `active`.

**MCP tools used:** `state-mcp`, `knowledge-mcp`

**System prompt:**
```
You are an expert D&D campaign designer. You have the player's character sheet.
Ask 3-5 focused questions about genre, tone, themes, and how the character's background
should drive the story. End your final response with [DONE].

Character: {character_json}
```

**Step-by-step flow:**

1. Call `state-mcp:get_campaign_context()` -> character details for prompt.
2. Call `state-mcp:get_turns(limit=20)` to load prior exchanges.
3. POST to `llm-service /generate` -> next question or `[DONE]`.
4. Call `state-mcp:log_turn(role="dm", content=<question>)`.
5. Publish SSE `{ type: "dm_text", content: <question> }`.
6. If response contains `[DONE]`:
   - POST to `llm-service /generate` with structured prompt -> parse `CampaignPlan` JSON.
   - Call `state-mcp:save_campaign_plan(plan_json=<plan>)`.
   - Call `knowledge-mcp:update_world(narrative_text=plan.synopsis + " " + plan.acts)` to seed the world graph.
   - Call `state-mcp:set_phase(phase="active")`.
   - Publish SSE `{ type: "campaign_plan_ready", plan: <plan> }`.
   - Publish SSE `{ type: "phase_change", phase: "active" }`.

**SSE events emitted:** `dm_text`, `campaign_plan_ready`, `phase_change`.

---

### 5.4 dm-agent (port 8012) — A2A Agent

**Purpose:** Core game DM. Processes player input, narrates the scene, generates audio and image concurrently, and optionally starts an NPC multi-turn conversation (generating the NPC briefing and opening line itself). Delegates all memory operations to `memory-agent`.

**MCP tools used:** `state-mcp`, `knowledge-mcp`, `media-mcp`

**A2A calls made:** -> `memory-agent:8014`

**System prompt:**
```
You are an expert Dungeon Master for a voice-driven D&D campaign.
Respond with JSON matching this schema exactly:
  - gm_speech:        string  — narration (2-5 sentences, vivid, immersive)
  - scene_description: string — visual description for image gen (400-800 chars)
  - memory_note:      string  — one-sentence summary of what changed in the world
  - invoke_npc:       object|null — null if no NPC conv should start; otherwise:
      {
        name:               string  — NPC name (must match an existing NPC or be new)
        role:               string  — short role description (used if NPC is new)
        visual_description: string  — appearance (used if NPC is new)
        voice_id:           string  — OpenAI voice preset (used if NPC is new)
        voice_instructions: string  — personality/delivery instructions (used if NPC is new)
        briefing:           object  — dynamic instructions for this conversation:
          {
            goals:       string  — what the NPC wants from this interaction
            knows:       string  — information the NPC possesses (can be withheld or revealed)
            mood:        string  — NPC's current emotional state
            reveal_if:   string  — condition under which hidden info should be shared
          }
        opening_line:   string  — the NPC's first words to the player
      }

Campaign: {campaign_json}
Character: {character_json}
Long-term memory: {long_term_summary}
Recent events (memory): {recent_events}
Recalled context (semantic): {recalled_context}
World context: {world_context}
Recent turns (last 10, NPC turns excluded): {recent_turns}
```

**Step-by-step flow:**

1. Call `state-mcp:get_campaign_context()` -> `{ campaign, character }`.
2. Call `state-mcp:get_turns(limit=10, exclude_roles=["npc"])` -> recent turn history (NPC chatter excluded; system summary turns are included so the DM sees past NPC conv summaries).
3. Call `knowledge-mcp:get_world_context(focus_text=player_message)` -> world graph context.
4. Send A2A task to `memory-agent` with `{ query: player_message, new_event: "" }` -> `{ recalled_context, long_term_summary, recent_events }`.
5. Call `state-mcp:log_turn(role="player", content=player_message)`.
6. Build LLM prompt from all gathered context. POST to `llm-service /generate` -> structured `DMResponse`.
7. Publish SSE `{ type: "dm_text", content: dm_response.gm_speech }`.
8. Launch concurrent tasks:
   - Task A: `media-mcp:speak(text=dm_response.gm_speech, voice_id="ash", voice_instructions="...")` -> `audio_path`.
   - Task B: `media-mcp:generate_image(prompt=dm_response.scene_description, type="scene")` -> `image_path`.
9. Await both tasks.
10. Call `state-mcp:log_turn(role="dm", content=dm_response.gm_speech, metadata={ audio_path, image_path })`.
11. Send A2A task to `memory-agent` with `{ query: player_message, new_event: dm_response.memory_note }` (post-turn store + consolidation). Fire-and-forget.
12. Call `knowledge-mcp:update_world(narrative_text=dm_response.gm_speech)`. Fire-and-forget.
13. Publish SSE `{ type: "audio_ready", file_path: audio_path }`.
14. Publish SSE `{ type: "scene_ready", file_path: image_path }`.
15. If `dm_response.invoke_npc != null`:
    - Call `state-mcp:save_npc(npc_json=invoke_npc)` -> `npc_id` (upsert — creates if new, updates if exists).
    - Concurrently: `media-mcp:generate_image(prompt=invoke_npc.visual_description, type="portrait")` -> `portrait_path`.
    - Call `state-mcp:save_npc(npc_json=invoke_npc, portrait_path=portrait_path)` (update portrait).
    - Call `media-mcp:speak(text=invoke_npc.opening_line, voice_id=invoke_npc.voice_id, voice_instructions=invoke_npc.voice_instructions)` -> `opening_audio_path`.
    - Call `state-mcp:log_turn(role="npc", content=invoke_npc.opening_line, npc_name=invoke_npc.name, metadata={ audio_path: opening_audio_path })` -> `conv_start_turn_id`.
    - Call `state-mcp:set_active_npc(npc_id=npc_id, briefing=invoke_npc.briefing, conv_start_turn_id=conv_start_turn_id)`.
    - Publish SSE `{ type: "npc_introduced", npc_name: invoke_npc.name, portrait_path: portrait_path }`.
    - Publish SSE `{ type: "npc_speech", npc_name: invoke_npc.name, content: invoke_npc.opening_line }`.
    - Publish SSE `{ type: "audio_ready", file_path: opening_audio_path }`.

**SSE events emitted:** `dm_text`, `audio_ready`, `scene_ready`, `npc_introduced`, `npc_speech`.

---

### 5.5 npc-agent (port 8013) — A2A Agent

**Purpose:** Handles one turn of a multi-turn NPC conversation. Called on every player message while `active_npc_id` is set. Each call is stateless — the full conversation context is rebuilt from the database on every turn.

**MCP tools used:** `state-mcp`, `media-mcp`

**How context is assembled per turn:**

The npc-agent builds a prompt from three distinct layers, each with clear role labels:

```
[PREAMBLE — 3 turns before the conversation started, for continuity]
[DM]     "You push open the tavern door. Elara looks up from the bar."
[Player] "I approach the innkeeper."
[DM]     "She sets down her cloth and turns to face you."

[NPC BRIEFING — DM's dynamic instructions for this conversation]
  goals:     "Protect her regular customers; be cautious with strangers"
  knows:     "Merchants left 3 nights ago toward Loppio via eastern road; she heard them arguing"
  mood:      "Tired, slightly suspicious"
  reveal_if: "Player seems trustworthy or offers coin"

[CONVERSATION SO FAR — all turns since conv_start_turn_id]
[You]    "Evening. What'll it be?"
[Player] "I'm looking for the missing merchants."
[You]    "Aye, they left three nights past..."
[Player] "Do you know where they went?"
```

**System prompt:**
```
You are {npc.name} in a live D&D session. Stay in character at all times.

STATIC PROFILE:
  Role: {npc.role}
  Voice/personality: {npc.voice_instructions}

BRIEFING FOR THIS CONVERSATION:
  Goals:      {briefing.goals}
  You know:   {briefing.knows}
  Your mood:  {briefing.mood}
  Reveal if:  {briefing.reveal_if}

CONTEXT (what led to this conversation):
{preamble_turns}

CONVERSATION SO FAR:
{conv_turns}

Respond with JSON:
  - npc_speech: string  — your next line, in character, natural dialogue length
  - done:       bool    — true ONLY if the conversation has reached a natural close
                          (either you or the player said goodbye / thanks / farewell,
                           or the topic is fully resolved and continuing would feel forced)

Never set done=true mid-conversation. The conversation ends when it genuinely feels over.
```

**Step-by-step flow:**

1. Call `state-mcp:get_active_npc_state()` -> `{ npc_id, briefing, conv_start_turn_id }`.
2. Call `state-mcp:get_npc(npc_id=npc_id)` -> full NPC record (name, role, voice_id, voice_instructions).
3. Call `state-mcp:get_turns(before_turn_id=conv_start_turn_id, limit=3)` -> 3-turn preamble (turns just before the conv started, gives context for how the player arrived at this NPC).
4. Call `state-mcp:get_turns(since_turn_id=conv_start_turn_id)` -> all turns since conv start (the conversation itself: player + npc turns interleaved).
5. Format preamble and conv turns with clear `[DM]`, `[Player]`, `[You]` labels.
6. POST to `llm-service /generate` -> `{ npc_speech, done }`.
7. Call `media-mcp:speak(text=npc_speech, voice_id=npc.voice_id, voice_instructions=npc.voice_instructions)` -> `audio_path`.
8. Call `state-mcp:log_turn(role="npc", content=npc_speech, npc_name=npc.name, metadata={ audio_path })`.
9. Publish SSE `{ type: "npc_speech", npc_name: npc.name, content: npc_speech }`.
10. Publish SSE `{ type: "audio_ready", file_path: audio_path }`.
11. If `done == true`:
    - POST to `llm-service /generate` with summary prompt: "Summarise this NPC conversation in 1-2 sentences from the DM's perspective, noting only what the player learned or agreed to." -> `summary_text`.
    - Call `state-mcp:log_turn(role="system", content=summary_text, metadata={ type: "npc_conv_summary", npc_name: npc.name })`.
    - Call `state-mcp:clear_active_npc()`.
    - Publish SSE `{ type: "npc_conversation_ended", npc_name: npc.name, summary: summary_text }`.

**Why `done` detection works:** The LLM sees the full conversation and player message. Natural conversation closers ("Thanks", "Goodbye", "That's all I needed", "Safe travels") from either side trigger `done=true`. The prompt explicitly instructs the model not to end prematurely.

**SSE events emitted:** `npc_speech`, `audio_ready`, `npc_conversation_ended`.

---

### 5.6 memory-agent (port 8014) — A2A Agent

**Purpose:** Encapsulates all memory intelligence. dm-agent and npc-agent call this instead of memory-mcp directly. Implements the same two-tier memory system as `src/memory.py` (`SimpleMemorySystem`): short-term list + compressed long-term markdown with LLM-driven cutoff detection and compression.

**MCP tools used:** `state-mcp` (STM/LTM in Postgres), `memory-mcp` (Qdrant vector ops)

**A2A input:** `{ query: string, new_event: string }`

- `query`: what the caller is looking for (used for semantic recall)
- `new_event`: one-sentence summary of what just happened (empty string for pre-turn recall)

**Step-by-step flow:**

1. Call `state-mcp:get_memory()` -> `{ short_term: list[str], long_term: str }`.
2. If `new_event != ""`:
   - Call `memory-mcp:store(text=new_event, role="dm")` -> stores in Qdrant.
   - Append `new_event` to `short_term` list.
3. Call `memory-mcp:recall(query=query, top_k=8)` -> semantic hits.
4. If `len(short_term) >= 5`:
   - POST to `llm-service /generate` with `CutoffDecision` prompt (mirrors `src/memory.py:check_for_cutoff` — checks for narrative breaks, location changes, concluded scenes).
   - If `should_compress == true`: POST to `llm-service /generate` with `MemoryCompression` prompt (mirrors `src/memory.py:compress_memory` — merges STM into LTM, keeps last 3 events in short_term).
5. Call `state-mcp:update_memory(short_term=short_term, long_term=long_term)`.
6. Return A2A result: `{ recalled_context: <semantic hits>, long_term_summary: <long_term>, recent_events: <short_term> }`.

**SSE events emitted:** None (internal service only).

---

## 6. API Layer

### 6.1 All Endpoints

All endpoints are prefixed with `/api/v1`.

#### Authentication

| Method | Path | Auth | Request Body | Response | Description |
|---|---|---|---|---|---|
| POST | `/auth/register` | None | `{ email, password }` | `{ user_id, email }` | Create user account |
| POST | `/auth/login` | None | `{ email, password }` | `{ access_token, token_type: "bearer" }` | Get access token (httpOnly cookie also set) |

#### Campaigns

| Method | Path | Auth | Request Body | Response | Description |
|---|---|---|---|---|---|
| GET | `/campaigns` | Bearer | — | `[{ id, title, phase, created_at }]` | List user's campaigns |
| POST | `/campaigns` | Bearer | `{ title?, language }` | `{ campaign_id, title, phase }` | Create new campaign |
| GET | `/campaigns/:id` | Bearer | — | Full campaign object | Get campaign details |
| DELETE | `/campaigns/:id` | Bearer | — | `{ ok: true }` | Delete campaign |

#### Game

| Method | Path | Auth | Request Body | Response | Description |
|---|---|---|---|---|---|
| POST | `/campaigns/:id/audio` | Bearer | multipart `file=<wav>` | `{ file_path, transcript }` | Upload player audio; calls `media-mcp:transcribe`; returns transcript |
| POST | `/campaigns/:id/message` | Bearer | `{ content: string }` | `{ ok: true }` | Submit player text; triggers A2A dispatcher as asyncio.create_task; returns immediately |
| GET | `/campaigns/:id/stream` | Bearer/Cookie | — | `text/event-stream` | SSE stream for this campaign |
| GET | `/campaigns/:id/turns` | Bearer | `?limit=50&role=&npc_name=` | `[turn]` | Load turn history |
| GET | `/media/:path*` | Bearer/Cookie | — | Binary file | Serve file from media volume |

**Notes:**

- `POST /campaigns/:id/message` returns HTTP 202 immediately. Agent responses arrive via SSE.
- `GET /media/:path*` uses `FileResponse` to stream from the `media_data` volume.

### 6.2 SSE Event Catalogue

All SSE events published on Redis channel `sse:campaign:<campaign_id>`.

| Event type | Data shape | Emitter | When |
|---|---|---|---|
| `dm_text` | `{ content: string }` | DM/Character/Campaign agents | LLM narration ready |
| `audio_ready` | `{ file_path: string }` | DM/NPC/Character agents | TTS file written |
| `scene_ready` | `{ file_path: string }` | dm-agent | Scene image written |
| `portrait_ready` | `{ file_path: string }` | character-creator | Character portrait written |
| `npc_introduced` | `{ npc: object, portrait_path: string }` | dm-agent | NPC enters scene |
| `npc_speech` | `{ npc_name: string, content: string }` | npc-agent | NPC dialogue line |
| `npc_conversation_ended` | `{ summary: string }` | npc-agent | NPC conversation concluded |
| `phase_change` | `{ phase: string }` | Setup agents | Phase transition |
| `campaign_plan_ready` | `{ plan: object }` | campaign-designer | Campaign plan finalised |
| `campaign_completed` | `{}` | api dispatcher | Campaign in completed state |
| `error` | `{ message: string, code: string }` | Any agent | Agent-level error |
| `ping` | `{}` | Server | Every 30s to keep SSE alive |

**SSE wire format:**
```
data: {"type":"dm_text","content":"The torchlight flickers..."}

data: {"type":"audio_ready","file_path":"audio/abc123.wav"}

data: {"type":"scene_ready","file_path":"images/456def.jpg"}
```

### 6.3 Authentication Flow

**Registration:** `POST /auth/register` hashes password with bcrypt (cost factor 12), inserts `users` row, returns user object (no token — user must login separately).

**Login:** `POST /auth/login` verifies password hash. On success generates:

- Access token: JWT HS256, payload `{ sub: user_id, exp: now+JWT_ACCESS_EXPIRE_MINUTES, type: "access" }`, signed with `JWT_SECRET`.
- Sets token as `httpOnly`, `SameSite=strict` cookie. This allows SSE and `<audio src>` / `<img src>` requests to authenticate automatically.

**No refresh tokens.** When the access token expires, the user logs in again.

**Protected endpoints:** `get_current_user` dependency checks `Authorization: Bearer <token>` header first, then the `access_token` cookie. Invalid/expired tokens return HTTP 401.

### 6.4 How campaign_id Flows

1. Client sends `POST /campaigns/:id/message` with JWT in `Authorization` header or cookie.
2. `get_current_user` resolves `user_id` from JWT.
3. Router reads `campaign_id` from path parameter, confirms `user_id` ownership. Returns HTTP 403 if not owned.
4. `game/service.py` passes `campaign_id` to `dispatcher.py`.
5. `dispatcher.py` calls `state-mcp:get_routing_state()`, then sends A2A JSON-RPC task to the correct agent with `{ task_id, campaign_id, message }`.
6. The receiving agent extracts `campaign_id` from task params and injects it as `X-Campaign-ID` header on every MCP call via `agents/shared/mcp_client.py`.
7. `CampaignIDMiddleware` on each MCP server reads `request.headers["X-Campaign-ID"]` and sets `request.state.campaign_id`. All tool handlers use `request.state.campaign_id` — never a body parameter.

The LLM never sees campaign_id. No MCP tool schema contains it.

---

## 7. Frontend

### 7.1 Route Breakdown

#### `/login` — LoginPage

```
LoginPage
├── LoginForm
│   ├── EmailInput
│   ├── PasswordInput
│   ├── SubmitButton
│   └── ErrorBanner (if auth fails)
└── RegisterLink -> /register (or inline toggle)
```

On successful login: stores token in `authStore`, redirects to `/dashboard`.

#### `/dashboard` — DashboardPage

```
DashboardPage
├── Header (user email, logout button)
├── NewCampaignButton -> POST /campaigns -> redirect to /campaigns/:id/setup
└── CampaignList
    └── CampaignCard[] (title, phase badge, continue/delete actions)
```

"Continue" on `phase == "active"` -> `/campaigns/:id/play`.
"Continue" on `phase == "character_creation"` or `"campaign_design"` -> `/campaigns/:id/setup`.

#### `/campaigns/:id/setup` — SetupPage

```
SetupPage
├── SetupWizard
│   ├── StepIndicator (Character Creation -> Campaign Design)
│   ├── CharacterStep (shown while phase == "character_creation")
│   │   ├── TranscriptPanel (dm_text events)
│   │   ├── AudioControls (record + upload)
│   │   └── PortraitPreview (portrait_ready event)
│   └── CampaignStep (shown while phase == "campaign_design")
│       ├── TranscriptPanel
│       ├── AudioControls
│       └── PlanSummary (campaign_plan_ready event)
└── SSE connection (useSSE hook)
```

When `phase_change` arrives with `phase == "active"`, router navigates to `/campaigns/:id/play`.

#### `/campaigns/:id/play` — PlayPage

```
PlayPage
├── GameView
│   ├── SceneImage (updated by scene_ready/portrait_ready events)
│   ├── NPCBadge (shown when active NPC exists)
│   ├── TranscriptPanel (scrolling log of all turns)
│   └── AudioControls (record, upload, loading state)
└── SSE connection (useSSE hook)
```

### 7.2 SSE Integration Pattern

```typescript
function useSSE(campaignId: string) {
  const gameStore = useGameStore();

  useEffect(() => {
    const source = new EventSource(
      `/api/v1/campaigns/${campaignId}/stream`,
      { withCredentials: true }  // sends httpOnly cookie automatically
    );

    source.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      switch (payload.type) {
        case "dm_text":      gameStore.appendMessage({ role: "dm", content: payload.content }); break;
        case "npc_speech":   gameStore.appendMessage({ role: "npc", npc_name: payload.npc_name, content: payload.content }); break;
        case "audio_ready":  gameStore.setCurrentAudio(payload.file_path); break;
        case "scene_ready":  gameStore.setCurrentImage(payload.file_path); break;
        case "npc_introduced": gameStore.setActiveNPC(payload.npc); break;
        case "npc_conversation_ended": gameStore.clearActiveNPC(); break;
        case "phase_change": handlePhaseChange(payload.phase); break;
        case "error":        gameStore.setError(payload.message); break;
      }
    };

    return () => source.close();
  }, [campaignId]);
}
```

`withCredentials: true` sends the httpOnly cookie automatically — no query parameter needed.

### 7.3 Audio Recording and Upload Flow

```
1. User clicks "Start Recording"
   -> audioRecorder.start() calls navigator.mediaDevices.getUserMedia({ audio: true })
   -> MediaRecorder starts capturing in audio/webm format

2. User clicks "Stop Recording"
   -> MediaRecorder.stop() -> ondataavailable fires with Blob
   -> Blob converted to WAV via Web Audio API
   -> AudioControls shows upload progress spinner

3. POST /api/v1/campaigns/:id/audio
   -> multipart/form-data, field name "file", filename "recording.wav"
   -> Server saves to media volume as uploads/<uuid>.wav
   -> Server calls media-mcp:transcribe(file_path=<absolute path>)
   -> Response: { file_path, transcript }

4. transcript displayed immediately in TranscriptPanel as player turn

5. POST /api/v1/campaigns/:id/message
   -> Body: { content: transcript }
   -> Server returns 202
   -> Agent responses arrive via SSE
```

### 7.4 State Management

**`authStore`:**

| Field | Type | Description |
|---|---|---|
| `user` | `{ id, email } \| null` | Logged-in user |
| `accessToken` | `string \| null` | JWT access token |
| `login(token)` | action | Store token, decode user from JWT |
| `logout()` | action | Clear token, redirect to /login |

Token persisted to `localStorage`. On app load, if token exists and is not expired it is restored. If expired, user is redirected to `/login`.

**`gameStore`:**

| Field | Type | Description |
|---|---|---|
| `messages` | `Message[]` | Full transcript array |
| `currentImage` | `string \| null` | Relative file path of current scene/portrait |
| `currentAudio` | `string \| null` | Relative file path of audio to play |
| `activeNPC` | `NPC \| null` | Currently active NPC object |
| `phase` | `string` | Current campaign phase |
| `isAgentRunning` | `boolean` | True while waiting for agent to finish |
| `error` | `string \| null` | Last error message |

`currentAudio` change triggers auto-play:

```typescript
useEffect(() => {
  if (currentAudio) {
    const audio = new Audio(`/api/v1/media/${currentAudio}`);
    audio.play();
  }
}, [currentAudio]);
```

---

## 8. Turn Flows

### 8.1 Normal DM Turn (Player Audio -> DM Response)

```
Browser           API (dispatcher)     dm-agent :8012     memory-agent :8014
  |                     |                    |                    |
  |--POST /audio------->|                    |                    |
  |                     |--media-mcp:transcribe                   |
  |<--{ transcript }----|                    |                    |
  |                     |                    |                    |
  |--POST /message----->|                    |                    |
  |<--202 OK------------|                    |                    |
  |                     |--get_routing_state |                    |
  |                     |  phase=active      |                    |
  |                     |  active_npc=null   |                    |
  |                     |--A2A tasks/send--->|                    |
  |                     |                    |--get_campaign_context
  |                     |                    |--get_turns(limit=6)|
  |                     |                    |--knowledge:get_world_context
  |                     |                    |--A2A (query, no event)-------->|
  |                     |                    |<--{ recalled_ctx, ltm, recent}-|
  |                     |                    |--llm-service /generate         |
  |                     |                    |--log_turn(player)              |
  |<==SSE dm_text=======|<==Redis PUBLISH====|                    |
  |                     |                    |--[parallel]        |
  |                     |                    |  media:speak       |
  |                     |                    |  media:generate_image
  |                     |                    |--log_turn(dm, audio, image)
  |                     |                    |--A2A (query, new_event)------->|
  |                     |                    |                    |--memory-mcp:store
  |                     |                    |                    |--memory-mcp:recall
  |                     |                    |                    |--[if cutoff] compress
  |                     |                    |                    |--state-mcp:update_memory
  |                     |                    |<--{ ok }-----------|
  |                     |                    |--knowledge:update_world [async]
  |<==SSE audio_ready===|<==Redis PUBLISH====|                    |
  |<==SSE scene_ready===|<==Redis PUBLISH====|                    |
  [browser plays audio, displays image]
```

Total wall-clock latency estimate: LLM call (~2-4s) + parallel TTS + image (~3-5s) = ~5-8s from message to scene display.

### 8.2 NPC Conversation Turn

```
Browser           API (dispatcher)     npc-agent :8013
  |                     |                    |
  |--POST /audio------->|                    |
  |                     |--media-mcp:transcribe
  |<--{ transcript }--  |                    |
  |--POST /message----->|                    |
  |<--202 OK------------|                    |
  |                     |--get_routing_state |
  |                     |  active_npc_id=<id>|
  |                     |--A2A tasks/send--->|
  |                     |                    |--get_campaign_context
  |                     |                    |--get_npc(name)
  |                     |                    |--get_turns(npc_name=X, limit=20)
  |                     |                    |--knowledge:get_world_context
  |                     |                    |--llm-service /generate -> { npc_speech, done }
  |                     |                    |--media:speak(npc.voice_id)
  |                     |                    |--log_turn(role=npc)
  |<==SSE npc_speech====|<==Redis PUBLISH====|
  |<==SSE audio_ready===|<==Redis PUBLISH====|
  |                     |                    |
  [if done=true]        |                    |
  |                     |                    |--clear_active_npc
  |                     |                    |--llm-service /generate -> ConversationSummary
  |                     |                    |--log_turn(system, summary)
  |                     |                    |--knowledge:update_world
  |<==SSE npc_conv_ended|<==Redis PUBLISH====|
```

### 8.3 Character Creation Turn (Setup Phase)

```
Browser           API (dispatcher)    character-creator :8010
  |                     |                    |
  |--POST /audio + /msg>|                    |
  |<--202 OK------------|                    |
  |                     |--get_routing_state |
  |                     |  phase=character_creation
  |                     |--A2A tasks/send--->|
  |                     |                    |--get_turns(limit=20)
  |                     |                    |--get_campaign_context
  |                     |                    |--llm-service -> next question or [DONE]
  |                     |                    |--log_turn(dm, question)
  |<==SSE dm_text=======|<==Redis PUBLISH====|
  |                     |                    |--media:speak(question)
  |<==SSE audio_ready===|<==Redis PUBLISH====|
  |                     |                    |
  [if [DONE] in response]                   |
  |                     |                    |--llm-service -> parse PlayerCharacter JSON
  |                     |                    |--media:generate_image(portrait)
  |                     |                    |--save_character
  |                     |                    |--set_phase(campaign_design)
  |<==SSE portrait_ready|<==Redis PUBLISH====|
  |<==SSE phase_change==|<==Redis PUBLISH====|
  [frontend switches to campaign design step]
```

### 8.4 Campaign Design Turn (Setup Phase)

```
Browser           API (dispatcher)    campaign-designer :8011
  |                     |                    |
  |--POST /message----->|                    |
  |<--202 OK------------|                    |
  |                     |--get_routing_state |
  |                     |  phase=campaign_design
  |                     |--A2A tasks/send--->|
  |                     |                    |--get_campaign_context (for character)
  |                     |                    |--get_turns(limit=20)
  |                     |                    |--llm-service -> next question or [DONE]
  |                     |                    |--log_turn(dm, question)
  |<==SSE dm_text=======|<==Redis PUBLISH====|
  |                     |                    |
  [if [DONE] in response]                   |
  |                     |                    |--llm-service -> parse CampaignPlan JSON
  |                     |                    |--save_campaign_plan
  |                     |                    |--knowledge:update_world(synopsis+acts)
  |                     |                    |--set_phase(active)
  |<==SSE camp_plan_rdy=|<==Redis PUBLISH====|
  |<==SSE phase_change==|<==Redis PUBLISH====|
  [frontend redirects to /campaigns/:id/play]
```

### 8.5 First DM Turn (Opening Scene)

The opening scene is triggered by the frontend navigating to `/campaigns/:id/play` when `phase == "active"` and there are zero turns. The frontend sends a synthetic message:

```
POST /campaigns/:id/message
{ "content": "__opening_scene__" }
```

The API's `game/service.py` detects this sentinel and passes a fixed opening prompt to the dispatcher:

```
"Describe the opening scene for {character.name}. Set up the initial situation that draws
them into the adventure, taking into account their background and the campaign synopsis.
Do not introduce an NPC in the opening scene."
```

This flows through the normal DM turn path. The frontend checks `turns.length == 0` on `PlayPage` mount and sends the sentinel automatically.

---

## 9. Docker Compose

```yaml
# docker-compose.yml
version: "3.9"

networks:
  dnd_net:
    driver: bridge

volumes:
  postgres_data:
  qdrant_data:
  neo4j_data:
  redis_data:
  media_data:
  whisper_cache:

services:

  # -------------------------------------------------
  # Infrastructure
  # -------------------------------------------------

  postgres:
    image: postgres:16-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - dnd_net
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5
    expose:
      - "5432"

  qdrant:
    image: qdrant/qdrant:v1.9.2
    restart: unless-stopped
    volumes:
      - qdrant_data:/qdrant/storage
    networks:
      - dnd_net
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:6333/healthz || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5
    expose:
      - "6333"

  neo4j:
    image: neo4j:5.19-community
    restart: unless-stopped
    environment:
      NEO4J_AUTH: ${NEO4J_USER}/${NEO4J_PASSWORD}
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_dbms_security_procedures_unrestricted: "apoc.*"
    volumes:
      - neo4j_data:/data
    networks:
      - dnd_net
    healthcheck:
      test: ["CMD-SHELL", "wget -q --spider http://localhost:7474 || exit 1"]
      interval: 15s
      timeout: 10s
      retries: 5
    expose:
      - "7687"
      - "7474"

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --save "" --appendonly no
    volumes:
      - redis_data:/data
    networks:
      - dnd_net
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
    expose:
      - "6379"

  # -------------------------------------------------
  # Provider Services
  # -------------------------------------------------

  llm-service:
    build:
      context: ./services/llm-service
      dockerfile: Dockerfile
    restart: unless-stopped
    environment:
      LLM_PROVIDER: ${LLM_PROVIDER:-gemini}
      GEMINI_API_KEY: ${GEMINI_API_KEY}
      GEMINI_MODEL: ${GEMINI_MODEL:-gemini-2.5-flash}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      OPENAI_MODEL: ${OPENAI_MODEL:-gpt-4o}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      ANTHROPIC_MODEL: ${ANTHROPIC_MODEL:-claude-sonnet-4-6}
      REDIS_URL: redis://redis:6379/0
      DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      PORT: 9001
    networks:
      - dnd_net
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9001/health || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5
    expose:
      - "9001"

  image-service:
    build:
      context: ./services/image-service
      dockerfile: Dockerfile
    restart: unless-stopped
    environment:
      IMAGE_PROVIDER: ${IMAGE_PROVIDER:-gemini}
      GEMINI_API_KEY: ${GEMINI_API_KEY}
      GEMINI_IMAGE_MODEL: ${GEMINI_IMAGE_MODEL:-gemini-2.5-flash-preview-image-generation}
      RUNWARE_API_KEY: ${RUNWARE_API_KEY}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      REDIS_URL: redis://redis:6379/0
      DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      PORT: 9002
    networks:
      - dnd_net
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9002/health || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5
    expose:
      - "9002"

  tts-service:
    build:
      context: ./services/tts-service
      dockerfile: Dockerfile
    restart: unless-stopped
    environment:
      TTS_PROVIDER: ${TTS_PROVIDER:-openai}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      OPENAI_TTS_MODEL: ${OPENAI_TTS_MODEL:-gpt-4o-mini-tts}
      ELEVENLABS_API_KEY: ${ELEVENLABS_API_KEY:-}
      REDIS_URL: redis://redis:6379/1
      DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      PORT: 9003
    volumes:
      - media_data:/media
    networks:
      - dnd_net
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9003/health || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5
    expose:
      - "9003"

  stt-service:
    build:
      context: ./services/stt-service
      dockerfile: Dockerfile
    restart: unless-stopped
    environment:
      STT_PROVIDER: ${STT_PROVIDER:-whisper_local}
      WHISPER_MODEL: ${WHISPER_MODEL:-small}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      PORT: 9004
    volumes:
      - whisper_cache:/root/.cache/whisper
      - media_data:/media
    networks:
      - dnd_net
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9004/health | grep -q 'model_loaded.*true' || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 20
    expose:
      - "9004"

  # -------------------------------------------------
  # MCP Servers
  # -------------------------------------------------

  state-mcp:
    build:
      context: ./mcp-servers/state-mcp
      dockerfile: Dockerfile
    restart: unless-stopped
    environment:
      DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      PORT: 8001
    networks:
      - dnd_net
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8001/health || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5
    expose:
      - "8001"

  memory-mcp:
    build:
      context: ./mcp-servers/memory-mcp
      dockerfile: Dockerfile
    restart: unless-stopped
    environment:
      QDRANT_URL: http://qdrant:6333
      GEMINI_API_KEY: ${GEMINI_API_KEY}
      PORT: 8002
    networks:
      - dnd_net
    depends_on:
      qdrant:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8002/health || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5
    expose:
      - "8002"

  knowledge-mcp:
    build:
      context: ./mcp-servers/knowledge-mcp
      dockerfile: Dockerfile
    restart: unless-stopped
    environment:
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: ${NEO4J_USER}
      NEO4J_PASSWORD: ${NEO4J_PASSWORD}
      LLM_SERVICE_URL: http://llm-service:9001
      PORT: 8003
    networks:
      - dnd_net
    depends_on:
      neo4j:
        condition: service_healthy
      llm-service:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8003/health || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5
    expose:
      - "8003"

  media-mcp:
    build:
      context: ./mcp-servers/media-mcp
      dockerfile: Dockerfile
    restart: unless-stopped
    environment:
      IMAGE_SERVICE_URL: http://image-service:9002
      TTS_SERVICE_URL: http://tts-service:9003
      STT_SERVICE_URL: http://stt-service:9004
      MEDIA_ROOT: /media
    volumes:
      - media_data:/media
    networks:
      - dnd_net
    depends_on:
      image-service:
        condition: service_healthy
      tts-service:
        condition: service_healthy
      stt-service:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8004/health || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5
    expose:
      - "8004"

  # -------------------------------------------------
  # A2A Agent Services
  # -------------------------------------------------

  character-creator:
    build:
      context: ./agents/character-creator
      dockerfile: Dockerfile
    restart: unless-stopped
    environment:
      STATE_MCP_URL: http://state-mcp:8001
      MEDIA_MCP_URL: http://media-mcp:8004
      LLM_SERVICE_URL: http://llm-service:9001
      REDIS_URL: redis://redis:6379/0
      PORT: 8010
    networks:
      - dnd_net
    depends_on:
      state-mcp:
        condition: service_healthy
      media-mcp:
        condition: service_healthy
      llm-service:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8010/.well-known/agent.json || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5
    expose:
      - "8010"

  campaign-designer:
    build:
      context: ./agents/campaign-designer
      dockerfile: Dockerfile
    restart: unless-stopped
    environment:
      STATE_MCP_URL: http://state-mcp:8001
      KNOWLEDGE_MCP_URL: http://knowledge-mcp:8003
      LLM_SERVICE_URL: http://llm-service:9001
      REDIS_URL: redis://redis:6379/0
      PORT: 8011
    networks:
      - dnd_net
    depends_on:
      state-mcp:
        condition: service_healthy
      knowledge-mcp:
        condition: service_healthy
      llm-service:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8011/.well-known/agent.json || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5
    expose:
      - "8011"

  memory-agent:
    build:
      context: ./agents/memory-agent
      dockerfile: Dockerfile
    restart: unless-stopped
    environment:
      STATE_MCP_URL: http://state-mcp:8001
      MEMORY_MCP_URL: http://memory-mcp:8002
      LLM_SERVICE_URL: http://llm-service:9001
      REDIS_URL: redis://redis:6379/0
      PORT: 8014
    networks:
      - dnd_net
    depends_on:
      state-mcp:
        condition: service_healthy
      memory-mcp:
        condition: service_healthy
      llm-service:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8014/.well-known/agent.json || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5
    expose:
      - "8014"

  dm-agent:
    build:
      context: ./agents/dm-agent
      dockerfile: Dockerfile
    restart: unless-stopped
    environment:
      STATE_MCP_URL: http://state-mcp:8001
      KNOWLEDGE_MCP_URL: http://knowledge-mcp:8003
      MEDIA_MCP_URL: http://media-mcp:8004
      MEMORY_AGENT_URL: http://memory-agent:8014
      LLM_SERVICE_URL: http://llm-service:9001
      REDIS_URL: redis://redis:6379/0
      PORT: 8012
    networks:
      - dnd_net
    depends_on:
      state-mcp:
        condition: service_healthy
      knowledge-mcp:
        condition: service_healthy
      media-mcp:
        condition: service_healthy
      memory-agent:
        condition: service_healthy
      llm-service:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8012/.well-known/agent.json || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5
    expose:
      - "8012"

  npc-agent:
    build:
      context: ./agents/npc-agent
      dockerfile: Dockerfile
    restart: unless-stopped
    environment:
      STATE_MCP_URL: http://state-mcp:8001
      KNOWLEDGE_MCP_URL: http://knowledge-mcp:8003
      MEDIA_MCP_URL: http://media-mcp:8004
      LLM_SERVICE_URL: http://llm-service:9001
      REDIS_URL: redis://redis:6379/0
      PORT: 8013
    networks:
      - dnd_net
    depends_on:
      state-mcp:
        condition: service_healthy
      knowledge-mcp:
        condition: service_healthy
      media-mcp:
        condition: service_healthy
      llm-service:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8013/.well-known/agent.json || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5
    expose:
      - "8013"

  # -------------------------------------------------
  # API
  # -------------------------------------------------

  api:
    build:
      context: ./api
      dockerfile: Dockerfile
    restart: unless-stopped
    environment:
      DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      REDIS_URL: redis://redis:6379/0
      JWT_SECRET: ${JWT_SECRET}
      JWT_ACCESS_EXPIRE_MINUTES: ${JWT_ACCESS_EXPIRE_MINUTES:-60}
      STATE_MCP_URL: http://state-mcp:8001
      MEDIA_ROOT: /media
      CHARACTER_CREATOR_URL: http://character-creator:8010
      CAMPAIGN_DESIGNER_URL: http://campaign-designer:8011
      DM_AGENT_URL: http://dm-agent:8012
      NPC_AGENT_URL: http://npc-agent:8013
      MEMORY_AGENT_URL: http://memory-agent:8014
    volumes:
      - media_data:/media
    networks:
      - dnd_net
    depends_on:
      state-mcp:
        condition: service_healthy
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
      character-creator:
        condition: service_healthy
      campaign-designer:
        condition: service_healthy
      dm-agent:
        condition: service_healthy
      npc-agent:
        condition: service_healthy
      memory-agent:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5
    ports:
      - "${API_PORT:-8000}:8000"

  # -------------------------------------------------
  # Frontend
  # -------------------------------------------------

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      args:
        VITE_API_BASE_URL: ${VITE_API_BASE_URL:-http://localhost:8000/api/v1}
    restart: unless-stopped
    networks:
      - dnd_net
    depends_on:
      - api
    ports:
      - "${FRONTEND_PORT:-3000}:80"
```

**Frontend Dockerfile note:** The frontend is built as a static bundle (Vite `npm run build`) served by Nginx. The `VITE_API_BASE_URL` build arg is baked into the bundle at build time.

---

## 10. Environment Variables

```bash
# .env.example
# Copy to .env and fill in all values before running docker-compose up.

# -------------------------------------------------
# PostgreSQL
# -------------------------------------------------
POSTGRES_DB=dnd
POSTGRES_USER=dnd_user
POSTGRES_PASSWORD=changeme_postgres

# -------------------------------------------------
# Neo4j
# -------------------------------------------------
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme_neo4j

# -------------------------------------------------
# JWT Authentication
# -------------------------------------------------
JWT_SECRET=changeme_jwt_secret_64_chars_min  # openssl rand -hex 32
JWT_ACCESS_EXPIRE_MINUTES=60                 # No refresh tokens; set longer if needed

# -------------------------------------------------
# LLM Provider (llm-service)
# -------------------------------------------------
LLM_PROVIDER=gemini                          # gemini | openai | anthropic
GEMINI_API_KEY=AIza...                       # Also used by memory-mcp (gemini-embedding-2)
GEMINI_MODEL=gemini-2.5-flash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6

# -------------------------------------------------
# Image Provider (image-service)
# -------------------------------------------------
IMAGE_PROVIDER=gemini                        # gemini | runware | dalle
GEMINI_IMAGE_MODEL=gemini-2.5-flash-preview-image-generation
RUNWARE_API_KEY=                             # Required if IMAGE_PROVIDER=runware

# -------------------------------------------------
# TTS Provider (tts-service)
# -------------------------------------------------
TTS_PROVIDER=openai                          # openai | elevenlabs | gemini
OPENAI_TTS_MODEL=gpt-4o-mini-tts
ELEVENLABS_API_KEY=                          # Required if TTS_PROVIDER=elevenlabs

# -------------------------------------------------
# STT Provider (stt-service)
# -------------------------------------------------
STT_PROVIDER=whisper_local                   # whisper_local | openai | gemini
WHISPER_MODEL=small                          # tiny | base | small | medium | large

# -------------------------------------------------
# Ports
# -------------------------------------------------
API_PORT=8000
FRONTEND_PORT=3000

# -------------------------------------------------
# Frontend build
# -------------------------------------------------
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

**Variable inventory by service:**

| Variable | Used by | Required |
|---|---|---|
| `POSTGRES_DB/USER/PASSWORD` | postgres, state-mcp, api, llm-service, image-service, tts-service, stt-service | Yes |
| `NEO4J_USER/PASSWORD` | neo4j, knowledge-mcp | Yes |
| `JWT_SECRET` | api | Yes |
| `JWT_ACCESS_EXPIRE_MINUTES` | api | No (default: 60) |
| `LLM_PROVIDER` | llm-service | No (default: gemini) |
| `GEMINI_API_KEY` | llm-service, image-service, memory-mcp | Yes (if using Gemini) |
| `GEMINI_MODEL` | llm-service | No (default: gemini-2.5-flash) |
| `GEMINI_IMAGE_MODEL` | image-service | No (default: gemini-2.5-flash-preview-image-generation) |
| `IMAGE_PROVIDER` | image-service | No (default: gemini) |
| `RUNWARE_API_KEY` | image-service | Yes (if `IMAGE_PROVIDER=runware`) |
| `TTS_PROVIDER` | tts-service | No (default: openai) |
| `STT_PROVIDER` | stt-service | No (default: whisper_local) |
| `OPENAI_API_KEY` | llm-service, image-service, tts-service, stt-service | Yes (if using any OpenAI service) |
| `OPENAI_TTS_MODEL` | tts-service | No (default: gpt-4o-mini-tts) |
| `ELEVENLABS_API_KEY` | tts-service | Yes (if `TTS_PROVIDER=elevenlabs`) |
| `WHISPER_MODEL` | stt-service | No (default: small) |
| `ANTHROPIC_API_KEY` | llm-service | Yes (if `LLM_PROVIDER=anthropic`) |
| `API_PORT` | docker-compose | No (default: 8000) |
| `FRONTEND_PORT` | docker-compose | No (default: 3000) |
| `VITE_API_BASE_URL` | frontend build | Yes |

---

## 11. Resolved Decisions

**1. Database migrations — Alembic**
Use Alembic for all PostgreSQL schema changes. An `alembic/` directory lives in `api/`. Migrations applied at startup before the application accepts traffic.

**2. Concurrent turn submission — Redis per-campaign lock**
The API acquires a Redis SETNX lock keyed on `lock:campaign:{campaign_id}` before spawning any A2A task. If the lock is already held, the endpoint returns 409 Conflict. Lock TTL is 60 seconds. Released in a `finally` block after the A2A task completes.

**3. Audio upload format — accept webm, convert in stt-service**
The browser uploads raw `audio/webm` (native MediaRecorder output). No browser-side conversion. `stt-service` runs `ffmpeg` internally to convert webm to WAV before passing to Whisper.

**4. SSE and media auth — httpOnly cookies**
The API issues `httpOnly`, `SameSite=strict` cookies on login. SSE connections (`withCredentials: true`) and `<audio src>` / `<img src>` requests use the cookie automatically. No `?token=` query parameter needed.

**5. Whisper cold start — model-load readiness gate**
`stt-service /health` returns `{ model_loaded: false }` until Whisper finishes loading. Docker Compose healthcheck for `stt-service` checks for `model_loaded: true`. `media-mcp` has `depends_on: stt-service: condition: service_healthy`. `tts-service` has no equivalent gate — it starts immediately.

**6. Neo4j full-text index — init script**
`knowledge-mcp` runs the following Cypher at startup (idempotent):
```cypher
CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
FOR (n:NPC|Location|Faction|Item)
ON EACH [n.name, n.description];
```

**7. Agent context strategy — stateless, Postgres-sourced per turn**
Agents hold no in-memory conversation state between A2A calls. On every turn each agent fetches what it needs fresh from state-mcp and assembles a full LLM prompt from these layers (in order):

```
[System]          Campaign plan · character sheet · visual style
[Long-term memory] campaigns.long_term_memory (compressed markdown, from memory-agent)
[Semantic recall]  Top-8 Qdrant hits for the current player message (from memory-agent)
[Recent events]    campaigns.short_term_memory list (from memory-agent)
[World context]    Neo4j graph excerpt (from knowledge-mcp)
[Recent turns]     Last 10 turns from turns table, NPC turns excluded (from state-mcp)
[User]            Player's current message
```

This means:
- No Redis conversation cache; no sticky routing; any agent pod can handle any turn.
- `short_term_memory` / `long_term_memory` in Postgres are the memory-agent's compression state, not the LLM's context source — the LLM gets raw recent turns directly.
- npc-agent follows the same pattern but replaces "recent turns" with its 3-layer context (preamble + briefing + conv history).

**8. Opening scene trigger — frontend sentinel**
When `set_phase('active')` is called, the frontend receives `phase_change` via SSE and navigates to `/campaigns/:id/play`. On mount, if `turns.length == 0`, the frontend sends `POST /message { content: "__opening_scene__" }`. The API detects this sentinel and substitutes the opening prompt before dispatching to dm-agent.

---

### Critical Files for Implementation (in order)

| Priority | File | Why first |
|---|---|---|
| 1 | api/alembic/versions/001_initial.sql | Schema (+ STM/LTM cols) must exist before anything else |
| 2 | mcp-servers/state-mcp/app/main.py | Every agent and MCP server depends on state-mcp |
| 3 | services/llm-service/app/main.py | All agents call llm-service for LLM inference |
| 4 | services/stt-service/app/main.py | media-mcp health depends on Whisper load; slow startup |
| 5 | services/tts-service/app/main.py | media-mcp health depends on this |
| 6 | services/image-service/app/main.py | media-mcp health depends on this |
| 7 | mcp-servers/memory-mcp/app/main.py | memory-agent uses this for Qdrant + gemini-embedding-2 |
| 8 | mcp-servers/knowledge-mcp/app/main.py | dm-agent and campaign-designer need world context |
| 9 | mcp-servers/media-mcp/app/main.py | character-creator, dm-agent, npc-agent need media |
| 10 | agents/shared/a2a.py | A2A protocol schemas used by all agent services |
| 11 | agents/memory-agent/app/agent.py | dm-agent depends on this; implements SimpleMemorySystem |
| 12 | agents/character-creator/app/agent.py | First user-facing A2A agent |
| 13 | agents/campaign-designer/app/agent.py | Second setup A2A agent |
| 14 | agents/dm-agent/app/agent.py | Core gameplay A2A agent |
| 15 | agents/npc-agent/app/agent.py | NPC conversation A2A agent |
| 16 | api/app/dispatcher.py | Routes all traffic to correct A2A agent |
