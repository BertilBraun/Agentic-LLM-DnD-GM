# Voice-Driven D&D Game Master

A production-grade multi-agent AI system that acts as a real-time Dungeon Master. Built to demonstrate practical experience with knowledge graphs, vector databases, multi-agent orchestration, and observable production LLM systems.

→ [Product overview](documentation/PRODUCT.md) · [Setup guide](SETUP.md)

---

## Architecture Overview

```text
Browser (React + TypeScript)
        │  REST + Server-Sent Events (JWT, httpOnly cookie)
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI API  :8000                        │
│  JWT auth · Alembic migrations · A2A dispatcher             │
│  SSE publisher ──► Redis pub/sub :6379                      │
└───────────────────────┬─────────────────────────────────────┘
                        │ A2A  (JSON-RPC 2.0 over HTTP)
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
  ┌──────────────┐ ┌──────────┐ ┌──────────────────────────────┐
  │  dm-agent    │ │npc-agent │ │ character-creator             │
  │    :8012     │ │  :8013   │ │ campaign-designer             │
  └──────┬───────┘ └────┬─────┘ │ memory-agent    (all :8010+) │
         │              │       └──────────────────────────────┘
         └──────┬────────┘
                │ MCP over HTTP  (X-Campaign-ID header)
    ┌───────────┼──────────────┬──────────────┐
    ▼           ▼              ▼              ▼
┌─────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────┐
│state-mcp│ │memory-mcp│ │knowledge-mcp │ │media-mcp │
│  :8001  │ │  :8002   │ │    :8003     │ │  :8004   │
└────┬────┘ └────┬─────┘ └──────┬───────┘ └────┬─────┘
     │           │              │               │
     ▼           ▼              ▼               ▼
 PostgreSQL    Qdrant         Neo4j       llm / image /
 (campaigns,  (semantic      (world       tts / stt
  turns, NPCs) memory,        knowledge    services
  Alembic)    1536-dim        graph)       :9001–9004
              embeddings)
```

---

## Key Engineering Decisions

### Multi-Agent Architecture via Google A2A

The system is decomposed into five specialised agents, each a standalone FastAPI service communicating via the [Google Agent-to-Agent (A2A) protocol](https://google.github.io/A2A/) — JSON-RPC 2.0 over HTTP with a `/.well-known/agent.json` card endpoint:

| Agent | Port | Responsibility |
| --- | --- | --- |
| `character-creator` | 8010 | Guided character creation wizard |
| `campaign-designer` | 8011 | Campaign plan generation, world seeding |
| `dm-agent` | 8012 | Core gameplay: narration, scene images, NPC invocation |
| `npc-agent` | 8013 | Extended NPC dialogue; returns summary to DM |
| `memory-agent` | 8014 | Semantic recall + hierarchical memory summarisation |

The `api` service reads campaign phase and active NPC state from the database, then dispatches the incoming player message to the correct agent over A2A. The DM agent can invoke the NPC agent (also over A2A) mid-turn, and the NPC agent's summary is published back to the DM to continue narration.

### MCP Tool Servers

Agents interact with shared infrastructure through four [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) HTTP tool servers. Every request carries an `X-Campaign-ID` header extracted by shared middleware, giving each tool call implicit campaign scope without threading IDs through every function signature:

| MCP Server | Port | Backing Store | Tools |
| --- | --- | --- | --- |
| `state-mcp` | 8001 | PostgreSQL | Campaign/character/NPC CRUD, turn log, routing state |
| `memory-mcp` | 8002 | Qdrant | Semantic store/recall using Gemini embeddings |
| `knowledge-mcp` | 8003 | Neo4j | Entity/relationship extraction, world context retrieval |
| `media-mcp` | 8004 | Volume + provider services | TTS, STT, image generation, file I/O |

### Knowledge Graph (Neo4j)

`knowledge-mcp` extracts named entities (NPCs, locations, factions, items) and their relationships from narrative text using a structured LLM call after every DM turn. These are written to Neo4j. Before each DM response the `get_world_context` tool traverses the graph from nodes matching the player's message, returning a compact context string that grounds the LLM in established lore — preventing hallucinated contradictions as campaigns grow.

### Vector Database (Qdrant)

`memory-mcp` stores per-turn narrative events as 1536-dimensional vectors using the `gemini-embedding-2` model. The `memory-agent` performs two operations each turn:

1. **Recall** — semantic nearest-neighbour search over campaign events to surface relevant past context
2. **Compression** — detects when short-term memory exceeds a threshold, summarises it with an LLM call, and writes the summary back as the new long-term baseline

This gives the DM agent both recency (short-term list) and depth (long-term summary) without unbounded token growth.

### Provider Services (Swappable LLM/Media Backends)

Four thin adapter services (`llm-service`, `image-service`, `tts-service`, `stt-service`) wrap external AI APIs behind a uniform FastAPI contract. Each provides:

- **Redis caching** — repeated requests return cached responses without hitting the API (24 h TTL for audio, 1 h for text)
- **Usage logging** — token counts written to PostgreSQL `llm_usage` table
- **Provider switching** — set `LLM_PROVIDER=anthropic`, `TTS_PROVIDER=gemini`, etc. in `.env` without changing agent code

### Real-time Event Delivery (SSE over Redis pub/sub)

The `api` dispatches agent tasks as a fire-and-forget background task and immediately returns `200 OK`. Agents publish typed SSE events (`dm_text`, `audio_ready`, `scene_ready`, `npc_introduced`, …) directly to a Redis channel keyed by `campaign_id`. The frontend holds an `EventSource` connection to `/campaigns/{id}/stream` which relays messages from that channel. This decouples agent execution time from HTTP response latency and allows the UI to stream partial results (text first, then audio, then image) as they become available.

---

## Service Map

| Service | Stack | Purpose |
| --- | --- | --- |
| `api` | FastAPI, SQLAlchemy, Alembic | Auth, campaign CRUD, SSE stream, A2A dispatch |
| `dm-agent` | FastAPI, httpx | Core DM logic, scene generation orchestration |
| `npc-agent` | FastAPI, httpx | NPC dialogue loop |
| `character-creator` | FastAPI, httpx | Character creation wizard |
| `campaign-designer` | FastAPI, httpx | Campaign plan + world seed |
| `memory-agent` | FastAPI, httpx | Memory recall + summarisation |
| `state-mcp` | FastAPI, SQLAlchemy | PostgreSQL state tool server |
| `memory-mcp` | FastAPI, Qdrant client | Vector memory tool server |
| `knowledge-mcp` | FastAPI, Neo4j driver | Knowledge graph tool server |
| `media-mcp` | FastAPI, httpx | Media orchestration tool server |
| `llm-service` | FastAPI | LLM adapter (Gemini / OpenAI / Anthropic) |
| `image-service` | FastAPI | Image adapter (Gemini / Runware / DALL-E) |
| `tts-service` | FastAPI | TTS adapter (OpenAI / Gemini / ElevenLabs) |
| `stt-service` | FastAPI | STT adapter (Whisper local / OpenAI) |
| `frontend` | React 18, TypeScript, Vite | SPA served by nginx |
| `postgres` | PostgreSQL 16 | Relational state, turn history, usage stats |
| `qdrant` | Qdrant 1.9 | Semantic memory vectors |
| `neo4j` | Neo4j 5 Community | World knowledge graph |
| `redis` | Redis 7 | SSE pub/sub, LLM/TTS response cache |

---

## Observability

All services run under `opentelemetry-instrument` with OTLP export to Tempo. Prometheus scrapes a metrics port on every service. Logs are forwarded via Promtail to Loki. Everything is visible in a single Grafana instance.

| Tool | Port | What it covers |
| --- | --- | --- |
| **Grafana** | 3002 | Unified dashboards: traces (Tempo), metrics (Prometheus), logs (Loki) |
| **Tempo** | 3200 | Distributed tracing — full request spans across API → agent → MCP → service |
| **Prometheus** | 9090 | RED metrics (requests, errors, duration) per service |
| **Loki + Promtail** | 3100 | Structured log aggregation from all Docker containers |
| **Langfuse** | 3001 | LLM-specific tracing: input/output, token usage, latency per generation |

Langfuse is optional — the system works without it. Leave `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` blank to disable.

---

## Data Flow: A Single Player Turn

```
1.  Player speaks → frontend uploads audio → api → media-mcp → stt-service
                                                                (Whisper local)
2.  api acquires Redis lock (prevents concurrent turns)
3.  api reads routing state from state-mcp (phase, active_npc_id)
4.  api dispatches to dm-agent over A2A (fire-and-forget background task)

5.  dm-agent gathers in parallel:
      · campaign context + recent turns  (state-mcp → PostgreSQL)
      · world context                    (knowledge-mcp → Neo4j graph traversal)
      · semantic memory recall           (memory-agent → Qdrant ANN search)

6.  dm-agent calls llm-service (structured JSON response):
      gm_speech, scene_description, memory_note, invoke_npc?

7.  dm-agent logs DM turn to state-mcp, publishes dm_text SSE event
    → frontend renders DM text immediately

8.  Background (parallel):
      · TTS: media-mcp → tts-service → audio file → audio_ready SSE
      · Image: media-mcp → image-service → scene file → scene_ready SSE
             (image_path logged to state-mcp for reload persistence)
      · Memory update: memory-agent stores event, compresses if needed
      · World update: knowledge-mcp extracts entities → Neo4j

9.  If invoke_npc is set:
      · dm-agent → save NPC, generate portrait + opening audio (parallel)
      · dm-agent → set_active_npc in state-mcp
      · publishes npc_introduced + npc_speech SSE events
      · api releases lock; next player message routes to npc-agent

10. npc-agent turn (same structure; calls state-mcp, knowledge-mcp, llm-service)
    On conversation end: publishes summary → dm-agent picks up next turn
```

---

## Repository Structure

```text
├── api/                     FastAPI backend (auth, campaigns, SSE, dispatcher)
│   └── alembic/             Database migrations
├── agents/
│   ├── shared/              A2A schemas, MCP client helpers, Pydantic response models
│   ├── dm-agent/
│   ├── npc-agent/
│   ├── character-creator/
│   ├── campaign-designer/
│   └── memory-agent/
├── mcp-servers/
│   ├── shared/              CampaignIDMiddleware, shared schemas
│   ├── state-mcp/
│   ├── memory-mcp/
│   ├── knowledge-mcp/
│   └── media-mcp/
├── services/
│   ├── llm-service/
│   ├── image-service/
│   ├── tts-service/
│   └── stt-service/
├── frontend/                React 18 SPA (TypeScript, Vite, Zustand)
├── observability/           Grafana, Prometheus, Tempo, Loki, Promtail config
├── docker-compose.yml       Full stack orchestration (22 services)
├── SETUP.md                 Installation and configuration guide
└── documentation/
    └── PRODUCT.md           Product overview and how-to-play guide
```

---

## Tech Stack Summary

| Concern | Technology | Why |
| --- | --- | --- |
| Agent communication | Google A2A (JSON-RPC 2.0) | Standard protocol; enforces service boundaries; agent card discovery |
| Tool calling | MCP over HTTP | Standardised tool interface; campaign-scoped by header middleware |
| LLM inference | Gemini 2.5 Flash (default) | Structured output, low cost; swappable to Anthropic/OpenAI |
| Embeddings | Gemini `gemini-embedding-2` (1536-dim) | Same API key; no local model to maintain |
| Vector store | Qdrant | Semantic memory recall, ANN search, persistent storage |
| Knowledge graph | Neo4j 5 + Cypher | Entity/relationship world model; graph traversal for context retrieval |
| Relational DB | PostgreSQL 16 + Alembic | Campaign state, turn history, usage accounting |
| Cache / pub-sub | Redis 7 | LLM/TTS response caching; SSE fan-out across API replicas |
| API framework | FastAPI + Pydantic v2 | Async-native, typed end-to-end, automatic OpenAPI docs |
| Frontend | React 18, TypeScript, Vite, Zustand | SSE + Web Audio API; typed API client |
| Containerisation | Docker Compose | 22-service stack; health checks on every service |
| Distributed tracing | OpenTelemetry → Grafana Tempo | Full request trace: browser → API → agent → MCP → provider |
| Metrics | Prometheus + Grafana | RED metrics per service, service graph |
| Log aggregation | Loki + Promtail | Structured logs from all containers |
| LLM observability | Langfuse 3 (optional) | Per-generation input/output/token tracking |
| Auth | JWT HS256, httpOnly cookie | Stateless; cookie enables SSE and media auth without JS token management |

---

## Setup

See [SETUP.md](SETUP.md) for the full guide. The short version:

```bash
cp .env.example .env          # fill in at least one LLM provider key
docker compose up -d          # builds and starts all 22 services
# Frontend:  http://localhost:3000
# API docs:  http://localhost:8000/docs
# Grafana:   http://localhost:3002
```

Optional Langfuse LLM tracing:

```bash
docker compose -f observability/docker-compose.yml up -d
# Then add LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY to .env and restart llm-service
```
