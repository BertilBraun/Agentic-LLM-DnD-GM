# Setup

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

## 1. Configure environment

Copy the example env file and fill in your API keys:

```bash
cp .env.example .env
```

Required keys (at minimum one LLM provider):

| Variable            | Where to get it                                        |
| ------------------- | ------------------------------------------------------ |
| `GEMINI_API_KEY`    | [aistudio.google.com](https://aistudio.google.com)     |
| `OPENAI_API_KEY`    | [platform.openai.com](https://platform.openai.com)     |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |

Set `LLM_PROVIDER`, `IMAGE_PROVIDER`, `TTS_PROVIDER` to match whichever keys you provided.

## 2. Start the application

```bash
docker compose up -d
```

The first run builds all images and may take several minutes.

| Service  | URL                     |
| -------- | ----------------------- |
| Frontend | <http://localhost:3000> |
| API      | <http://localhost:8000> |

## 3. Start observability (Langfuse)

```bash
docker compose -f observability/docker-compose.yml up -d
```

Wait ~2 minutes for first-boot migrations to finish, then open **<http://localhost:3001>** and create an account.

Navigate to **Settings → API Keys**, create a key pair, and copy the values into `.env`:

```text
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://host.docker.internal:3001
```

> Langfuse is optional — the app works without it. Skip this step and leave the keys blank to disable tracing.
