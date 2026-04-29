"""Shared async helper functions for all agent services."""
from __future__ import annotations

import json
import logging
import os
from typing import TypeVar, Type

import httpx
import redis.asyncio as aioredis
from pydantic import BaseModel

LLM_SERVICE_URL = os.environ.get("LLM_SERVICE_URL", "http://llm-service:9001")
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

# MCP / agent service URLs — imported by all agent modules
STATE_MCP_URL = os.environ.get("STATE_MCP_URL", "http://state-mcp:8001")
MEMORY_MCP_URL = os.environ.get("MEMORY_MCP_URL", "http://memory-mcp:8002")
KNOWLEDGE_MCP_URL = os.environ.get("KNOWLEDGE_MCP_URL", "http://knowledge-mcp:8003")
MEDIA_MCP_URL = os.environ.get("MEDIA_MCP_URL", "http://media-mcp:8004")
MEMORY_AGENT_URL = os.environ.get("MEMORY_AGENT_URL", "http://memory-agent:8014")

# DM voice — shared between character-creator and dm-agent
DM_VOICE_ID: str = "ash"
DM_VOICE_INSTRUCTIONS: str = "Speak in a deep, authoritative voice with dramatic pauses and varied intonation to bring the fantasy world to life"

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_redis: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL)
    return _redis


async def call_mcp(base_url: str, tool: str, body: dict, campaign_id: str, response_model: Type[T], timeout: float = 60.0) -> T:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{base_url}/tools/{tool}",
            json=body,
            headers={"X-Campaign-ID": campaign_id},
        )
        resp.raise_for_status()
        return response_model.model_validate(resp.json())


async def call_llm(
    messages: list[dict],
    response_format: str = "text",
    response_json_schema: dict | None = None,
    timeout: float = 60.0,
) -> dict:
    payload: dict = {"messages": messages, "response_format": response_format}
    if response_json_schema is not None:
        payload["response_json_schema"] = response_json_schema
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{LLM_SERVICE_URL}/generate", json=payload)
        resp.raise_for_status()
        return resp.json()


async def call_llm_structured(
    messages: list[dict],
    response_model: Type[T],
    timeout: float = 60.0,
) -> T:
    """Call LLM with the Pydantic model's JSON schema for constrained generation, then validate."""
    schema = response_model.model_json_schema()
    resp = await call_llm(messages, response_format="json", response_json_schema=schema, timeout=timeout)
    return response_model.model_validate_json(resp["text"])


async def publish_event(campaign_id: str, event: dict) -> None:
    await get_redis().publish(f"sse:campaign:{campaign_id}", json.dumps(event))
