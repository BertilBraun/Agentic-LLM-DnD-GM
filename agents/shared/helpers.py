"""Shared async helper functions for all agent services."""
from __future__ import annotations

import json
import logging
import os

import httpx
import redis.asyncio as aioredis

LLM_SERVICE_URL = os.environ.get("LLM_SERVICE_URL", "http://llm-service:9001")
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

logger = logging.getLogger(__name__)


async def call_mcp(base_url: str, tool: str, body: dict, campaign_id: str, timeout: float = 60.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{base_url}/tools/{tool}",
            json=body,
            headers={"X-Campaign-ID": campaign_id},
        )
        resp.raise_for_status()
        return resp.json()


async def call_llm(messages: list[dict], response_format: str = "text", timeout: float = 60.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{LLM_SERVICE_URL}/generate",
            json={"messages": messages, "response_format": response_format},
        )
        resp.raise_for_status()
        return resp.json()


async def call_llm_json(messages: list[dict], timeout: float = 60.0) -> dict:
    """Call LLM with json response_format and parse the returned text as JSON."""
    resp = await call_llm(messages, response_format="json", timeout=timeout)
    return json.loads(resp["text"])


async def publish_event(redis_client: aioredis.Redis, campaign_id: str, event: dict) -> None:
    await redis_client.publish(f"sse:campaign:{campaign_id}", json.dumps(event))
