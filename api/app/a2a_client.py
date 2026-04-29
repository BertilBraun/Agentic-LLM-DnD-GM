"""Thin A2A JSON-RPC client — sends tasks/send to agent services."""
from __future__ import annotations

import uuid

import httpx
from pydantic import BaseModel


class A2AResult(BaseModel):
    output: str = ""
    done: bool = True


class A2AResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: A2AResult
    id: int | str | None = None


async def send_task(agent_url: str, campaign_id: str, message: str, timeout: float = 120) -> A2AResponse:
    task_id = str(uuid.uuid4())
    payload = {
        "jsonrpc": "2.0",
        "method": "tasks/send",
        "params": {"task_id": task_id, "campaign_id": campaign_id, "message": message},
        "id": 1,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(agent_url + "/", json=payload)
        resp.raise_for_status()
        return A2AResponse.model_validate(resp.json())
