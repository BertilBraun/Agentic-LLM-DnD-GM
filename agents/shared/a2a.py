"""Google A2A protocol — shared schemas for all agent services."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel


class AgentCard(BaseModel):
    name: str
    description: str
    version: str = "1.0.0"
    skills: list[dict[str, Any]] = []


class A2ATaskParams(BaseModel):
    task_id: str
    campaign_id: str
    message: str


class A2ARequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str = "tasks/send"
    params: A2ATaskParams
    id: Optional[Any] = None


class A2AResult(BaseModel):
    output: str
    done: bool = True


class A2AResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: A2AResult
    id: Optional[Any] = None
