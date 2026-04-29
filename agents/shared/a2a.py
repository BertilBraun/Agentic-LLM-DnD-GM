"""Google A2A protocol — shared schemas and base router for all agent services."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


class AgentCard(BaseModel):
    name: str
    description: str
    version: str = "1.0.0"
    skills: list[dict] = []


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


def make_agent_router(card: AgentCard, handler) -> APIRouter:
    """
    Returns a FastAPI router with:
      GET  /.well-known/agent.json  -> AgentCard
      POST /                        -> JSON-RPC 2.0 tasks/send
    handler(params: A2ATaskParams) -> A2AResult (or coroutine)
    """
    router = APIRouter()

    @router.get("/.well-known/agent.json")
    async def agent_card_endpoint():
        return card.model_dump()

    @router.post("/")
    async def handle_task(req: A2ARequest):
        import asyncio
        result = handler(req.params)
        if asyncio.iscoroutine(result):
            result = await result
        return A2AResponse(result=result, id=req.id)

    return router
