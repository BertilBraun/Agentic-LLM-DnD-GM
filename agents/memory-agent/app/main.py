import os
from fastapi import FastAPI
from pydantic import BaseModel
from .agent import run

app = FastAPI(title="memory-agent")


class TaskParams(BaseModel):
    task_id: str
    campaign_id: str
    message: str  # JSON string: {"query": "...", "new_event": "..."}


@app.get("/.well-known/agent.json")
async def agent_card():
    return {
        "name": "memory-agent",
        "description": "Memory consolidation and semantic recall",
        "version": "1.0.0",
        "skills": [],
    }


@app.post("/")
async def handle(req: dict):
    params = req.get("params", {})
    campaign_id = params.get("campaign_id", "")
    import json
    try:
        payload = json.loads(params.get("message", "{}"))
    except Exception:
        payload = {}
    query = payload.get("query", "")
    new_event = payload.get("new_event", "")
    result = await run(campaign_id, query, new_event)
    return {
        "jsonrpc": "2.0",
        "result": {"output": json.dumps(result), "done": True},
        "id": req.get("id"),
    }
