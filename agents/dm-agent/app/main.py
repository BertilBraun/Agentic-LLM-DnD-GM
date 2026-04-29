import json
from fastapi import FastAPI
from .agent import run

app = FastAPI(title="dm-agent")


@app.get("/.well-known/agent.json")
async def agent_card():
    return {
        "name": "dm-agent",
        "description": "Core Dungeon Master agent for active gameplay",
        "version": "1.0.0",
        "skills": [],
    }


@app.post("/")
async def handle(req: dict):
    params = req.get("params", {})
    campaign_id = params.get("campaign_id", "")
    message = params.get("message", "")
    output = await run(campaign_id, message)
    return {
        "jsonrpc": "2.0",
        "result": {"output": output, "done": True},
        "id": req.get("id"),
    }
