import json
from fastapi import FastAPI
from .agent import run

app = FastAPI(title="character-creator")


@app.get("/.well-known/agent.json")
async def agent_card():
    return {
        "name": "character-creator",
        "description": "Guides players through D&D character creation",
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
