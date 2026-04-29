from fastapi import FastAPI

from shared.a2a import A2ARequest, A2AResponse, A2AResult, AgentCard
from .agent import run

app = FastAPI(title='npc-agent')


@app.get('/.well-known/agent.json')
async def agent_card() -> dict:
    return AgentCard(
        name='npc-agent',
        description='Handles multi-turn NPC conversations',
    ).model_dump()


@app.post('/')
async def handle(req: A2ARequest) -> A2AResponse:
    output = await run(req.params.campaign_id, req.params.message)
    return A2AResponse(result=A2AResult(output=output), id=req.id)
