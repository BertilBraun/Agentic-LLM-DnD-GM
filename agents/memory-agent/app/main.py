import json

from fastapi import FastAPI

from shared.a2a import A2ARequest, A2AResponse, A2AResult, AgentCard
from .agent import run

app = FastAPI(title='memory-agent')


@app.get('/.well-known/agent.json')
async def agent_card() -> dict:
    return AgentCard(
        name='memory-agent',
        description='Memory consolidation and semantic recall',
    ).model_dump()


@app.post('/')
async def handle(req: A2ARequest) -> A2AResponse:
    try:
        payload = json.loads(req.params.message)
    except json.JSONDecodeError:
        payload = {}
    query: str = payload.get('query', '')
    new_event: str = payload.get('new_event', '')
    result = await run(req.params.campaign_id, query, new_event)
    return A2AResponse(result=A2AResult(output=json.dumps(result)), id=req.id)
