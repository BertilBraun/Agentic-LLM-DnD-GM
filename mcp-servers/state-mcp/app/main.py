from fastapi import FastAPI
from .middleware import CampaignIDMiddleware
from .tools import campaign, character, npc, turns, memory

app = FastAPI(title="state-mcp")
app.add_middleware(CampaignIDMiddleware)

app.include_router(campaign.router)
app.include_router(character.router)
app.include_router(npc.router)
app.include_router(turns.router)
app.include_router(memory.router)


@app.get("/health")
async def health():
    return {"ok": True}
