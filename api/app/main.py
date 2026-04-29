from fastapi import FastAPI
from .auth.router import router as auth_router
from .campaigns.router import router as campaigns_router
from .game.router import router as game_router

app = FastAPI(title="llm-dnd-api", version="1.0.0")

PREFIX = "/api/v1"

app.include_router(auth_router, prefix=PREFIX)
app.include_router(campaigns_router, prefix=PREFIX)
app.include_router(game_router, prefix=PREFIX)


@app.get("/health")
async def health():
    return {"ok": True}
