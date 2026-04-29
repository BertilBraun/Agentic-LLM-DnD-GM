from contextlib import asynccontextmanager
from fastapi import FastAPI
from shared.middleware import CampaignIDMiddleware
from .qdrant_client import ensure_collection
from .tools import memory


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_collection()
    yield


app = FastAPI(title="memory-mcp", lifespan=lifespan)
app.add_middleware(CampaignIDMiddleware)
app.include_router(memory.router)


@app.get("/health")
async def health():
    return {"ok": True}
