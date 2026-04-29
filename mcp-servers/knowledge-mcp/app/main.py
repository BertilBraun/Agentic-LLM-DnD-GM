from contextlib import asynccontextmanager
from fastapi import FastAPI
from .middleware import CampaignIDMiddleware
from .neo4j_driver import ensure_indexes, close_driver
from .tools import knowledge


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_indexes()
    yield
    await close_driver()


app = FastAPI(title="knowledge-mcp", lifespan=lifespan)
app.add_middleware(CampaignIDMiddleware)

app.include_router(knowledge.router)


@app.get("/health")
async def health():
    return {"ok": True}
