from fastapi import FastAPI
from shared.middleware import CampaignIDMiddleware
from .tools import image, tts, stt

app = FastAPI(title="media-mcp")
app.add_middleware(CampaignIDMiddleware)

app.include_router(image.router)
app.include_router(tts.router)
app.include_router(stt.router)


@app.get("/health")
async def health():
    return {"ok": True}
