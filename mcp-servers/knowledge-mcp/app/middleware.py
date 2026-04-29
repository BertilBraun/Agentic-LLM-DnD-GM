from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class CampaignIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.campaign_id = request.headers.get("X-Campaign-ID")
        return await call_next(request)
