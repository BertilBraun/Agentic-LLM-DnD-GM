from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class CampaignIDMiddleware(BaseHTTPMiddleware):
    """Reads X-Campaign-ID header and stores it in request.state.campaign_id.
    All MCP tool handlers use request.state.campaign_id — never a body parameter.
    """

    async def dispatch(self, request: Request, call_next):
        request.state.campaign_id = request.headers.get("X-Campaign-ID")
        return await call_next(request)
