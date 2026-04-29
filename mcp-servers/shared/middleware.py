from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class CampaignIDMiddleware(BaseHTTPMiddleware):
    """Reads X-Campaign-ID header and stores it in request.state.campaign_id.
    All MCP tool handlers use request.state.campaign_id — never a body parameter.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request.state.campaign_id = request.headers.get("X-Campaign-ID")
        return await call_next(request)
