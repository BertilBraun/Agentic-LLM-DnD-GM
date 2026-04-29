"""Thin async HTTP client for MCP server tool calls.
Injects X-Campaign-ID on every request.
"""
from __future__ import annotations

import httpx


class MCPClient:
    def __init__(self, base_url: str, campaign_id: str, timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._campaign_id = campaign_id
        self._timeout = timeout

    async def call(self, tool_name: str, body: dict | None = None) -> dict:
        url = f"{self._base_url}/tools/{tool_name}"
        headers = {"X-Campaign-ID": self._campaign_id}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=body or {}, headers=headers)
            resp.raise_for_status()
            return resp.json()
