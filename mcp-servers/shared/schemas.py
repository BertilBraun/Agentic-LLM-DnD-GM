"""Pydantic models shared across all MCP servers."""

from pydantic import BaseModel


class OkOut(BaseModel):
    ok: bool = True
