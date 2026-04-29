"""Shared async DB helpers for MCP servers that need PostgreSQL access."""
import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession


def make_session_factory(database_url: str | None = None):
    url = database_url or os.environ["DATABASE_URL"]
    engine = create_async_engine(url, pool_pre_ping=True)
    return async_sessionmaker(engine, expire_on_commit=False)
