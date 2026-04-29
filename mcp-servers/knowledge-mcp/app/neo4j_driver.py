import os
from contextlib import asynccontextmanager
from neo4j import AsyncGraphDatabase, AsyncDriver

_driver: AsyncDriver | None = None

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")


def get_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


async def close_driver() -> None:
    global _driver
    if _driver:
        await _driver.close()
        _driver = None


async def ensure_indexes() -> None:
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            "CREATE FULLTEXT INDEX entity_search IF NOT EXISTS "
            "FOR (n:NPC|Location|Faction|Item) ON EACH [n.name, n.description]"
        )
        await result.consume()
        result = await session.run("CREATE INDEX npc_campaign IF NOT EXISTS FOR (n:NPC) ON (n.campaign_id, n.name)")
        await result.consume()
        result = await session.run("CREATE INDEX location_campaign IF NOT EXISTS FOR (n:Location) ON (n.campaign_id, n.name)")
        await result.consume()
        result = await session.run("CREATE INDEX faction_campaign IF NOT EXISTS FOR (n:Faction) ON (n.campaign_id, n.name)")
        await result.consume()
