from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional

from ..neo4j_driver import get_driver
from ..extractor import extract_entities

router = APIRouter()

MAX_NODES = 50


class UpdateWorldIn(BaseModel):
    narrative_text: str


class UpdateWorldOut(BaseModel):
    entities_added: int
    relationships_added: int


class GetWorldContextIn(BaseModel):
    focus_text: Optional[str] = None


class GetWorldContextOut(BaseModel):
    context: str


@router.post("/tools/update_world", response_model=UpdateWorldOut)
async def update_world(body: UpdateWorldIn, request: Request):
    campaign_id = request.state.campaign_id
    extracted = await extract_entities(body.narrative_text)
    nodes = extracted.get("nodes", [])
    relationships = extracted.get("relationships", [])

    driver = get_driver()
    entities_added = 0
    rels_added = 0

    async with driver.session() as session:
        for node in nodes:
            label = node.get("label", "NPC")
            name = node.get("name", "")
            if not name:
                continue
            props = node.get("properties", {})
            props["campaign_id"] = campaign_id
            props["name"] = name
            cypher = f"MERGE (n:{label} {{campaign_id: $campaign_id, name: $name}}) SET n += $props"
            await session.run(cypher, campaign_id=campaign_id, name=name, props=props)
            entities_added += 1

        for rel in relationships:
            from_label = rel.get("from_label", "NPC")
            from_name = rel.get("from_name", "")
            to_label = rel.get("to_label", "NPC")
            to_name = rel.get("to_name", "")
            rel_type = rel.get("type", "INVOLVES")
            rel_props = rel.get("properties", {})
            if not from_name or not to_name:
                continue
            cypher = (
                f"MATCH (a:{from_label} {{campaign_id: $cid, name: $from_name}}) "
                f"MATCH (b:{to_label} {{campaign_id: $cid, name: $to_name}}) "
                f"MERGE (a)-[r:{rel_type}]->(b) SET r += $props"
            )
            await session.run(cypher, cid=campaign_id, from_name=from_name, to_name=to_name, props=rel_props)
            rels_added += 1

    return UpdateWorldOut(entities_added=entities_added, relationships_added=rels_added)


@router.post("/tools/get_world_context", response_model=GetWorldContextOut)
async def get_world_context(body: GetWorldContextIn, request: Request):
    campaign_id = request.state.campaign_id
    driver = get_driver()
    lines: list[str] = []

    async with driver.session() as session:
        if body.focus_text:
            result = await session.run(
                "CALL db.index.fulltext.queryNodes('entity_search', $query) "
                "YIELD node, score "
                "WHERE node.campaign_id = $campaign_id "
                "RETURN node, score ORDER BY score DESC LIMIT 10",
                query=body.focus_text,
                campaign_id=campaign_id,
            )
            focus_nodes = [r["node"] for r in await result.data()]
        else:
            result = await session.run(
                "MATCH (n) WHERE n.campaign_id = $campaign_id RETURN n LIMIT $limit",
                campaign_id=campaign_id,
                limit=MAX_NODES,
            )
            focus_nodes = [r["n"] for r in await result.data()]

        entity_lines: list[str] = []
        rel_lines: list[str] = []

        for node in focus_nodes:
            label = list(node.labels)[0] if node.labels else "Entity"
            name = node.get("name", "?")
            entity_lines.append(f"- [{label}] {name}")

            # 1-hop relationships
            rel_result = await session.run(
                "MATCH (n {campaign_id: $cid, name: $name})-[r]->(m) "
                "RETURN type(r) as rel_type, m.name as target, properties(r) as props LIMIT 5",
                cid=campaign_id,
                name=name,
            )
            for rr in await rel_result.data():
                props_str = ", ".join(f"{k}: {v}" for k, v in (rr["props"] or {}).items())
                suffix = f" ({props_str})" if props_str else ""
                rel_lines.append(f"- {name} {rr['rel_type']} {rr['target']}{suffix}")

    if not entity_lines:
        return GetWorldContextOut(context="## World Context\nNo world data available yet.")

    context = "## World Context\n### Entities\n"
    context += "\n".join(entity_lines)
    if rel_lines:
        context += "\n### Relationships\n" + "\n".join(rel_lines)
    return GetWorldContextOut(context=context)
