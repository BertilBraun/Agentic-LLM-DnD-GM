"""Routes player messages to the correct A2A agent based on campaign phase."""
import json
import httpx

from .config import settings
from .a2a_client import send_task
from .sse.publisher import publish_event

OPENING_SCENE_SENTINEL = "__opening_scene__"


async def _get_routing_state(campaign_id: str) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{settings.state_mcp_url}/tools/get_routing_state",
            json={},
            headers={"X-Campaign-ID": campaign_id},
        )
        resp.raise_for_status()
        return resp.json()


async def _get_campaign_context(campaign_id: str) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{settings.state_mcp_url}/tools/get_campaign_context",
            json={},
            headers={"X-Campaign-ID": campaign_id},
        )
        resp.raise_for_status()
        return resp.json()


def _opening_prompt(ctx: dict) -> str:
    character = ctx.get("character") or {}
    name = character.get("name", "the adventurer")
    return (
        f"Describe the opening scene for {name}. Set up the initial situation that draws "
        "them into the adventure, taking into account their background and the campaign synopsis. "
        "Do not introduce an NPC in the opening scene."
    )


async def dispatch(campaign_id: str, message: str) -> None:
    """Dispatch message to correct agent. Called as asyncio.create_task from the game router."""
    routing = await _get_routing_state(campaign_id)
    phase = routing.get("phase", "character_creation")
    active_npc_id = routing.get("active_npc_id")

    # Rewrite sentinel
    if message == OPENING_SCENE_SENTINEL:
        ctx = await _get_campaign_context(campaign_id)
        message = _opening_prompt(ctx)

    if phase == "character_creation":
        await send_task(settings.character_creator_url, campaign_id, message)
    elif phase == "campaign_design":
        await send_task(settings.campaign_designer_url, campaign_id, message)
    elif phase == "active":
        if active_npc_id:
            await send_task(settings.npc_agent_url, campaign_id, message)
        else:
            await send_task(settings.dm_agent_url, campaign_id, message)
    elif phase == "completed":
        await publish_event(campaign_id, {"type": "campaign_completed"})
