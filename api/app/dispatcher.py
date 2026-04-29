"""Routes player messages to the correct A2A agent based on campaign phase."""
from __future__ import annotations

import httpx
from pydantic import BaseModel

from .config import settings
from .a2a_client import send_task
from .sse.publisher import publish_event

OPENING_SCENE_SENTINEL = '__opening_scene__'


class RoutingState(BaseModel):
    phase: str = "character_creation"
    active_npc_id: str | None = None


class CharacterContext(BaseModel):
    name: str = "the adventurer"


class CampaignContext(BaseModel):
    character: CharacterContext | None = None


async def _get_routing_state(campaign_id: str) -> RoutingState:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f'{settings.state_mcp_url}/tools/get_routing_state',
            json={},
            headers={'X-Campaign-ID': campaign_id},
        )
        resp.raise_for_status()
        return RoutingState.model_validate(resp.json())


async def _get_campaign_context(campaign_id: str) -> CampaignContext:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f'{settings.state_mcp_url}/tools/get_campaign_context',
            json={},
            headers={'X-Campaign-ID': campaign_id},
        )
        resp.raise_for_status()
        return CampaignContext.model_validate(resp.json())


def _opening_prompt(ctx: CampaignContext) -> str:
    name = ctx.character.name if ctx.character else "the adventurer"
    return (
        f'Describe the opening scene for {name}. Set up the initial situation that draws '
        'them into the adventure, taking into account their background and the campaign synopsis. '
        'Do not introduce an NPC in the opening scene.'
    )


async def dispatch(campaign_id: str, message: str) -> None:
    """Dispatch message to correct agent. Called as asyncio.create_task from the game router."""
    routing = await _get_routing_state(campaign_id)

    if message == OPENING_SCENE_SENTINEL:
        ctx = await _get_campaign_context(campaign_id)
        message = _opening_prompt(ctx)

    if routing.phase == 'character_creation':
        await send_task(settings.character_creator_url, campaign_id, message)
    elif routing.phase == 'campaign_design':
        await send_task(settings.campaign_designer_url, campaign_id, message)
    elif routing.phase == 'active':
        if routing.active_npc_id:
            result = await send_task(settings.npc_agent_url, campaign_id, message)
            routing_after = await _get_routing_state(campaign_id)
            if not routing_after.active_npc_id:
                summary = result.result.output
                assert summary, 'NPC agent must return a summary of the conversation when it ends'
                dm_message = f'(The conversation just ended. {summary})'
                await send_task(settings.dm_agent_url, campaign_id, dm_message)
        else:
            await send_task(settings.dm_agent_url, campaign_id, message)
    elif routing.phase == 'completed':
        await publish_event(campaign_id, {'type': 'campaign_completed'})
