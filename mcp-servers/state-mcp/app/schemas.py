from typing import Any, Optional
from pydantic import BaseModel


# create_campaign
class CreateCampaignIn(BaseModel):
    title: Optional[str] = 'Untitled Campaign'
    language: str = 'en'


class CreateCampaignOut(BaseModel):
    campaign_id: str


# save_campaign_plan
class SaveCampaignPlanIn(BaseModel):
    plan_json: dict[str, Any]
    visual_style: Optional[str] = None


# save_character
class SaveCharacterIn(BaseModel):
    character_json: dict[str, Any]
    portrait_path: Optional[str] = None


class SaveCharacterOut(BaseModel):
    character_id: str


# set_phase
class SetPhaseIn(BaseModel):
    phase: str  # character_creation | campaign_design | active | completed


# set_active_npc
class SetActiveNPCIn(BaseModel):
    npc_id: str
    briefing: dict[str, Any]
    conv_start_turn_id: str


# get_active_npc_state
class ActiveNPCStateOut(BaseModel):
    npc_id: Optional[str]
    briefing: Optional[dict[str, Any]] = None
    conv_start_turn_id: Optional[str] = None


# get_routing_state
class RoutingStateOut(BaseModel):
    phase: str
    active_npc_id: Optional[str]


# get_campaign_context
class CampaignContextOut(BaseModel):
    campaign: dict[str, Any]
    character: Optional[dict[str, Any]]


# save_npc
class SaveNPCIn(BaseModel):
    npc_json: dict[str, Any]
    portrait_path: Optional[str] = None


class SaveNPCOut(BaseModel):
    npc_id: str


# get_npc
class GetNPCIn(BaseModel):
    npc_id: Optional[str] = None
    name: Optional[str] = None


class GetNPCOut(BaseModel):
    npc: Optional[dict[str, Any]]


# list_npcs
class ListNPCsOut(BaseModel):
    npcs: list[dict[str, Any]]


# log_turn
class LogTurnIn(BaseModel):
    role: str  # player | dm | npc | system
    content: str
    npc_name: Optional[str] = None
    audio_path: Optional[str] = None
    image_path: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class LogTurnOut(BaseModel):
    turn_id: str


# get_turns
class GetTurnsIn(BaseModel):
    limit: Optional[int] = 20
    exclude_roles: Optional[list[str]] = None
    since_turn_id: Optional[str] = None
    before_turn_id: Optional[str] = None
    npc_name: Optional[str] = None
    phase_filter: Optional[str] = None  # e.g. 'active' to exclude setup turns


class GetTurnsOut(BaseModel):
    turns: list[dict[str, Any]]


# get_memory / update_memory
class GetMemoryOut(BaseModel):
    short_term: list[str]
    long_term: str


class UpdateMemoryIn(BaseModel):
    short_term: list[str]
    long_term: str
