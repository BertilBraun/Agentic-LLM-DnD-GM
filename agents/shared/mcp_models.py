"""Pydantic response models for all MCP tool calls used by agents."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# ─── State MCP ────────────────────────────────────────────────────


class Turn(BaseModel):
    id: str = ''
    campaign_id: str = ''
    role: str = ''
    content: str = ''
    npc_name: str | None = None
    audio_path: str | None = None
    image_path: str | None = None
    metadata: dict = Field(default_factory=dict)
    created_at: str = ''


class LogTurnOut(BaseModel):
    turn_id: str = ''


class GetTurnsOut(BaseModel):
    turns: list[Turn] = Field(default_factory=list)


class CampaignData(BaseModel):
    model_config = ConfigDict(extra='allow')
    visual_style: str = ''


class CharacterData(BaseModel):
    model_config = ConfigDict(extra='allow')


class CampaignContextOut(BaseModel):
    campaign: CampaignData = Field(default_factory=CampaignData)
    character: CharacterData | None = None


class NpcBriefing(BaseModel):
    goals: str = ''
    knows: str = ''
    mood: str = ''
    reveal_if: str = ''


class NpcData(BaseModel):
    model_config = ConfigDict(extra='allow')
    name: str = ''
    role: str = ''
    voice_id: str = 'ash'
    voice_instructions: str = ''
    visual_description: str = ''
    opening_line: str = ''
    portrait_path: str | None = None


class SaveNpcOut(BaseModel):
    npc_id: str = ''


class GetNpcOut(BaseModel):
    npc: NpcData | None = None


class ActiveNpcStateOut(BaseModel):
    npc_id: str | None = None
    briefing: NpcBriefing | None = None
    conv_start_turn_id: str | None = None


class GetMemoryOut(BaseModel):
    short_term: list[str] = Field(default_factory=list)
    long_term: str = ''


# ─── Memory MCP ───────────────────────────────────────────────────


class RecallOut(BaseModel):
    context: str = ''


# ─── Knowledge MCP ────────────────────────────────────────────────


class WorldContextOut(BaseModel):
    context: str = ''


class UpdateWorldOut(BaseModel):
    entities_added: int = 0
    relationships_added: int = 0


# ─── Media MCP ────────────────────────────────────────────────────


class SpeakOut(BaseModel):
    stream_path: str = ''


class ImageOut(BaseModel):
    file_path: str = ''


# ─── Generic ──────────────────────────────────────────────────────


class OkOut(BaseModel):
    ok: bool = True
