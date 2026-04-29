from typing import Optional, Any
from pydantic import BaseModel


class CreateCampaignRequest(BaseModel):
    title: Optional[str] = 'Untitled Campaign'
    language: str = 'en'


class CampaignOut(BaseModel):
    id: str
    title: str
    phase: str
    created_at: str


class CampaignDetailOut(BaseModel):
    id: str
    title: str
    language: str
    phase: str
    plan_json: Optional[dict[str, Any]] = None
    visual_style: Optional[str] = None
    created_at: str
    updated_at: str
