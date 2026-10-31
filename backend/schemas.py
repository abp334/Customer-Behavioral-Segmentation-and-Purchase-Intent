"""Pydantic request / response models for the NEXUS API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SimpleProfileRequest(BaseModel):
    """Marketer-friendly profile knobs (mapped server-side to 17 features)."""

    recency_days: float = Field(3, ge=0, le=30)
    total_spend: float = Field(50, ge=0)
    cart_count: float = Field(1, ge=0)
    total_sessions: float = Field(5, ge=0)
    total_events: float = Field(20, ge=0)


class CampaignRequest(BaseModel):
    goal: str = Field(
        "maximize_profit",
        description="maximize_profit | win_back_lapsed | convert_cart_abandoners | reward_loyal_champions",
    )
    total_budget: float = Field(5000, gt=0)
    cost_per_contact: float = Field(1.0, gt=0)


class DemoGuessRequest(BaseModel):
    case_id: int
    guessed_persona_cluster: int = Field(..., ge=0, le=3)
    guessed_will_buy: bool
