"""
Pydantic schemas for analytics endpoints
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

class InteractionCreate(BaseModel):
    """Schema for creating interaction"""
    device_key: str
    track_id: int
    age: int = Field(..., ge=0, le=100)
    gender: str = Field(..., pattern="^(male|female)$")
    emotion: str
    ad_id: str
    ad_name: Optional[str] = None

class InteractionResponse(BaseModel):
    """Schema for interaction response"""
    id: int
    timestamp: datetime
    age: int
    gender: str
    emotion: str
    ad_id: str
    ad_name: Optional[str] = None
    
    class Config:
        from_attributes = True

class AnalyticsStats(BaseModel):
    """Schema for analytics statistics"""
    total_interactions: int
    unique_customers: int
    avg_age: float
    gender_distribution: dict
    emotion_distribution: dict
    top_ads: List[dict]

class TimeRangeQuery(BaseModel):
    """Schema for time range queries"""
    start_time: datetime
    end_time: datetime
    device_id: Optional[int] = None

