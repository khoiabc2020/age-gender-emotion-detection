"""
Analytics API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional

from app.core.database import get_db
from app.db import crud
from app.schemas.analytics import InteractionCreate, InteractionResponse, AnalyticsStats, TimeRangeQuery

router = APIRouter()


@router.post("/interactions", response_model=InteractionResponse)
async def create_interaction(
    interaction: InteractionCreate,
    db: Session = Depends(get_db)
):
    """
    Create new interaction record from edge device
    This endpoint is called by MQTT worker or directly from edge device
    """
    # Get or create device
    device = crud.get_device_by_key(db, interaction.device_key)
    if not device:
        device = crud.create_device(db, interaction.device_key)
    
    # Get or create active session
    session = crud.get_active_session(db, device.id, interaction.track_id)
    if not session:
        session = crud.create_session(db, device.id, interaction.track_id)
    
    # Create interaction
    db_interaction = crud.create_interaction(
        db=db,
        session_id=session.id,
        age=interaction.age,
        gender=interaction.gender,
        emotion=interaction.emotion,
        ad_id=interaction.ad_id,
        ad_name=interaction.ad_name
    )
    
    return db_interaction


@router.get("/stats", response_model=AnalyticsStats)
async def get_analytics_stats(
    hours: int = 24,
    device_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Get analytics statistics for the last N hours
    """
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)
    
    interactions = crud.get_interactions_by_time_range(
        db, start_time, end_time, device_id
    )
    
    if not interactions:
        return AnalyticsStats(
            total_interactions=0,
            unique_customers=0,
            avg_age=0.0,
            gender_distribution={},
            emotion_distribution={},
            top_ads=[]
        )
    
    # Calculate statistics
    total_interactions = len(interactions)
    unique_customers = len(set(i.session_id for i in interactions))
    avg_age = sum(i.age for i in interactions) / total_interactions
    
    # Gender distribution
    gender_dist = {}
    for i in interactions:
        gender_dist[i.gender] = gender_dist.get(i.gender, 0) + 1
    
    # Emotion distribution
    emotion_dist = {}
    for i in interactions:
        emotion_dist[i.emotion] = emotion_dist.get(i.emotion, 0) + 1
    
    # Top ads
    ad_counts = {}
    for i in interactions:
        ad_counts[i.ad_id] = ad_counts.get(i.ad_id, 0) + 1
    top_ads = [
        {"ad_id": ad_id, "count": count}
        for ad_id, count in sorted(ad_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    ]
    
    return AnalyticsStats(
        total_interactions=total_interactions,
        unique_customers=unique_customers,
        avg_age=round(avg_age, 2),
        gender_distribution=gender_dist,
        emotion_distribution=emotion_dist,
        top_ads=top_ads
    )


@router.get("/age-by-hour")
async def get_age_by_hour(
    hours: int = 24,
    device_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get average age grouped by hour"""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)
    
    results = crud.get_average_age_by_hour(db, start_time, end_time, device_id)
    return results


@router.get("/emotion-distribution")
async def get_emotion_distribution(
    hours: int = 24,
    device_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get emotion distribution"""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)
    
    results = crud.get_emotion_distribution(db, start_time, end_time, device_id)
    return results


@router.get("/ad-performance")
async def get_ad_performance(
    hours: int = 24,
    device_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get advertisement performance metrics"""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)
    
    results = crud.get_ad_performance(db, start_time, end_time, device_id)
    return results

