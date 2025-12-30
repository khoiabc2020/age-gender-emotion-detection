"""
AI Agent API endpoints
Giai đoạn 6: Generative AI Integration
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional
from pydantic import BaseModel

from app.core.database import get_db
from app.core.config import settings
from app.services.ai_agent import AIAgent, AIProvider
from app.db import crud
from app.main import get_current_user

router = APIRouter()


class ChatRequest(BaseModel):
    """Chat request schema"""
    question: str
    time_range_hours: Optional[int] = 24


class GenerateReportRequest(BaseModel):
    """Generate report request schema"""
    time_range_hours: Optional[int] = 24
    include_charts: Optional[bool] = True


# Initialize AI Agent (singleton)
_ai_agent: Optional[AIAgent] = None


def get_ai_agent() -> Optional[AIAgent]:
    """Get or create AI Agent instance"""
    global _ai_agent
    
    if _ai_agent is None:
        google_key = getattr(settings, 'GOOGLE_AI_API_KEY', None)
        openai_key = getattr(settings, 'OPENAI_API_KEY', None)
        provider = getattr(settings, 'AI_PROVIDER', 'google_ai')
        
        if provider == 'both':
            provider_enum = AIProvider.BOTH
        elif provider == 'chatgpt':
            provider_enum = AIProvider.CHATGPT
        else:
            provider_enum = AIProvider.GOOGLE_AI
        
        if google_key or openai_key:
            _ai_agent = AIAgent(
                google_ai_api_key=google_key,
                openai_api_key=openai_key,
                provider=provider_enum
            )
    
    return _ai_agent


@router.post("/analyze")
async def analyze_analytics(
    time_range_hours: int = 24,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyze analytics data using AI
    """
    agent = get_ai_agent()
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI Agent not configured. Please set API keys in settings."
        )
    
    # Fetch analytics data
    from datetime import datetime, timedelta
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=time_range_hours)
    
    interactions = crud.get_interactions_by_time_range(db, start_time, end_time)
    
    if not interactions:
        return {
            "message": "No data available for analysis",
            "insights": [],
            "recommendations": []
        }
    
    # Calculate stats
    total_interactions = len(interactions)
    unique_customers = len(set(i.session_id for i in interactions))
    avg_age = sum(i.age for i in interactions) / total_interactions if total_interactions > 0 else 0
    
    gender_dist = {}
    for i in interactions:
        gender_dist[i.gender] = gender_dist.get(i.gender, 0) + 1
    
    emotion_dist = {}
    for i in interactions:
        emotion_dist[i.emotion] = emotion_dist.get(i.emotion, 0) + 1
    
    ad_counts = {}
    for i in interactions:
        ad_counts[i.ad_id] = ad_counts.get(i.ad_id, 0) + 1
    top_ads = [
        {"ad_id": ad_id, "count": count}
        for ad_id, count in sorted(ad_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    ]
    
    stats = {
        "total_interactions": total_interactions,
        "unique_customers": unique_customers,
        "avg_age": round(avg_age, 2),
        "gender_distribution": gender_dist,
        "emotion_distribution": emotion_dist,
        "top_ads": top_ads
    }
    
    age_by_hour = crud.get_average_age_by_hour(db, start_time, end_time)
    emotion_distribution = crud.get_emotion_distribution(db, start_time, end_time)
    ad_performance = crud.get_ad_performance(db, start_time, end_time)
    
    # Analyze with AI
    analysis = agent.analyze_analytics(
        stats=stats,
        age_by_hour=age_by_hour,
        emotion_distribution=emotion_distribution,
        gender_distribution=gender_dist,
        ad_performance=ad_performance
    )
    
    return analysis


@router.post("/chat")
async def chat_with_data(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Chat with analytics data using AI
    """
    agent = get_ai_agent()
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI Agent not configured. Please set API keys in settings."
        )
    
    # Fetch context data
    from datetime import datetime, timedelta
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=request.time_range_hours)
    
    interactions = crud.get_interactions_by_time_range(db, start_time, end_time)
    
    context_data = {
        "total_interactions": len(interactions),
        "time_range_hours": request.time_range_hours
    }
    
    if interactions:
        context_data["avg_age"] = sum(i.age for i in interactions) / len(interactions)
        context_data["gender_distribution"] = {}
        context_data["emotion_distribution"] = {}
        
        for i in interactions:
            context_data["gender_distribution"][i.gender] = \
                context_data["gender_distribution"].get(i.gender, 0) + 1
            context_data["emotion_distribution"][i.emotion] = \
                context_data["emotion_distribution"].get(i.emotion, 0) + 1
    
    # Get AI response
    response = agent.chat_with_data(request.question, context_data)
    
    return {
        "question": request.question,
        "answer": response,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/generate-report")
async def generate_report(
    request: GenerateReportRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate automated analytics report
    """
    agent = get_ai_agent()
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI Agent not configured. Please set API keys in settings."
        )
    
    # Fetch stats
    from datetime import datetime, timedelta
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=request.time_range_hours)
    
    interactions = crud.get_interactions_by_time_range(db, start_time, end_time)
    
    if not interactions:
        return {
            "report": "No data available for the selected time range.",
            "time_range": f"{request.time_range_hours} hours"
        }
    
    total_interactions = len(interactions)
    unique_customers = len(set(i.session_id for i in interactions))
    avg_age = sum(i.age for i in interactions) / total_interactions
    
    gender_dist = {}
    for i in interactions:
        gender_dist[i.gender] = gender_dist.get(i.gender, 0) + 1
    
    emotion_dist = {}
    for i in interactions:
        emotion_dist[i.emotion] = emotion_dist.get(i.emotion, 0) + 1
    
    stats = {
        "total_interactions": total_interactions,
        "unique_customers": unique_customers,
        "avg_age": round(avg_age, 2),
        "gender_distribution": gender_dist,
        "emotion_distribution": emotion_dist
    }
    
    # Generate report
    report = agent.generate_report(stats, f"{request.time_range_hours} hours")
    
    return {
        "report": report,
        "time_range": f"{request.time_range_hours} hours",
        "generated_at": datetime.utcnow().isoformat()
    }


@router.get("/status")
async def get_ai_agent_status(
    current_user: dict = Depends(get_current_user)
):
    """
    Get AI Agent configuration status
    """
    agent = get_ai_agent()
    
    google_configured = bool(getattr(settings, 'GOOGLE_AI_API_KEY', None))
    openai_configured = bool(getattr(settings, 'OPENAI_API_KEY', None))
    provider = getattr(settings, 'AI_PROVIDER', 'google_ai')
    
    return {
        "available": agent is not None,
        "google_ai_configured": google_configured,
        "openai_configured": openai_configured,
        "provider": provider,
        "status": "ready" if agent else "not_configured"
    }

