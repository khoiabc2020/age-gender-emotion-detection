"""
Dashboard API endpoints (protected)
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional

from app.core.database import get_db
from app.db import crud, models

router = APIRouter()

@router.get("/devices")
async def get_devices(db: Session = Depends(get_db)):
    """Get all devices"""
    devices = db.query(models.Device).all()
    return devices

@router.get("/sessions")
async def get_sessions(
    limit: int = 100,
    device_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get recent sessions"""
    query = db.query(models.Session).order_by(models.Session.start_time.desc())
    
    if device_id:
        query = query.filter(models.Session.device_id == device_id)
    
    sessions = query.limit(limit).all()
    return sessions

