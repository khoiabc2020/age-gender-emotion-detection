"""
Advertisement management API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.db import crud
from app.db.models import Advertisement

router = APIRouter()


@router.get("/", response_model=List[dict])
async def get_all_ads(db: Session = Depends(get_db)):
    """Get all advertisements"""
    ads = crud.get_all_advertisements(db)
    return [
        {
            "id": ad.id,
            "ad_id": ad.ad_id,
            "name": ad.name,
            "description": ad.description,
            "target_age_min": ad.target_age_min,
            "target_age_max": ad.target_age_max,
            "target_gender": ad.target_gender,
            "priority": ad.priority
        }
        for ad in ads
    ]


@router.get("/{ad_id}")
async def get_ad(ad_id: str, db: Session = Depends(get_db)):
    """Get advertisement by ID"""
    ad = crud.get_advertisement(db, ad_id)
    if not ad:
        raise HTTPException(status_code=404, detail="Advertisement not found")
    return ad

