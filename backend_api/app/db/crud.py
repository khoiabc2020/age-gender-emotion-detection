"""
CRUD operations for database models
"""

from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from app.db.models import Device, Session as DBSession, Interaction, Advertisement


# Device CRUD
def get_device_by_key(db: Session, device_key: str) -> Optional[Device]:
    """Get device by device_key"""
    return db.query(Device).filter(Device.device_key == device_key).first()


def create_device(db: Session, device_key: str, name: str = None, location: str = None) -> Device:
    """Create new device"""
    device = Device(device_key=device_key, name=name, location=location)
    db.add(device)
    db.commit()
    db.refresh(device)
    return device


# Session CRUD
def create_session(db: Session, device_id: int, track_id: int) -> DBSession:
    """Create new customer session"""
    session = DBSession(device_id=device_id, track_id=track_id)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_active_session(db: Session, device_id: int, track_id: int) -> Optional[DBSession]:
    """Get active session for track_id"""
    return db.query(DBSession).filter(
        and_(
            DBSession.device_id == device_id,
            DBSession.track_id == track_id,
            DBSession.end_time.is_(None)
        )
    ).first()


def end_session(db: Session, session_id: int):
    """End a session"""
    session = db.query(DBSession).filter(DBSession.id == session_id).first()
    if session:
        session.end_time = datetime.utcnow()
        db.commit()


# Interaction CRUD
def create_interaction(
    db: Session,
    session_id: int,
    age: int,
    gender: str,
    emotion: str,
    ad_id: str,
    ad_name: str = None
) -> Interaction:
    """Create new interaction record"""
    interaction = Interaction(
        session_id=session_id,
        age=age,
        gender=gender,
        emotion=emotion,
        ad_id=ad_id,
        ad_name=ad_name
    )
    db.add(interaction)
    db.commit()
    db.refresh(interaction)
    return interaction


# Analytics queries
def get_interactions_by_time_range(
    db: Session,
    start_time: datetime,
    end_time: datetime,
    device_id: Optional[int] = None
) -> List[Interaction]:
    """Get interactions within time range"""
    query = db.query(Interaction).filter(
        and_(
            Interaction.timestamp >= start_time,
            Interaction.timestamp <= end_time
        )
    )
    
    if device_id:
        query = query.join(DBSession).filter(DBSession.device_id == device_id)
    
    return query.all()


def get_average_age_by_hour(
    db: Session,
    start_date: datetime,
    end_date: datetime,
    device_id: Optional[int] = None
) -> List[Dict]:
    """Get average age grouped by hour"""
    query = db.query(
        func.extract('hour', Interaction.timestamp).label('hour'),
        func.avg(Interaction.age).label('avg_age'),
        func.count(Interaction.id).label('count')
    ).filter(
        and_(
            Interaction.timestamp >= start_date,
            Interaction.timestamp <= end_date
        )
    )
    
    if device_id:
        query = query.join(DBSession).filter(DBSession.device_id == device_id)
    
    results = query.group_by(func.extract('hour', Interaction.timestamp)).all()
    
    return [
        {
            'hour': int(row.hour),
            'avg_age': float(row.avg_age),
            'count': int(row.count)
        }
        for row in results
    ]


def get_emotion_distribution(
    db: Session,
    start_date: datetime,
    end_date: datetime,
    device_id: Optional[int] = None
) -> List[Dict]:
    """Get emotion distribution"""
    query = db.query(
        Interaction.emotion,
        func.count(Interaction.id).label('count')
    ).filter(
        and_(
            Interaction.timestamp >= start_date,
            Interaction.timestamp <= end_date
        )
    )
    
    if device_id:
        query = query.join(DBSession).filter(DBSession.device_id == device_id)
    
    results = query.group_by(Interaction.emotion).all()
    
    return [
        {'emotion': row.emotion, 'count': int(row.count)}
        for row in results
    ]


def get_ad_performance(
    db: Session,
    start_date: datetime,
    end_date: datetime,
    device_id: Optional[int] = None
) -> List[Dict]:
    """Get advertisement performance metrics"""
    query = db.query(
        Interaction.ad_id,
        func.count(Interaction.id).label('display_count'),
        func.avg(Interaction.age).label('avg_age')
    ).filter(
        and_(
            Interaction.timestamp >= start_date,
            Interaction.timestamp <= end_date
        )
    )
    
    if device_id:
        query = query.join(DBSession).filter(DBSession.device_id == device_id)
    
    results = query.group_by(Interaction.ad_id).all()
    
    return [
        {
            'ad_id': row.ad_id,
            'display_count': int(row.display_count),
            'avg_age': float(row.avg_age) if row.avg_age else None
        }
        for row in results
    ]


# Advertisement CRUD
def get_advertisement(db: Session, ad_id: str) -> Optional[Advertisement]:
    """Get advertisement by ad_id"""
    return db.query(Advertisement).filter(Advertisement.ad_id == ad_id).first()


def get_all_advertisements(db: Session) -> List[Advertisement]:
    """Get all advertisements"""
    return db.query(Advertisement).all()

