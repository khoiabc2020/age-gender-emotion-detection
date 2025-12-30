"""
SQLAlchemy database models
Optimized for time-series data using PostgreSQL
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Device(Base):
    """Edge device information"""
    __tablename__ = "devices"
    
    id = Column(Integer, primary_key=True, index=True)
    device_key = Column(String(100), unique=True, index=True, nullable=False)
    name = Column(String(200))
    location = Column(String(200))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    sessions = relationship("Session", back_populates="device")


class Session(Base):
    """Customer session tracking"""
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False)
    track_id = Column(Integer, nullable=False)  # Tracking ID from edge device
    start_time = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    device = relationship("Device", back_populates="sessions")
    interactions = relationship("Interaction", back_populates="session")


class Interaction(Base):
    """Individual customer interaction/analytics data"""
    __tablename__ = "interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Customer attributes
    age = Column(Integer, nullable=False)
    gender = Column(String(20), nullable=False)
    emotion = Column(String(50), nullable=False)
    
    # Advertisement
    ad_id = Column(String(100), nullable=False)
    ad_name = Column(String(200))
    
    # Relationships
    session = relationship("Session", back_populates="interactions")
    
    # Index for time-series queries
    __table_args__ = (
        Index('idx_interactions_timestamp', 'timestamp'),
        Index('idx_interactions_session_timestamp', 'session_id', 'timestamp'),
    )


class Advertisement(Base):
    """Advertisement metadata"""
    __tablename__ = "advertisements"
    
    id = Column(Integer, primary_key=True, index=True)
    ad_id = Column(String(100), unique=True, index=True, nullable=False)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    target_age_min = Column(Integer)
    target_age_max = Column(Integer)
    target_gender = Column(String(20))
    target_emotions = Column(String(200))  # JSON array as string
    priority = Column(Integer, default=5)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

