"""
Database Models - SQLite + SQLAlchemy
Giai đoạn 3 Tuần 9: Local Database & Reporting
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pathlib import Path

Base = declarative_base()


class CustomerInteraction(Base):
    """Customer interaction record"""
    __tablename__ = 'customer_interactions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.now, nullable=False)
    
    # Demographics
    age = Column(Float, nullable=True)
    gender = Column(String(10), nullable=True)  # 'male' or 'female'
    emotion = Column(String(20), nullable=True)  # 'happy', 'sad', etc.
    
    # Interaction data
    dwell_time = Column(Float, nullable=True)  # Seconds
    ad_shown = Column(String(100), nullable=True)  # Ad ID/name
    ad_clicked = Column(Integer, default=0)  # 0 or 1
    
    # Location
    bbox_x = Column(Float, nullable=True)
    bbox_y = Column(Float, nullable=True)
    bbox_w = Column(Float, nullable=True)
    bbox_h = Column(Float, nullable=True)
    
    # Additional metadata
    metadata = Column(Text, nullable=True)  # JSON string


class Session(Base):
    """Session record"""
    __tablename__ = 'sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    start_time = Column(DateTime, default=datetime.now, nullable=False)
    end_time = Column(DateTime, nullable=True)
    total_customers = Column(Integer, default=0)
    total_interactions = Column(Integer, default=0)


class DatabaseManager:
    """Database manager for local SQLite database"""
    
    def __init__(self, db_path: str = "data/edge_app.db"):
        """
        Initialize database manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create engine
        self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def add_interaction(self, **kwargs):
        """Add customer interaction"""
        session = self.get_session()
        try:
            interaction = CustomerInteraction(**kwargs)
            session.add(interaction)
            session.commit()
            return interaction.id
        except Exception as e:
            session.rollback()
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error adding interaction: {e}", exc_info=True)
            return None
        finally:
            session.close()
    
    def get_interactions(self, start_time=None, end_time=None, limit=1000):
        """Get interactions within time range"""
        session = self.get_session()
        try:
            query = session.query(CustomerInteraction)
            
            if start_time:
                query = query.filter(CustomerInteraction.timestamp >= start_time)
            if end_time:
                query = query.filter(CustomerInteraction.timestamp <= end_time)
            
            return query.order_by(CustomerInteraction.timestamp.desc()).limit(limit).all()
        finally:
            session.close()
    
    def get_statistics(self, start_time=None, end_time=None):
        """Get statistics"""
        session = self.get_session()
        try:
            query = session.query(CustomerInteraction)
            
            if start_time:
                query = query.filter(CustomerInteraction.timestamp >= start_time)
            if end_time:
                query = query.filter(CustomerInteraction.timestamp <= end_time)
            
            total = query.count()
            
            # Get unique tracks (more efficient)
            unique_tracks = session.query(
                CustomerInteraction.track_id
            ).distinct().count()
            
            # Average age (more efficient query)
            from sqlalchemy import func
            avg_age_result = query.filter(
                CustomerInteraction.age.isnot(None)
            ).with_entities(func.avg(CustomerInteraction.age)).scalar()
            
            avg_age = float(avg_age_result) if avg_age_result else 0.0
            
            return {
                'total_interactions': total,
                'unique_customers': unique_tracks,
                'average_age': avg_age
            }
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error getting statistics: {e}", exc_info=True)
            return {
                'total_interactions': 0,
                'unique_customers': 0,
                'average_age': 0.0
            }
        finally:
            session.close()


if __name__ == "__main__":
    # Test
    db = DatabaseManager()
    print("⚡ Database initialized")
    
    # Add test interaction
    interaction_id = db.add_interaction(
        track_id=1,
        age=25.0,
        gender='female',
        emotion='happy',
        dwell_time=5.0,
        ad_shown='ad_001'
    )
    print(f"⚡ Added interaction: {interaction_id}")

