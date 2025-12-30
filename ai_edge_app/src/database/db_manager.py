"""
Database Manager - SQLite + SQLAlchemy
Tuần 9: Local Database & Reporting
Quản lý database và queries
"""

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from .models import Base, CustomerInteraction, Session as SessionModel


class DatabaseManager:
    """
    Database Manager for SQLite
    """
    
    def __init__(self, db_path: str = "data/retail_analytics.db"):
        """
        Initialize Database Manager
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create engine
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def add_customer_interaction(
        self,
        track_id: int,
        age: Optional[float] = None,
        gender: Optional[str] = None,
        emotion: Optional[str] = None,
        dwell_time: Optional[float] = None,
        ad_shown: Optional[str] = None,
        ad_clicked: int = 0,
        bbox: Optional[tuple] = None,
        metadata: Optional[Dict] = None
    ) -> CustomerInteraction:
        """
        Add customer interaction record
        
        Args:
            track_id: Track ID
            age: Customer age
            gender: Customer gender
            emotion: Detected emotion
            dwell_time: Dwell time in seconds
            ad_shown: Ad ID shown
            ad_clicked: Whether ad was clicked (0 or 1)
            bbox: Bounding box (x, y, w, h)
            metadata: Additional metadata dict
            
        Returns:
            CustomerInteraction object
        """
        session = self.get_session()
        
        interaction = CustomerInteraction(
            track_id=track_id,
            timestamp=datetime.now(),
            age=age,
            gender=gender,
            emotion=emotion,
            dwell_time=dwell_time,
            ad_shown=ad_shown,
            ad_clicked=ad_clicked,
            bbox_x=bbox[0] if bbox else None,
            bbox_y=bbox[1] if bbox else None,
            bbox_w=bbox[2] if bbox else None,
            bbox_h=bbox[3] if bbox else None,
            metadata=json.dumps(metadata) if metadata else None
        )
        
        session.add(interaction)
        session.commit()
        session.refresh(interaction)
        session.close()
        
        return interaction
    
    def get_customer_interactions(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[CustomerInteraction]:
        """
        Get customer interactions
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of records
            
        Returns:
            List of CustomerInteraction objects
        """
        session = self.get_session()
        
        query = session.query(CustomerInteraction)
        
        if start_time:
            query = query.filter(CustomerInteraction.timestamp >= start_time)
        if end_time:
            query = query.filter(CustomerInteraction.timestamp <= end_time)
        
        query = query.order_by(CustomerInteraction.timestamp.desc())
        
        if limit:
            query = query.limit(limit)
        
        interactions = query.all()
        session.close()
        
        return interactions
    
    def get_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict:
        """
        Get statistics summary
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            Dictionary with statistics
        """
        session = self.get_session()
        
        query = session.query(CustomerInteraction)
        
        if start_time:
            query = query.filter(CustomerInteraction.timestamp >= start_time)
        if end_time:
            query = query.filter(CustomerInteraction.timestamp <= end_time)
        
        total = query.count()
        avg_age = query.with_entities(func.avg(CustomerInteraction.age)).scalar() or 0
        avg_dwell = query.with_entities(func.avg(CustomerInteraction.dwell_time)).scalar() or 0
        
        # Gender distribution
        male_count = query.filter(CustomerInteraction.gender == 'male').count()
        female_count = query.filter(CustomerInteraction.gender == 'female').count()
        
        # Emotion distribution
        emotion_counts = {}
        for emotion in ['happy', 'sad', 'angry', 'surprise', 'fear', 'neutral']:
            count = query.filter(CustomerInteraction.emotion == emotion).count()
            emotion_counts[emotion] = count
        
        session.close()
        
        return {
            'total_customers': total,
            'avg_age': round(avg_age, 1),
            'avg_dwell_time': round(avg_dwell, 2),
            'male_count': male_count,
            'female_count': female_count,
            'emotion_distribution': emotion_counts
        }
    
    def export_to_dict_list(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Export interactions to list of dictionaries
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            List of dictionaries
        """
        interactions = self.get_customer_interactions(start_time, end_time)
        
        result = []
        for interaction in interactions:
            result.append({
                'track_id': interaction.track_id,
                'timestamp': interaction.timestamp.isoformat(),
                'age': interaction.age,
                'gender': interaction.gender,
                'emotion': interaction.emotion,
                'dwell_time': interaction.dwell_time,
                'ad_shown': interaction.ad_shown,
                'ad_clicked': interaction.ad_clicked,
                'bbox': (interaction.bbox_x, interaction.bbox_y, interaction.bbox_w, interaction.bbox_h) if interaction.bbox_x else None,
                'metadata': json.loads(interaction.metadata) if interaction.metadata else None
            })
        
        return result

