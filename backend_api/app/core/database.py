"""
Database configuration and session management
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
import os

# Determine database URL - use SQLite fallback if PostgreSQL unavailable
def get_database_url():
    """Get database URL with SQLite fallback"""
    if settings.USE_SQLITE_FALLBACK:
        # Try PostgreSQL first, fallback to SQLite
        try:
            # Test PostgreSQL connection
            test_engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
            with test_engine.connect() as conn:
                conn.execute("SELECT 1")
            return settings.DATABASE_URL
        except Exception:
            # PostgreSQL not available, use SQLite
            sqlite_path = os.path.join(os.path.dirname(__file__), "..", "..", "retail_analytics.db")
            print(f"INFO: PostgreSQL not available, using SQLite: {sqlite_path}")
            return f"sqlite:///{sqlite_path}"
    return settings.DATABASE_URL

# Create database engine
database_url = get_database_url()
engine = create_engine(
    database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    connect_args={"check_same_thread": False} if "sqlite" in database_url else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

