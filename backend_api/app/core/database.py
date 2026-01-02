"""
Database configuration and session management
"""

from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
import os

# Determine database URL - use SQLite fallback if PostgreSQL unavailable
def get_database_url():
    """Get database URL with SQLite fallback"""
    if getattr(settings, 'USE_SQLITE_FALLBACK', True):
        # Try PostgreSQL first, fallback to SQLite
        try:
            # Test PostgreSQL connection
            test_engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True, connect_args={"connect_timeout": 2})
            with test_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return settings.DATABASE_URL
        except Exception as e:
            # PostgreSQL not available, use SQLite
            sqlite_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "retail_analytics.db"))
            print(f"INFO: PostgreSQL not available ({e}), using SQLite: {sqlite_path}")
            return f"sqlite:///{sqlite_path}"
    return settings.DATABASE_URL

# Create database engine
database_url = get_database_url()
is_sqlite = "sqlite" in database_url.lower()

engine = create_engine(
    database_url,
    pool_pre_ping=True,
    pool_size=10 if not is_sqlite else 1,
    max_overflow=20 if not is_sqlite else 0,
    connect_args={"check_same_thread": False} if is_sqlite else {}
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

