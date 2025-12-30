"""
Application configuration
"""

from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/retail_analytics"
    
    # Security
    # ⚠️ IMPORTANT: Set SECRET_KEY in .env file for production!
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8501"]
    
    # MQTT
    MQTT_BROKER: str = "localhost"
    MQTT_PORT: int = 1883
    MQTT_TOPIC: str = "retail/analytics"
    
    # Redis (optional, for caching)
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # AI Agent Configuration (Giai đoạn 6)
    GOOGLE_AI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    AI_PROVIDER: str = "google_ai"  # google_ai, chatgpt, or both
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

