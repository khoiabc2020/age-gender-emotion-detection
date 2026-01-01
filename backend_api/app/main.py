"""
FastAPI Main Application
Backend API for Smart Retail Analytics System
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import uvicorn

from app.core.config import settings
from app.core.database import engine, Base
from app.api import analytics, dashboard, ads, auth, websocket, ai_agent
from app.core.security import verify_token

# Create database tables on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI app"""
    # Startup
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        # Database connection failed - continue without database
        print(f"Warning: Database connection failed: {e}")
        print("Continuing without database. Some features may be limited.")
    yield
    # Shutdown (if needed)

# Initialize FastAPI app
app = FastAPI(
    title="Smart Retail Analytics API",
    description="Backend API for customer analytics and targeted advertising",
    version="3.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Dependency for authentication
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Verify JWT token"""
    token = credentials.credentials
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
app.include_router(
    dashboard.router, 
    prefix="/api/v1/dashboard", 
    tags=["dashboard"],
    dependencies=[Depends(get_current_user)]
)
app.include_router(ads.router, prefix="/api/v1/ads", tags=["ads"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])
app.include_router(
    ai_agent.router,
    prefix="/api/v1/ai",
    tags=["ai-agent"],
    dependencies=[Depends(get_current_user)]
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Smart Retail Analytics API",
        "version": "3.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )

