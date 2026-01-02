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
from app.core.database import engine, Base, SessionLocal
from app.api import analytics, dashboard, ads, auth, websocket, ai_agent
from app.core.security import verify_token, get_password_hash
from app.models.user import User

# Create database tables on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI app"""
    # Startup
    try:
        Base.metadata.create_all(bind=engine)
        
        # Create default admin user if not exists
        db = SessionLocal()
        try:
            admin_user = db.query(User).filter(User.username == "admin").first()
            if not admin_user:
                admin_user = User(
                    username="admin",
                    email="admin@retail.com",
                    hashed_password=get_password_hash("admin123"),
                    full_name="Administrator",
                    is_active=True,
                    is_superuser=True
                )
                db.add(admin_user)
                db.commit()
                print("INFO: Default admin user created (admin/admin123)")
            else:
                print("INFO: Admin user already exists")
        finally:
            db.close()
    except Exception as e:
        # Database connection failed - continue without database
        print(f"WARNING: Database connection failed: {e}")
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
# Add alias routes for convenience (backward compatibility)
app.include_router(auth.router, prefix="/auth", tags=["authentication"])
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
    import socket
    
    # Check if port is available, try alternative ports if needed
    def find_free_port(start_port: int, max_attempts: int = 10) -> int:
        """Find a free port starting from start_port"""
        for i in range(max_attempts):
            port = start_port + i
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(('0.0.0.0', port))
                sock.close()
                return port
            except OSError:
                continue
        return start_port  # Return original if all fail
    
    port = find_free_port(settings.PORT)
    if port != settings.PORT:
        print(f"Port {settings.PORT} is in use, using port {port} instead")
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=port,
        reload=settings.DEBUG
    )

