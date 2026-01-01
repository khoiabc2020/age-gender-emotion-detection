"""
Authentication API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from pydantic import BaseModel
from typing import Optional

from app.core.database import get_db
from app.core.config import settings
from app.core.security import create_access_token, verify_password, get_password_hash

router = APIRouter()

# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict

class UserInfo(BaseModel):
    username: str
    email: str
    full_name: str

# Simple in-memory user store (in production, use database)
# Default admin credentials
# Precomputed hash for "admin123" to avoid runtime hashing issues
USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyWE3Q7vZxBS",  # admin123
        "email": "admin@retail.com",
        "full_name": "Administrator"
    }
}

def authenticate_user(username: str, password: str) -> dict:
    """Authenticate user"""
    user = USERS_DB.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Login endpoint - returns JWT token
    Default credentials: admin / admin123
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "email": user["email"]},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "username": user["username"],
            "email": user["email"],
            "full_name": user["full_name"]
        }
    }

@router.get("/me")
async def get_current_user_info(
    current_user: dict = Depends(get_current_user)
):
    """Get current user information"""
    username = current_user.get("sub")
    user = USERS_DB.get(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "username": user["username"],
        "email": user["email"],
        "full_name": user["full_name"]
    }

