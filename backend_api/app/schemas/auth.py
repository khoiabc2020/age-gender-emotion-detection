"""
Authentication schemas
"""

from pydantic import BaseModel, EmailStr
from typing import Optional

class Token(BaseModel):
    """Token response schema"""
    access_token: str
    token_type: str
    user: dict

class UserCreate(BaseModel):
    """User creation schema"""
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class UserResponse(BaseModel):
    """User response schema"""
    username: str
    email: str
    full_name: Optional[str] = None

