# db/schemas.py
from pydantic import BaseModel, EmailStr, Field, field_validator
from uuid import UUID
from datetime import datetime, timedelta
from typing import Optional

#User Schemas
class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=100)

    @field_validator('username')
    def validate_username(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username can only contain letters, numbers, hyphens, and underscores')
        return v.lower()  # Normalize to lowercase

    @field_validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(char.islower() for char in v):
            raise ValueError('Password must contain at least one lowercase letter')
        return v

class UserResponse(BaseModel):
    user_id: UUID
    email: str
    username: str
    created_at: datetime
    last_active: datetime

    class Config:
        from_attributes = True  # Allows conversion from SQLAlchemy models

class UserLogin(BaseModel):
    username: str  # Can be username or email
    password: str

class LoginResponse(BaseModel):
    user: UserResponse
    session_id: UUID
    message: str = "Login successful"

#Session Schemas
class SessionCreate(BaseModel):
    user_id: Optional[UUID] = None  # None for anonymous sessions
    

class SessionResponse(BaseModel):
    session_id: UUID
    user_id: Optional[UUID]
    created_at: datetime
    last_active: datetime
    is_anonymous: bool
    
    class Config:
        from_attributes = True
    
    @property
    def is_expired(self) -> bool:
        # Check if session expired (30 minutes of inactivity)
        return datetime.utcnow() - self.last_active > timedelta(minutes=30)


#Prediciton Schemas
class PredictionCreate(BaseModel):
    session_id: UUID
    user_id: Optional[UUID] = None  # Can be null for anonymous predictions
    text: str = Field(min_length=1, max_length=10000)  # Reasonable limits
    predicted_sentiment: str = Field(pattern="^(positive|negative|neutral)$")  # Validate sentiment values
    confidence_score: float = Field(ge=0.0, le=1.0)  # Between 0 and 1
    prob_positive: float = Field(ge=0.0, le=1.0)
    prob_negative: float = Field(ge=0.0, le=1.0)
    prob_neutral: float = Field(ge=0.0, le=1.0)
    
    @field_validator('text')
    def validate_text(cls, v):
        return v.strip()  # Remove extra whitespace

class PredictionResponse(BaseModel):
    prediction_id: UUID
    session_id: UUID
    user_id: Optional[UUID]
    text: str
    predicted_sentiment: str
    confidence_score: float
    prob_positive: float
    prob_negative: float
    prob_neutral: float
    created_at: datetime
    
    class Config:
        from_attributes = True
