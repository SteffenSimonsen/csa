from pydantic import BaseModel, Field, field_validator
from typing import Dict, Optional
from uuid import UUID

class PredictionRequest(BaseModel):
    text: str 
    session_id: Optional[UUID] = None
    
    @field_validator('text')
    def text_must_not_be_empty_or_whitespace(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v.strip()  # Return cleaned text

class Prediction(BaseModel):
    label: str
    label_id: int
    confidence: float

class PredictionResponse(BaseModel):
    text: str
    prediction: Prediction
    probabilities: Dict[str, float]
    session_id: UUID  
    prediction_id: UUID  
