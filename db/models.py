#db/models.py
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
import uuid
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)

class Session(Base):
    __tablename__ = 'sessions'

    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    is_anonymous = Column(Boolean(), default=True)

class Prediction(Base):
    __tablename__ = 'predictions'

    prediction_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('sessions.session_id'))
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'), nullable=True)
    text = Column(String())
    predicted_sentiment = Column(String)
    confidence_score = Column(Float) 
    prob_positive = Column(Float)
    prob_negative = Column(Float)
    prob_neutral = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
