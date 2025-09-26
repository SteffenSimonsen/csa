# db/operations.py
from db.connection import get_db
from db.models import User, Session as SessionModel, Prediction
from db.schemas import UserCreate, UserResponse, SessionResponse, PredictionCreate, PredictionResponse
from sqlalchemy.exc import  IntegrityError
from typing import Optional
from uuid import UUID
from datetime import datetime

def create_user(user_data: UserCreate) -> UserResponse:
    """
    Creates a new user in the database with proper validation
    """
    db = next(get_db())
    try:
        db_user = User(
            email=user_data.email,
            username=user_data.username
        )
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        return UserResponse.model_validate(db_user)
    
    except IntegrityError as e:
        db.rollback()
        # Check if it's a duplicate email or username
        if "users_email_key" in str(e):
            raise ValueError(f"Email {user_data.email} already exists")
        elif "users_username_key" in str(e):
            raise ValueError(f"Username {user_data.username} already exists")
        else:
            raise ValueError("User creation failed due to constraint violation")
    
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def get_user_by_id(user_id: UUID) -> Optional[UserResponse]:
    """
    Get user by their UUID
    """
    db = next(get_db())
    try:
        db_user = db.query(User).filter(User.user_id == user_id).first()
        if db_user:
            return UserResponse.model_validate(db_user)
        return None
    finally:
        db.close()

def get_user_by_email(email: str) -> Optional[UserResponse]:
    """
    Get user by their email address
    """
    db = next(get_db())
    try:
        db_user = db.query(User).filter(User.email == email).first()
        if db_user:
            return UserResponse.model_validate(db_user)
        return None
    finally:
        db.close()

def get_user_by_username(username: str) -> Optional[UserResponse]:
    """
    Get user by their username
    """
    db = next(get_db())
    try:
        db_user = db.query(User).filter(User.username == username.lower()).first()
        if db_user:
            return UserResponse.model_validate(db_user)
        return None
    finally:
        db.close()


def create_anonymous_session() -> SessionResponse:
    """Create a new anonymous session"""
    db = next(get_db())
    try:
        db_session = SessionModel(
            user_id=None,  # Anonymous
            is_anonymous=True
        )
        
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        
        return SessionResponse.model_validate(db_session)
    
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def create_user_session(user_id: UUID) -> SessionResponse:
    """Create a session for a logged-in user"""
    db = next(get_db())
    try:
        db_session = SessionModel(
            user_id=user_id,
            is_anonymous=False
        )
        
        db.add(db_session)
        db.commit() 
        db.refresh(db_session)
        
        return SessionResponse.model_validate(db_session)
    
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def get_session_by_id(session_id: UUID) -> Optional[SessionResponse]:
    """Get session by ID and check if it's expired"""
    db = next(get_db())
    try:
        db_session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
        if db_session:
            session_response = SessionResponse.model_validate(db_session)
            return session_response
        return None
    finally:
        db.close()

def update_session_activity(session_id: UUID) -> Optional[SessionResponse]:
    """Update the last_active timestamp for a session"""
    db = next(get_db())
    try:
        db_session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
        if not db_session:
            return None
            
        # Update last_active timestamp
        db_session.last_active = datetime.utcnow()
        db.commit()
        db.refresh(db_session)
        
        return SessionResponse.model_validate(db_session)
        
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def get_or_create_session(session_id: Optional[UUID] = None) -> SessionResponse:
    """
    Core function for API calls:
    - If session_id provided and valid -> update activity and return it
    - If session_id expired or invalid -> create new anonymous session
    - If no session_id -> create new anonymous session
    """
    if session_id:
        session = get_session_by_id(session_id)
        if session and not session.is_expired:
            # Valid session exists, update activity
            return update_session_activity(session_id)
    
    # Create new anonymous session if none exists or expired
    return create_anonymous_session()




def save_prediction(prediction_data: PredictionCreate) -> PredictionResponse:
    """
    Save a prediction and update session activity
    """
    db = next(get_db())
    try:
        # Create the prediction
        db_prediction = Prediction(
            session_id=prediction_data.session_id,
            user_id=prediction_data.user_id,
            text=prediction_data.text,
            predicted_sentiment=prediction_data.predicted_sentiment,
            confidence_score=prediction_data.confidence_score,
            prob_positive=prediction_data.prob_positive,
            prob_negative=prediction_data.prob_negative,
            prob_neutral=prediction_data.prob_neutral
        )
        
        db.add(db_prediction)
        
        # Update session last_active timestamp
        session = db.query(SessionModel).filter(SessionModel.session_id == prediction_data.session_id).first()
        if session:
            session.last_active = datetime.utcnow()
        
        db.commit()
        db.refresh(db_prediction)
        
        return PredictionResponse.model_validate(db_prediction)
        
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def get_session_predictions(session_id: UUID) -> list[PredictionResponse]:
    """
    Get all predictions for a specific session
    """
    db = next(get_db())
    try:
        predictions = db.query(Prediction).filter(Prediction.session_id == session_id).order_by(Prediction.created_at.desc()).all()
        return [PredictionResponse.model_validate(pred) for pred in predictions]
    finally:
        db.close()

def get_user_predictions(user_id: UUID) -> list[PredictionResponse]:
    """
    Get all predictions for a specific user across all their sessions
    """
    db = next(get_db())
    try:
        predictions = db.query(Prediction).filter(Prediction.user_id == user_id).order_by(Prediction.created_at.desc()).all()
        return [PredictionResponse.model_validate(pred) for pred in predictions]
    finally:
        db.close()
