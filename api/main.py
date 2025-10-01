from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.models import PredictionRequest, PredictionResponse
from ml.inference.predictor import SentimentPredictor
from db.connection import create_tables  
from db.operations import (
    get_or_create_session, save_prediction, get_session_predictions,
    create_user, authenticate_user, create_user_session, convert_anonymous_session_to_user,
    get_user_sessions
)
from db.schemas import PredictionCreate, UserCreate, UserLogin, LoginResponse, UserResponse
from uuid import UUID
import logging

SENTIMENT_MAPPING = {
    0: 'negative',
    1: 'neutral', 
    2: 'positive'
}



# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment Analysis API")

# Add this startup event
@app.on_event("startup")
async def startup_event():
    """Create database tables on startup"""
    logger.info("Creating database tables...")
    create_tables()
    logger.info("Database tables ready")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Development: allow all origins
    # allow_origins=["https://your-frontend.com"],  # Production: specify exact domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Initialize predictor with error handling
try:
    # Try production model first, fallback to checkpoint
    model_paths = [
        "ml/models/production_sentiment_model.pth",
        "checkpoints/final_sentiment_model.pth"
    ]
    
    predictor = None
    for model_path in model_paths:
        try:
            predictor = SentimentPredictor(model_path=model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            break
        except FileNotFoundError:
            logger.warning(f"Model not found at {model_path}")
            continue
    
    if predictor is None:
        raise Exception("No trained model found")
        
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    predictor = None

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running!"}

@app.get("/health")
def health():
    if predictor is None:
        return {"status": "unhealthy", "error": "Model not loaded"}
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # 1. Handle session (create if needed, update if exists)
        session = get_or_create_session(request.session_id)
        
        # 2. Make ML prediction
        ml_result = predictor.predict(request.text)
        
        # 3. Save prediction to database
        prediction_data = PredictionCreate(
            session_id=session.session_id,
            user_id=session.user_id,  # None for anonymous
            text=request.text,
            predicted_sentiment=ml_result['prediction']['label'],
            confidence_score=ml_result['prediction']['confidence'],
            prob_positive=ml_result['probabilities']['positive'],
            prob_negative=ml_result['probabilities']['negative'],
            prob_neutral=ml_result['probabilities']['neutral']
        )
        
        saved_prediction = save_prediction(prediction_data)
        
        # 4. Return enhanced response
        return PredictionResponse(
            text=ml_result['text'],
            prediction=ml_result['prediction'],
            probabilities=ml_result['probabilities'],
            session_id=session.session_id,
            prediction_id=saved_prediction.prediction_id
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
def model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    
    return {
        "model_type": "DistilBERT", 
        "num_classes": predictor.model.num_classes,  
        "labels": list(SENTIMENT_MAPPING.values()),  
        "max_length": predictor.model.max_length,  
        "device": str(predictor.device),
        "model_name": predictor.model.distilbert.config.name_or_path  
    }

# get a users session ids and return the latest
@app.get("/users/{user_id}/sessions")
def get_user_sessions_by_id(user_id: UUID):
    try:
        sessions = get_user_sessions(user_id)
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Failed to get user sessions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user sessions")


@app.get("/sessions/{session_id}/predictions")
def get_session_predictions_endpoint(session_id: UUID):
    """Get all predictions for a session"""
    try:
        predictions = get_session_predictions(session_id)
        return {"predictions": predictions, "session_id": session_id}
    except Exception as e:
        logger.error(f"Failed to get predictions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve predictions")

@app.post("/register", response_model=UserResponse)
def register_user(user_data: UserCreate):
    """Register a new user"""
    try:
        user = create_user(user_data)
        logger.info(f"New user registered: {user.username}")
        return user
    except ValueError as e:
        # Handle duplicate username/email
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/login", response_model=LoginResponse)
def login_user(login_data: UserLogin, anonymous_session_id: UUID = None):
    """
    Login a user and create/convert session
    If anonymous_session_id is provided, converts that session to a user session
    """
    try:
        # Authenticate user
        user = authenticate_user(login_data.username, login_data.password)

        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password")

        # If there's an anonymous session, convert it
        if anonymous_session_id:
            try:
                session = convert_anonymous_session_to_user(anonymous_session_id, user.user_id)
                logger.info(f"User logged in and converted anonymous session: {user.username}")
            except ValueError:
                # Session not found or invalid, create new one
                session = create_user_session(user.user_id)
                logger.info(f"User logged in with new session: {user.username}")
        else:
            # Create a new user session
            session = create_user_session(user.user_id)
            logger.info(f"User logged in: {user.username}")

        return LoginResponse(
            user=UserResponse.model_validate(user),
            session_id=session.session_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/logout")
def logout_user(session_id: UUID):
    """Logout a user (currently just acknowledges, session expires naturally)"""
    try:
        # In a production app, you might want to invalidate the session here
        # For now, sessions expire naturally after 30 minutes of inactivity
        return {"message": "Logged out successfully", "session_id": session_id}
    except Exception as e:
        logger.error(f"Logout failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Logout failed")
