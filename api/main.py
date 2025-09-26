from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.models import PredictionRequest, PredictionResponse
from ml.inference.predictor import SentimentPredictor
from data.data_preprocessing import SENTIMENT_MAPPING
from db.operations import get_or_create_session, save_prediction, get_session_predictions
from db.schemas import PredictionCreate
from uuid import UUID
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment Analysis API")

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


@app.get("/sessions/{session_id}/predictions")
def get_session_predictions_endpoint(session_id: UUID):
    """Get all predictions for a session"""
    try:
        predictions = get_session_predictions(session_id)
        return {"predictions": predictions, "session_id": session_id}
    except Exception as e:
        logger.error(f"Failed to get predictions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve predictions")
