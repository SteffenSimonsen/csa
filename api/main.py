from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.models import PredictionRequest, PredictionResponse
from ml.inference.predictor import SentimentPredictor
from data.data_preprocessing import SENTIMENT_MAPPING
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
    predictor = SentimentPredictor()
    logger.info("Model loaded successfully")
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
    # Check if model is loaded
    if predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not available. Service is unavailable."
        )
    
    
    try:
        # Make prediction
        result = predictor.predict(request.text)
        logger.info(f"Prediction made for text: {request.text[:50]}...")
        return result
        
    except Exception as e:
        # Log the error
        logger.error(f"Prediction failed: {str(e)}")
        
        # Return user-friendly error
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

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