import torch
from pathlib import Path
from ..models.sentiment_model import SentimentModel

SENTIMENT_MAPPING = {
    0: 'negative',
    1: 'neutral', 
    2: 'positive'
}


class SentimentPredictor:
    def __init__(self, model_path=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = SentimentModel(num_classes=3)
        
        # Load trained weights if provided
        if model_path:
            self.load_model(model_path)
        
        self.model.to(self.device)
        self.model.eval()
    
    def load_model(self, model_path):
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")
    
    def predict(self, text):
        # Input validation
        if text is None or not isinstance(text, str):
            raise ValueError("Input must be a string")
        if len(text.strip()) == 0:
            raise ValueError("Input text cannot be empty")
        
        with torch.no_grad():
            # Get model predictions
            logits = self.model(text)
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get prediction and confidence
            confidence, predicted_class = torch.max(probabilities, dim=-1)
            
            # Convert to Python types
            predicted_class = predicted_class.item()
            confidence = confidence.item()
            probs = probabilities[0].cpu().numpy()
        
        # Format output
        result = {
            'text': text,
            'prediction': {
                'label': SENTIMENT_MAPPING[predicted_class],
                'label_id': predicted_class,
                'confidence': float(confidence)
            },
            'probabilities': {
                'negative': float(probs[0]),
                'neutral': float(probs[1]),
                'positive': float(probs[2])
            }
        }
        
        return result
    
    def predict_proba(self, text):
        with torch.no_grad():
            logits = self.model(text)
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        return probabilities[0].cpu().numpy()
    
    def get_label(self, text):
        result = self.predict(text)
        return result['prediction']['label']
