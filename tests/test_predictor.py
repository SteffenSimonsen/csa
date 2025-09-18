import pytest
import torch
from ml.inference.predictor import SentimentPredictor

class TestSentimentPredictor:
    
    @pytest.fixture
    def predictor(self):
        # Create predictor with untrained model for testing
        return SentimentPredictor()
    
    def test_predictor_initialization(self, predictor):
        assert predictor.device in ['cuda', 'cpu']
        assert predictor.model is not None
        assert not predictor.model.training  # Should be in eval mode
        
    def test_predict_returns_correct_structure(self, predictor):
        result = predictor.predict("Great product!")
        
        assert 'text' in result
        assert 'prediction' in result
        assert 'probabilities' in result
        
        assert 'label' in result['prediction']
        assert 'label_id' in result['prediction']
        assert 'confidence' in result['prediction']
        
        assert 'negative' in result['probabilities']
        assert 'neutral' in result['probabilities']
        assert 'positive' in result['probabilities']
        
    def test_predict_label_is_valid(self, predictor):
        result = predictor.predict("Test text")
        
        assert result['prediction']['label'] in ['negative', 'neutral', 'positive']
        assert result['prediction']['label_id'] in [0, 1, 2]
        
    def test_probabilities_sum_to_one(self, predictor):
        result = predictor.predict("Test text")
        
        probs = result['probabilities']
        total = probs['negative'] + probs['neutral'] + probs['positive']
        
        assert abs(total - 1.0) < 0.001  # Should sum to ~1.0
        
    def test_confidence_in_valid_range(self, predictor):
        result = predictor.predict("Test text")
        
        confidence = result['prediction']['confidence']
        assert 0 <= confidence <= 1
        
    def test_text_preserved_in_output(self, predictor):
        text = "This is my test review"
        result = predictor.predict(text)
        
        assert result['text'] == text
        
    def test_predict_proba_returns_array(self, predictor):
        probs = predictor.predict_proba("Test text")
        
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 0.001
        
    def test_get_label_returns_string(self, predictor):
        label = predictor.get_label("Test text")
        
        assert isinstance(label, str)
        assert label in ['negative', 'neutral', 'positive']
        
    def test_empty_string_validation(self, predictor):
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            predictor.predict("")
            
    def test_none_input_validation(self, predictor):
        with pytest.raises(ValueError, match="Input must be a string"):
            predictor.predict(None)
            
    def test_non_string_input_validation(self, predictor):
        with pytest.raises(ValueError, match="Input must be a string"):
            predictor.predict(123)
            
    def test_handles_long_text(self, predictor):
        long_text = "A" * 10000
        result = predictor.predict(long_text)
        
        # Should complete without error
        assert 'prediction' in result
        assert result['text'] == long_text
        
    def test_handles_special_characters(self, predictor):
        text = "Great! @#$% Amazing product!!! 😊"
        result = predictor.predict(text)
        
        assert 'prediction' in result
        assert result['text'] == text
        
    def test_model_in_eval_mode(self, predictor):
        # Verify model stays in eval mode during prediction
        predictor.predict("Test")
        assert not predictor.model.training
        
    def test_predictions_are_deterministic(self, predictor):
        # Same input should give same output (since in eval mode, no dropout)
        text = "Consistent predictions test"
        
        result1 = predictor.predict(text)
        result2 = predictor.predict(text)
        
        assert result1['prediction']['label_id'] == result2['prediction']['label_id']
        assert abs(result1['prediction']['confidence'] - result2['prediction']['confidence']) < 0.001