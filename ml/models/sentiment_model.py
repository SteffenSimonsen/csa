import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from typing import Union, List, Dict
import torch.nn.functional as F

class SentimentModel(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', num_classes=3, max_length=512):
        super(SentimentModel, self).__init__()
        
        self.max_length = max_length
        self.num_classes = num_classes
        
        # Load pretrained DistilBERT and tokenizer
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
        
    def forward(self, texts: Union[str, List[str]], return_attention_weights=False):
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
            
        # Tokenize inputs
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to same device as model
        input_ids = encoded['input_ids'].to(self.distilbert.device)
        attention_mask = encoded['attention_mask'].to(self.distilbert.device)
        
        # Forward through DistilBERT
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if return_attention_weights:
            return logits, outputs.attentions
        return logits
    
    def predict(self, texts: Union[str, List[str]], return_probabilities=True):
        self.eval()
        with torch.no_grad():
            logits = self.forward(texts)
            
            if return_probabilities:
                probabilities = F.softmax(logits, dim=-1)
                return probabilities
            else:
                predictions = torch.argmax(logits, dim=-1)
                return predictions
    
    def predict_with_confidence(self, texts: Union[str, List[str]]) -> Dict:
        probabilities = self.predict(texts, return_probabilities=True)
        predictions = torch.argmax(probabilities, dim=-1)
        confidence_scores = torch.max(probabilities, dim=-1)[0]
        
        return {
            'predictions': predictions.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(), 
            'confidence': confidence_scores.cpu().numpy()
        }