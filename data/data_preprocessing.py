# data/src/data_preprocessing.py
import re

SENTIMENT_MAPPING = {
    0: 'negative',  # ratings 1-2
    1: 'neutral',   # rating 3
    2: 'positive'   # ratings 4-5
}

def preprocess_text(text):
    # Fix HTML entities
    text = text.replace('&#34;', '"')
    text = text.replace('&#39;', "'")
    
    # Remove HTML tags
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Fix spacing after periods
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    
    text = text.strip()
    
    return text

def clean_and_label_sample(sample):
    # Create sentiment label
    rating = sample['rating']
    if rating <= 2:
        sentiment = 0  # negative
    elif rating == 3:
        sentiment = 1  # neutral  
    else:
        sentiment = 2  # positive
    
    # Return cleaned sample with only needed fields
    return {
        'text': preprocess_text(sample['text']),  # Apply text preprocessing
        'title': preprocess_text(sample['title']), # Apply to title too
        'rating': sample['rating'],
        'sentiment': sentiment,
        'asin': sample['asin'],
        'parent_asin': sample['parent_asin'],
        'verified_purchase': sample['verified_purchase'],
        'helpful_vote': sample['helpful_vote']
    }