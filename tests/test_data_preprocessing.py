import pytest
from data.data_preprocessing import preprocess_text, clean_and_label_sample, SENTIMENT_MAPPING

class TestPreprocessText:
    def test_removes_html_tags(self):
        text = "Great product <br> highly recommend"
        result = preprocess_text(text)
        assert "<br>" not in result
        assert "Great product" in result    
        
    def test_fixes_html_entities(self):
        text = "It&#39;s amazing &#34;truly&#34;"
        result = preprocess_text(text)
        assert "'" in result
        assert '"' in result
        assert "&#39;" not in result
        assert "&#34;" not in result
        
    def test_fixes_spacing_after_periods(self):
        text = "Great.Amazing product"
        result = preprocess_text(text)
        assert result == "Great. Amazing product"
        
    def test_strips_whitespace(self):
        text = "  Great product  "
        result = preprocess_text(text)
        assert result == "Great product"
        
    def test_handles_empty_string(self):
        result = preprocess_text("")
        assert result == ""
        
    def test_handles_multiple_issues(self):
        text = "  Great&#39;s product<br>Amazing.Works well  "
        result = preprocess_text(text)
        assert "Great's product" in result
        assert "<br>" not in result
        assert ". Works" in result

class TestCleanAndLabelSample:
    def test_negative_sentiment_rating_1(self):
        sample = {
            'text': 'Bad product',
            'title': 'Terrible',
            'rating': 1.0,
            'asin': 'B123',
            'parent_asin': 'P123',
            'verified_purchase': True,
            'helpful_vote': 0
        }
        result = clean_and_label_sample(sample)
        assert result['sentiment'] == 0  # negative
        
    def test_negative_sentiment_rating_2(self):
        sample = {
            'text': 'Poor quality',
            'title': 'Not good',
            'rating': 2.0,
            'asin': 'B123',
            'parent_asin': 'P123',
            'verified_purchase': True,
            'helpful_vote': 0
        }
        result = clean_and_label_sample(sample)
        assert result['sentiment'] == 0  # negative
        
    def test_neutral_sentiment_rating_3(self):
        sample = {
            'text': 'Okay product',
            'title': 'Average',
            'rating': 3.0,
            'asin': 'B123',
            'parent_asin': 'P123',
            'verified_purchase': True,
            'helpful_vote': 5
        }
        result = clean_and_label_sample(sample)
        assert result['sentiment'] == 1  # neutral
        
    def test_positive_sentiment_rating_4(self):
        sample = {
            'text': 'Good product',
            'title': 'Nice',
            'rating': 4.0,
            'asin': 'B123',
            'parent_asin': 'P123',
            'verified_purchase': False,
            'helpful_vote': 10
        }
        result = clean_and_label_sample(sample)
        assert result['sentiment'] == 2  # positive
        
    def test_positive_sentiment_rating_5(self):
        sample = {
            'text': 'Excellent product',
            'title': 'Amazing',
            'rating': 5.0,
            'asin': 'B123',
            'parent_asin': 'P123',
            'verified_purchase': True,
            'helpful_vote': 20
        }
        result = clean_and_label_sample(sample)
        assert result['sentiment'] == 2  # positive
        
    def test_cleans_text_and_title(self):
        sample = {
            'text': '<br>Great product',
            'title': 'Amazing&#39;s',
            'rating': 5.0,
            'asin': 'B123',
            'parent_asin': 'P123',
            'verified_purchase': True,
            'helpful_vote': 0
        }
        result = clean_and_label_sample(sample)
        assert '<br>' not in result['text']
        assert "Amazing's" in result['title']
        
    def test_preserves_required_fields(self):
        sample = {
            'text': 'Great',
            'title': 'Title',
            'rating': 5.0,
            'asin': 'B123',
            'parent_asin': 'P456',
            'verified_purchase': True,
            'helpful_vote': 10
        }
        result = clean_and_label_sample(sample)
        
        assert 'text' in result
        assert 'title' in result
        assert 'rating' in result
        assert 'sentiment' in result
        assert 'asin' in result
        assert 'parent_asin' in result
        assert 'verified_purchase' in result
        assert 'helpful_vote' in result

class TestSentimentMapping:
    def test_mapping_values(self):
        assert SENTIMENT_MAPPING[0] == 'negative'
        assert SENTIMENT_MAPPING[1] == 'neutral'
        assert SENTIMENT_MAPPING[2] == 'positive'
        
    def test_mapping_complete(self):
        assert len(SENTIMENT_MAPPING) == 3