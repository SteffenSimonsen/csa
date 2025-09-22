import streamlit as st
import requests
import json

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎭",
    layout="wide"
)

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar for history (fixed position)
with st.sidebar:
    st.header("📋 Analysis History")
    
    if st.session_state.history:
        # Summary stats
        sentiments = [item['result']['prediction']['label'] for item in st.session_state.history]
        positive_count = sentiments.count('positive')
        negative_count = sentiments.count('negative')
        neutral_count = sentiments.count('neutral')
        
        st.write(f"**Total Reviews**: {len(st.session_state.history)}")
        st.write(f"✅ {positive_count} | ⚠️ {neutral_count} | ❌ {negative_count}")
        
        # Clear history button
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.history = []
            st.rerun()
        
        st.divider()
        
        # Display history items
        for i, item in enumerate(st.session_state.history):
            with st.container():
                # Truncate text for display
                display_text = item['text'][:60] + "..." if len(item['text']) > 60 else item['text']
                
                prediction = item['result']['prediction']
                label = prediction['label']
                confidence = prediction['confidence']
                
                # Color-coded header
                if label == 'positive':
                    st.success(f"**{label.upper()}**")
                elif label == 'negative':
                    st.error(f"**{label.upper()}**")
                else:
                    st.warning(f"**{label.upper()}**")
                
                # Review text and confidence
                st.write(f"*{display_text}*")
                st.caption(f"Confidence: {confidence:.1%}")
                
                # Reanalyze button
                if st.button("🔄 Reanalyze", key=f"reanalyze_{i}"):
                    st.session_state.review_text = item['text']
                    st.rerun()
                
                st.divider()
    else:
        st.write("No reviews analyzed yet.")
        st.write("💡 Your history will appear here.")

# Title
st.title("🎭 Customer Review Sentiment Analyzer")
st.write("Analyze the sentiment of customer reviews using AI")

# API configuration
API_URL = "http://localhost:8000"

# Input section
st.header("Enter Review")
review_text = st.text_area(
    "Review Text", 
    value=st.session_state.get('review_text', ''),
    placeholder="Enter a customer review here...",
    height=150
)

# Submit button
if st.button("Analyze Sentiment", type="primary"):
    if review_text.strip():
        try:
            # Make API call
            with st.spinner("Analyzing sentiment..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": review_text},
                    timeout=10
                )
            
            if response.status_code == 200:
                result = response.json()
                
                # Add to history
                history_item = {
                    'text': review_text,
                    'result': result,
                    'timestamp': len(st.session_state.history)
                }
                st.session_state.history.insert(0, history_item)  # Add to beginning
                
                # Keep only last 20 items
                if len(st.session_state.history) > 20:
                    st.session_state.history = st.session_state.history[:20]
                
                # Display results
                st.header("Results")
                
                # Main prediction
                prediction = result['prediction']
                label = prediction['label']
                confidence = prediction['confidence']
                
                # Color coding for sentiment
                if label == 'positive':
                    st.success(f"**Sentiment**: {label.title()} (Confidence: {confidence:.1%})")
                elif label == 'negative':
                    st.error(f"**Sentiment**: {label.title()} (Confidence: {confidence:.1%})")
                else:
                    st.warning(f"**Sentiment**: {label.title()} (Confidence: {confidence:.1%})")
                
                # Probability breakdown
                st.subheader("Probability Breakdown")
                probabilities = result['probabilities']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Negative", f"{probabilities['negative']:.1%}")
                
                with col2:
                    st.metric("Neutral", f"{probabilities['neutral']:.1%}")
                
                with col3:
                    st.metric("Positive", f"{probabilities['positive']:.1%}")
            
            else:
                st.error(f"API Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure your FastAPI server is running on localhost:8000")
        except requests.exceptions.Timeout:
            st.error("❌ Request timed out. The model might be loading.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("⚠️ Please enter some review text")

# Instructions
with st.expander("ℹ️ How to use"):
    st.write("""
    1. Enter a customer review in the text box above
    2. Click 'Analyze Sentiment' to get predictions
    3. View the sentiment label with confidence score
    4. Check the probability breakdown for all sentiment classes
    5. See your analysis history in the sidebar
    
    **Note**: Make sure your FastAPI server is running on localhost:8000
    """)

# Example reviews
with st.expander("💡 Try these examples"):
    examples = [
        "This product is absolutely amazing! Best purchase I've ever made.",
        "Terrible quality. Complete waste of money. Very disappointed.",
        "It's okay, nothing special but does the job adequately."
    ]
    
    for i, example in enumerate(examples, 1):
        if st.button(f"Example {i}", key=f"example_{i}"):
            st.session_state.review_text = example
            st.rerun()
