import streamlit as st
import requests
import json
import os
import plotly.express as px
import pandas as pd


# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎭",
    layout="wide"
)


# Sidebar for API status
with st.sidebar:
    st.header("🔧 System Status")
    
    # API Health Check
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            st.success("✅ API Online")
            st.write(f"Status: {health_data.get('status', 'Unknown')}")
            if health_data.get('model_loaded'):
                st.write("🤖 Model: Loaded")
            else:
                st.warning("⚠️ Model: Not Loaded")
        else:
            st.error("❌ API Error")
            st.write(f"Status Code: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ API Offline")
        st.write("Cannot connect to localhost:8000")
    except requests.exceptions.Timeout:
        st.warning("⏱️ API Timeout")
        st.write("Server not responding")
    except Exception as e:
        st.error("❌ Unknown Error")
        st.write(f"Error: {str(e)}")
    
    # Refresh button
    if st.button("🔄 Check Status"):
        st.rerun()
    
    
# Title
st.title("🎭 Sentiment Analyzer for Customer Reviews ")
st.write("Analyze the sentiment of customer reviews using AI")

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎭 Analyze", "📋 History", "📊 Analytics"])

with tab1:
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
                    
                    # Display results in card format
                    st.header("Results")

                    prediction = result['prediction']
                    label = prediction['label']
                    confidence = prediction['confidence']
                    probabilities = result['probabilities']

                    # Main result card
                    with st.container(border=True):
                        if label == 'positive':
                            st.success(f"✅ POSITIVE")
                        elif label == 'negative':
                            st.error(f"❌ NEGATIVE")
                        else:
                            st.warning(f"⚠️ NEUTRAL")
                        
                        st.markdown("**Confidence:**")
                        st.progress(confidence, text=f"{confidence:.1%}")
                        
                        st.markdown("**Probability Breakdown:**")
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

    with st.expander("ℹ️ How to use"):
        st.write("""
        1. Enter a customer review in the text box above
        2. Click 'Analyze Sentiment' to get predictions  
        3. View the sentiment classification with confidence score
        4. Check the probability breakdown for all three sentiment classes
        5. Visit the **History** tab to see all your previous analyses
        6. Check the **System Status** in the sidebar to ensure the API is running
        7. Try the example reviews below to test different sentiments
        
        **Tip:** Your analysis history is saved during this session - switch to the History tab to review past results!
        """)

    with st.expander("💡 Try these examples"):
        examples = [
            "This product is absolutely amazing! Best purchase I've ever made.",
            "Terrible quality. Complete waste of money. Very disappointed.",
            "It's okay, nothing special but does the job adequately.",
            "Love the packaging and fast shipping, but the product itself was smaller than expected.",
            "Exceeded my expectations! Great value for money and excellent customer service.",
            "Not sure if I like this yet. Need to use it more to form an opinion.",
            "Complete garbage. Broke after one day. Don't waste your time.",
            "Pretty good overall. A few minor issues but would recommend to others.",
            "Mixed feelings about this purchase. Some good points, some bad.",
            "Outstanding quality! This company really knows what they're doing."
        ]
        
        for i, example in enumerate(examples):
            if st.button(example, key=f"example_{i}"):
                st.session_state.review_text = example
                st.rerun()



with tab2:
    st.header("📋 Analysis History")
    
    if st.session_state.history:
        # Summary stats at the top
        sentiments = [item['result']['prediction']['label'] for item in st.session_state.history]
        positive_count = sentiments.count('positive')
        negative_count = sentiments.count('negative')
        neutral_count = sentiments.count('neutral')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Reviews", len(st.session_state.history))
        with col2:
            st.metric("Positive", positive_count)
        with col3:
            st.metric("Neutral", neutral_count)
        with col4:
            st.metric("Negative", negative_count)
        
        # Clear history button
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.history = []
            st.rerun()
        
        st.divider()
        
        
        for i, item in enumerate(st.session_state.history):
            prediction = item['result']['prediction']
            label = prediction['label']
            confidence = prediction['confidence']
            
            # Create colored card based on sentiment
            if label == 'positive':
                with st.container(border=True):
                    st.success(f"✅ POSITIVE - Review #{len(st.session_state.history) - i}")
                    st.markdown("**Review Text:**")
                    st.write(f"*{item['text']}*")
                    st.markdown("**Confidence:**")
                    st.progress(confidence, text=f"{confidence:.1%}")
                    st.write("")
                    
            elif label == 'negative':
                with st.container(border=True):
                    st.error(f"❌ NEGATIVE - Review #{len(st.session_state.history) - i}")
                    st.markdown("**Review Text:**")
                    st.write(f"*{item['text']}*")
                    st.markdown("**Confidence:**")
                    st.progress(confidence, text=f"{confidence:.1%}")
                    st.write("")
                    
            else:  
                with st.container(border=True):
                    st.warning(f"⚠️ NEUTRAL - Review #{len(st.session_state.history) - i}")
                    st.markdown("**Review Text:**")
                    st.write(f"*{item['text']}*")
                    st.markdown("**Confidence:**")
                    st.progress(confidence, text=f"{confidence:.1%}")
                    st.write("")
    else:
        st.info("No reviews analyzed yet. Go to the Analyze tab to get started!")

with tab3:
    st.header("📊 Analytics Dashboard")
    
    if st.session_state.history:
        # Get sentiment data
        sentiments = [item['result']['prediction']['label'] for item in st.session_state.history]
        
        # Sentiment counts
        positive_count = sentiments.count('positive')
        negative_count = sentiments.count('negative')
        neutral_count = sentiments.count('neutral')
        total = len(sentiments)
        
        
        sentiment_data = pd.DataFrame({
            'Sentiment': ['Positive', 'Negative', 'Neutral'],
            'Count': [positive_count, negative_count, neutral_count]
        })
        
        st.subheader("Sentiment Distribution")

        if total > 0:
            import plotly.express as px
            
            # Create pie chart data
            labels = []
            values = []
            colors = []
            
            if positive_count > 0:
                labels.append('Positive')
                values.append(positive_count)
                colors.append('#28a745')  # Green
            
            if negative_count > 0:
                labels.append('Negative') 
                values.append(negative_count)
                colors.append('#dc3545')  # Red
                
            if neutral_count > 0:
                labels.append('Neutral')
                values.append(neutral_count) 
                colors.append('#ffc107')  # Yellow
            
            # Create pie chart
            fig = px.pie(values=values, names=labels, color_discrete_sequence=colors)
            st.plotly_chart(fig, use_container_width=True)

        # Confidence Score Analysis
        st.subheader("How confident is the model in its predictions?")

        if total > 0:
            # Get confidence scores
            confidences = [item['result']['prediction']['confidence'] for item in st.session_state.history]
        
        
            confidence_df = pd.DataFrame({
                'Confidence': confidences,
                'Sentiment': sentiments
            })
        
            
            fig = px.histogram(
                confidence_df,
                y='Confidence',
                color='Sentiment',
                nbins=10,
                orientation='h',
                color_discrete_map={
                    'positive': '#28a745',
                    'negative': '#dc3545',
                    'neutral': '#ffc107'
                }
            )
            fig.update_layout(
                xaxis_title="Number of Reviews",
                yaxis_title="Confidence Score",
                xaxis=dict(
                    dtick=1,                    
                    tickmode='linear'           
                )
            )
        
            st.plotly_chart(fig, use_container_width=True)
        
            # Show confidence stats
            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
            max_confidence = max(confidences)
        
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Confidence", f"{avg_confidence:.1%}")
            with col2:
                st.metric("Lowest Confidence", f"{min_confidence:.1%}")
            with col3:
                st.metric("Highest Confidence", f"{max_confidence:.1%}")

            
        # Average Confidence by Sentiment  
        st.subheader("Is the model more confident about certain sentiments?")

        if total > 0:
            # Calculate average confidence for each sentiment
            sentiment_confidence = {}
            for sentiment in ['positive', 'negative', 'neutral']:
                sentiment_reviews = [confidences[i] for i, s in enumerate(sentiments) if s == sentiment]
                if sentiment_reviews:
                    sentiment_confidence[sentiment] = sum(sentiment_reviews) / len(sentiment_reviews)
            
            if sentiment_confidence:
                
                sentiment_names = list(sentiment_confidence.keys())
                confidence_values = list(sentiment_confidence.values())
                
                fig = px.bar(
                    x=sentiment_names,
                    y=confidence_values,
                    color=sentiment_names,
                    color_discrete_map={
                        'positive': '#28a745',
                        'negative': '#dc3545', 
                        'neutral': '#ffc107'
                    }
                )
                
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',     
                    paper_bgcolor='rgba(0,0,0,0)',    
                    font=dict(color='white', size=12),  
                    showlegend=False,  
                    xaxis_title="Sentiment Type",
                    yaxis_title="Average Confidence",
                    margin=dict(l=40, r=40, t=60, b=40)  
                )
                
                
                fig.update_traces(
                    marker_line_width=0,               
                    texttemplate='%{y:.1%}',           
                    textposition='outside',            
                    textfont_size=12
                )
                
              
                fig.update_layout(
                    xaxis=dict(
                        tickfont_size=12,
                        title_font_size=14
                    ),
                    yaxis=dict(
                        tickformat='.0%',           
                        tickfont_size=12,
                        title_font_size=14
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
            
    else:
        st.info("No data to display yet. Analyze some reviews first!")
            
