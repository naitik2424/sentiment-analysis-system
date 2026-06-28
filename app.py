import os
import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px

# Set page config for a premium look
st.set_page_config(
    page_title="AI Sentiment Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling (dark mode and cards)
st.markdown("""
<style>
    .main {
        background-color: #0f1116;
        color: #e2e8f0;
    }
    .stApp {
        background: radial-gradient(circle, #1a1c24 0%, #0f1116 100%);
    }
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #ffffff;
    }
    .sentiment-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2d3142;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    }
    .positive-card {
        background-color: rgba(16, 185, 129, 0.1);
        border-color: #10b981;
    }
    .negative-card {
        background-color: rgba(239, 68, 68, 0.1);
        border-color: #ef4444;
    }
    .neutral-card {
        background-color: rgba(245, 158, 11, 0.1);
        border-color: #f59e0b;
    }
    .metric-title {
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Download NLTK VADER Lexicon if not already available
@st.cache_resource
def setup_nltk_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

# Lazy loading of Transformers pipeline
@st.cache_resource
def load_transformers_pipeline():
    try:
        from transformers import pipeline
        # Use a lightweight DistilBERT model
        classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        return classifier
    except ImportError:
        return None
    except Exception as e:
        st.error(f"Error loading Transformers: {e}")
        return None

# Initialize VADER
sia = setup_nltk_vader()

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.image("https://images.unsplash.com/photo-1507146426996-ef05306b995a?q=80&w=300&auto=format&fit=crop", use_container_width=True)
st.sidebar.title("🧠 Analyzer Controls")

model_choice = st.sidebar.selectbox(
    "Choose Sentiment Model",
    ["VADER (Lexicon-based)", "DistilBERT (Transformer Deep Learning)"],
    help="VADER is fast, lightweight, and rule-based. DistilBERT is a state-of-the-art transformer model."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Info")

if model_choice == "VADER (Lexicon-based)":
    st.sidebar.success("✅ **Active Model: VADER**")
    st.sidebar.info(
        "💡 **VADER (Valence Aware Dictionary and sEntiment Reasoner)** "
        "is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. "
        "It doesn't require machine learning training; it matches words to a valence dictionary."
    )
else:
    st.sidebar.success("✅ **Active Model: DistilBERT**")
    st.sidebar.info(
        "🤖 **DistilBERT** is a small, fast, cheap and light Transformer model "
        "trained by distilling BERT base. It has 40% fewer parameters than BERT-base-uncased, "
        "runs 60% faster, and preserves over 95% of BERT's performances on sentiment tasks."
    )

# --- MAIN APP INTERFACE ---
st.title("🧠 AI Sentiment Analysis Dashboard")
st.caption("Analyze the sentiment of text inputs using rule-based or deep learning approaches.")

# Layout: Split into tabs
tab1, tab2 = st.tabs([
    "✍️ Real-time Analysis", 
    "📁 Bulk CSV Upload & Charts"
])

# --- TAB 1: REAL-TIME ANALYSIS ---
with tab1:
    st.subheader("Analyze Text Sentiment in Real Time")
    user_text = st.text_area(
        "Enter text to analyze:",
        placeholder="Type something here... e.g., 'The service was incredibly fast, but the food was quite average. I might go back again.'",
        height=150
    )
    
    if st.button("Run Analysis", type="primary"):
        if user_text.strip() == "":
            st.warning("Please enter some text before analyzing.")
        else:
            with st.spinner("Analyzing sentiment..."):
                if model_choice == "VADER (Lexicon-based)":
                    scores = sia.polarity_scores(user_text)
                    compound = scores['compound']
                    
                    # Determine classification based on compound score
                    if compound >= 0.05:
                        label = "POSITIVE"
                        card_class = "positive-card"
                        emoji = "🟢"
                    elif compound <= -0.05:
                        label = "NEGATIVE"
                        card_class = "negative-card"
                        emoji = "🔴"
                    else:
                        label = "NEUTRAL"
                        card_class = "neutral-card"
                        emoji = "🟡"
                        
                    # Display results card
                    st.markdown(f"""
                    <div class="sentiment-card {card_class}">
                        <div class="metric-title">Sentiment Label</div>
                        <div class="metric-value">{label} {emoji}</div>
                        <p style="margin-top: 10px; margin-bottom: 0px;">
                            Compound Score: <strong>{compound:.4f}</strong> (Ranges from -1.0 to +1.0)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display sub-scores
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Positivity", f"{scores['pos']:.1%}")
                    col2.metric("Neutrality", f"{scores['neu']:.1%}")
                    col3.metric("Negativity", f"{scores['neg']:.1%}")
                    
                else:
                    # Transformer execution
                    classifier = load_transformers_pipeline()
                    if classifier is None:
                        st.error("Could not load Hugging Face pipeline. Make sure you have installed transformers and torch: `pip install transformers torch`")
                    else:
                        results = classifier(user_text)[0]
                        label = results['label']
                        score = results['score']
                        
                        card_class = "positive-card" if label == "POSITIVE" else "negative-card"
                        emoji = "🟢" if label == "POSITIVE" else "🔴"
                        
                        st.markdown(f"""
                        <div class="sentiment-card {card_class}">
                            <div class="metric-title">Sentiment Label</div>
                            <div class="metric-value">{label} {emoji}</div>
                            <p style="margin-top: 10px; margin-bottom: 0px;">
                                Model Confidence Score: <strong>{score:.2%}</strong>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

# --- TAB 2: BULK CSV UPLOAD ---
with tab2:
    st.subheader("Bulk Sentiment Analyzer")
    st.write("Upload a CSV file containing textual data (e.g. feedback, reviews, tweets) to analyze in bulk.")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        # Display preview
        st.write("Preview of original CSV:")
        st.dataframe(df_upload.head(), use_container_width=True)
        
        # Column selection
        text_column = st.selectbox(
            "Select the column containing the text reviews:",
            df_upload.columns
        )
        
        if st.button("Perform Bulk Analysis", type="primary"):
            with st.spinner("Analyzing all rows... (this might take a moment)"):
                labels = []
                scores_list = []
                
                if model_choice == "VADER (Lexicon-based)":
                    for text in df_upload[text_column]:
                        # Handle potential empty rows
                        text_str = str(text) if pd.notnull(text) else ""
                        scores = sia.polarity_scores(text_str)
                        compound = scores['compound']
                        scores_list.append(compound)
                        
                        if compound >= 0.05:
                            labels.append("POSITIVE")
                        elif compound <= -0.05:
                            labels.append("NEGATIVE")
                        else:
                            labels.append("NEUTRAL")
                else:
                    classifier = load_transformers_pipeline()
                    if classifier is None:
                        st.error("Transformers model is not loaded. Please install dependencies.")
                    else:
                        for text in df_upload[text_column]:
                            text_str = str(text) if pd.notnull(text) else ""
                            if text_str.strip() == "":
                                labels.append("NEUTRAL")
                                scores_list.append(0.5)
                            else:
                                res = classifier(text_str[:512])[0]  # Truncate to 512 tokens to avoid length errors
                                labels.append(res['label'])
                                scores_list.append(res['score'])
                                
                # Append findings to the dataframe
                df_upload['Sentiment_Label'] = labels
                df_upload['Sentiment_Score'] = scores_list
                
                st.success("Bulk analysis completed!")
                
                # Visualizations
                st.subheader("📊 Sentiment Analysis Visualizations")
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    # Pie Chart
                    fig_pie = px.pie(
                        df_upload, 
                        names='Sentiment_Label', 
                        title="Distribution of Sentiment Labels",
                        color='Sentiment_Label',
                        color_discrete_map={'POSITIVE': '#10b981', 'NEGATIVE': '#ef4444', 'NEUTRAL': '#f59e0b'}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                with col_chart2:
                    # Bar Chart of counts
                    fig_bar = px.histogram(
                        df_upload,
                        x='Sentiment_Label',
                        color='Sentiment_Label',
                        title="Count of Reviews by Sentiment",
                        color_discrete_map={'POSITIVE': '#10b981', 'NEGATIVE': '#ef4444', 'NEUTRAL': '#f59e0b'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Show results dataframe
                st.subheader("📋 Analysis Results Table")
                st.dataframe(df_upload, use_container_width=True)
                
                # Download Button
                csv_data = df_upload.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Analyzed Data as CSV",
                    data=csv_data,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv",
                    type="secondary"
                )


