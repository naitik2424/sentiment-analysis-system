# 🧠 AI Sentiment Analysis Dashboard

![App Banner](https://images.unsplash.com/photo-1507146426996-ef05306b995a?q=80&w=1000&auto=format&fit=crop)

A versatile sentiment analysis dashboard that allows users to process text in real-time or upload bulk CSV files for automated processing and visualization.

## 🌟 Key Features
- **Dual-Model Support:** Toggle between a fast, rule-based lexicon model (**NLTK VADER**) and a state-of-the-art transformer model (**Hugging Face DistilBERT**).
- **Real-Time Analysis:** Instantly analyze sentiment intensity of single sentences or paragraphs.
- **Bulk CSV Uploads:** Analyze thousands of reviews at once by uploading a CSV. The app automatically visualizes sentiment distributions with Plotly pie charts and histograms.
- **Data Export:** Download the fully processed, labeled dataset directly from the web interface.

## 🛠️ Tech Stack
- **Frontend & App Framework:** Streamlit
- **Language:** Python
- **Data Processing:** Pandas
- **Visualization:** Plotly
- **Natural Language Processing:** NLTK, Transformers (Hugging Face)
