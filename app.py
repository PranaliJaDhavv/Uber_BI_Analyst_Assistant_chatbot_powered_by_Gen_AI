import streamlit as st
import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from groq import Groq

# Apply custom CSS for aesthetic
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    html, body {
        background-color: #000 !important;
        color: #fff;
    }
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #111;
    }
    .css-1d391kg {
        background-color: #fff;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    button, .stButton>button {
        font-family: 'Inter', sans-serif;
        border-radius: 8px;
        border: none;
        background-color: #00bfa5;
        color: #fff;
        padding: 0.5rem 1rem;
        transition: background-color 0.2s ease;
    }
    button:hover, .stButton>button:hover {
        background-color: #009e8d;
    }
    .stMetric {
        background-color: #fff;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        padding: 1rem;
        transition: box-shadow 0.2s ease;
    }
    .stMetric:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .st-expanderHeader, .stDropdown > button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #111;
    }
    a {
        color: #00bfa5;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Data Loading and Cleaning ---
@st.cache_data
def load_and_clean_data(uploaded_file):
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    df.drop_duplicates(inplace=True)
    df.dropna(how='all', inplace=True)
    df.replace(["", "N/A", "NA", "null"], np.nan, inplace=True)
    # Process 'score' column
    if 'score' in df.columns:
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        df.loc[(df['score'] < 1) | (df['score'] > 5), 'score'] = np.nan
        if df['score'].notna().sum() > 0:
            df['score'].fillna(df['score'].median(), inplace=True)
        else:
            df['score'] = 3.0
    # Fill numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != 'score':
            df[col].fillna(df[col].mean(), inplace=True)
    # Fill string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    # Remove emojis/non-standard chars
    def clean_text(text):
        if isinstance(text, str):
            text = re.sub(r'[^\w\s:]', '', text)
            emoji_pattern = re.compile(
                "[" 
                "\U0001F600-\U0001F64F"
                "\U0001F300-\U0001F5FF"
                "\U0001F680-\U0001F6FF"
                "\U0001F700-\U0001F77F"
                "\U0001F780-\U0001F7FF"
                "\U0001F800-\U0001F8FF"
                "\U0001F900-\U0001F9FF"
                "\U0001FA00-\U0001FA6F"
                "\U0001FA70-\U0001FAFF"
                "\U00002702-\U000027B0"
                "\U000024C2-\U0001F251"
                "]+"
            )
            return emoji_pattern.sub(r'', text)
        return text

    if 'content' in df.columns:
        df['content'] = df['content'].apply(clean_text)
    if 'replyContent' in df.columns:
        df['replyContent'] = df['replyContent'].apply(clean_text)

    return df

# --- Load sentiment pipeline ---
@st.cache_resource
def load_sentiment_pipeline():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, truncation=True, max_length=512)

def safe_sentiment(text, sentiment_pipeline):
    if not isinstance(text, str) or not text.strip():
        return "neutral"
    try:
        result = sentiment_pipeline(text)[0]
        return result['label'].lower()
    except:
        return "neutral"

def batch_sentiment_analysis(texts, sentiment_pipeline, batch_size=32):
    results = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        progress = (i + len(batch)) / total * 100
        eta = (total - (i + len(batch))) / batch_size * 0.5  # rough estimate
        st.write(f"Processing batch {i//batch_size+1} / {(total//batch_size)+1}... Progress: {progress:.2f}% | ETA: {eta:.1f} min")
        try:
            batch_results = sentiment_pipeline(batch)
            results.extend(batch_results)
        except:
            results.extend([{"label": "neutral"}] * len(batch))
    return results

# --- Clustering ---
@st.cache_data
def cluster_reviews(df, n_clusters=5):
    if 'content' not in df.columns:
        return df, {}
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = df['content'].fillna("").tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(embeddings)
    cluster_keywords = {}
    for c in range(n_clusters):
        cluster_texts = df[df['cluster'] == c]['content'].tolist()
        if cluster_texts:
            try:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
                vectorizer.fit_transform(cluster_texts)
                cluster_keywords[c] = vectorizer.get_feature_names_out().tolist()
            except:
                cluster_keywords[c] = []
    return df, cluster_keywords

# --- Ask AI ---
def ask_ai_analyst(question, df, cluster_keywords, groq_client):
    if not groq_client:
        return "⚠️ API client not initialized."
    analysis = {
        "total_reviews": len(df),
        "sentiment_counts": df['sentiment'].value_counts().to_dict() if 'sentiment' in df.columns else {},
        "clusters": []
    }
    if 'cluster' in df.columns:
        total_clusters = len(df['cluster'].unique())
        for c in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == c]
            analysis["clusters"].append({
                "cluster_id": int(c),
                "size": len(cluster_df),
                "percentage": round(len(cluster_df)/len(df)*100, 1),
                "top_keywords": list(cluster_keywords.get(c, [])),
                "sentiment_breakdown": cluster_df['sentiment'].value_counts().to_dict() if 'sentiment' in df.columns else {},
                "sample_reviews": cluster_df['content'].sample(min(2,len(cluster_df))).tolist()
            })
    system_instruction = """You are a business analyst specializing in customer feedback analysis.
Use the data provided to identify patterns, root causes, and give concise, data-backed insights and recommendations."""
    user_prompt = f"""Data Summary:\n{json.dumps(analysis, indent=2)}\n\nQuestion: {question}\n\nProvide insights and recommendations."""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error during AI call: {e}"

# =========================
# Main App
# =========================
st.title("🚗 Uber powered by Gen AI Dashboard")
st.markdown("**Analyze customer reviews with AI-powered sentiment analysis and clustering**")

# Sidebar setup
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("🔑 Groq API Key")
    with st.expander("📖 How to get your FREE Groq API key", expanded=False):
        st.markdown("""
        **Step 1:** Visit [console.groq.com](https://console.groq.com)  
        **Step 2:** Sign up (free)  
        **Step 3:** Create API key  
        **Step 4:** Copy and paste below  
        ⚡ Free tier: 14,400 requests/day, fast inference, no credit card
        """)
    groq_api_key = st.text_input("Paste your Groq API Key:", type="password", placeholder="gsk_...", help="Your API key is private")
    groq_client = None
    if groq_api_key:
        if groq_api_key.startswith("gsk_"):
            try:
                st.write("Initializing Groq client...")
                groq_client = Groq(api_key=groq_api_key)
                st.write("Groq client initialized.")
                st.success("✅ API key validated!")
            except Exception as e:
                st.write(f"Error initializing Groq: {e}")
                st.error("❌ Invalid API key.")
        else:
            st.write("Invalid API key format.")
            st.warning("⚠️ API key should start with 'gsk_'")
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown("""
    **Built with:**  
    - 🤖 Sentiment Analysis (RoBERTa)  
    - 🎯 KMeans Clustering  
    - 💬 AI Assistant (Groq LLM)
    """)

# File uploader
uploaded_file = st.file_uploader("📁 Upload your Uber review CSV", type=["csv"], help="CSV should have a 'content' column with review text")
if uploaded_file:
    with st.spinner("Loading and cleaning data..."):
        df = load_and_clean_data(uploaded_file)
    st.success(f"✅ Loaded {len(df):,} reviews")
    if 'content' not in df.columns:
        st.error("❌ CSV must have a 'content' column.")
        st.stop()

    with st.expander("📊 Data Preview"):
        st.dataframe(df.head(), use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🎯 Sentiment Analysis", "🔍 Clustering", "💬 AI Assistant"])

    # Overview
    with tab1:
        st.header("📊 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", f"{len(df):,}")
        if 'score' in df.columns:
            try:
                avg_score = df['score'].mean()
                col2.metric("Avg Rating", f"{avg_score:.2f}/5")
            except:
                col2.metric("Avg Rating", "N/A")
        else:
            col2.metric("Avg Rating", "N/A")
        col3.metric("Data Points", f"{df.shape[1]}")

    # Sentiment Analysis
    with tab2:
        if 'sentiment' not in df.columns:
            with st.spinner("Running sentiment analysis... This may take a few minutes."):
                sentiment_pipeline = load_sentiment_pipeline()
                df['content'] = df['content'].astype(str)
                results = batch_sentiment_analysis(df['content'].tolist(), sentiment_pipeline, 32)
                df['sentiment'] = [res['label'].lower() for res in results]
                st.success("✅ Sentiment analysis complete!")

    # Clustering
    with tab3:
        if st.button("🔄 Re-run Clustering") or 'cluster' not in df.columns:
            with st.spinner(f"Clustering into {st.session_state.get('n_clusters', 5)} groups..."):
                df, cluster_keywords = cluster_reviews(df, st.session_state.get('n_clusters', 5))
            st.success("✅ Clustering complete!")
        else:
            # Use existing clusters
            cluster_keywords = {}
            for c in df['cluster'].unique():
                texts = df[df['cluster'] == c]['content'].tolist()
                if texts:
                    try:
                        vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
                        vectorizer.fit_transform(texts)
                        cluster_keywords[c] = vectorizer.get_feature_names_out().tolist()
                    except:
                        cluster_keywords[c] = []

    # AI Assistant
    with tab4:
        if not groq_client:
            st.warning("⚠️ Add your Groq API key in the sidebar.")
        else:
            question = st.text_input("Ask a question about reviews:", placeholder="e.g., What patterns do you see?")
            if question:
                with st.spinner("Analyzing data and generating insights..."):
                    # Prepare cluster_keywords if clustering was run
                    if 'cluster' in df.columns:
                        cluster_keywords = {}
                        for c in df['cluster'].unique():
                            texts = df[df['cluster'] == c]['content'].tolist()
                            if texts:
                                try:
                                    vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
                                    vectorizer.fit_transform(texts)
                                    cluster_keywords[c] = vectorizer.get_feature_names_out().tolist()
                                except:
                                    cluster_keywords[c] = []
                    answer = ask_ai_analyst(question, df, cluster_keywords, groq_client)
                    st.subheader("💡 AI Response")
                    st.markdown(answer)

else:
    st.info("👆 Please upload a CSV file to begin analysis.")

# Footer
st.markdown(
    """
    <div style='text-align: center; color: #0d0c0c;'>
        <p>• Powered by AI</p>
    </div>
    """,
    unsafe_allow_html=True
)