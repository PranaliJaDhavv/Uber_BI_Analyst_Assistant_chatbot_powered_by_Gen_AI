import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq
import json
import re
from datetime import datetime

# =========================
# Inject Custom CSS for Uber-like UI and Button Styling
# =========================
st.markdown(
    """
    <!-- Import Uber-like font (Inter) from Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">

    <style>
    /* Set overall background and font */
    body {
        background-color: #ffffff; /* Uber's clean white background */
        font-family: 'Inter', sans-serif;
        color: #222; /* Dark text color for readability */
        line-height: 1.6;
    }

    /* Headers styles */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #222;
        font-weight: 600;
    }

    /* Main container blocks */
    div[data-testid="stVerticalBlock"], div[data-testid="stHorizontalBlock"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f5f5f7; /* Light grey sidebar similar to Uber app */
        border-radius: 8px;
        padding: 10px;
        font-family: 'Inter', sans-serif;
    }

    /* All buttons default style to black background and white font */
    button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        background-color: #000 !important;
        color: #fff !important;
        border: none;
        padding: 10px 20px;
        border-radius: 4px;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }
    button:hover {
        background-color: #333 !important;
    }

    /* Title section with black background and white text */
    .title-section {
        background-color: #000;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }

    /* Style markdown text inside markdown containers */
    .markdown-text-container {
        font-family: 'Inter', sans-serif;
        color: #222;
    }

    /* Headers inside markdown */
    .markdown-text-container h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #222;
        line-height: 1.2;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Helper Functions
# =========================

@st.cache_resource
def load_data():
    with open('preprocessed_reviews_2025_full.pkl', 'rb') as f:
        return pickle.load(f)

def validate_groq_api(api_key):
    try:
        client = Groq(api_key=api_key)
        return client
    except:
        return None

# ---------- DATE FILTER ----------
def filter_data_by_date(df, question):
    df_temp = df.copy()
    if 'at' not in df_temp.columns:
        return df_temp
    df_temp['at'] = pd.to_datetime(df_temp['at'], errors='coerce')
    df_temp.dropna(subset=['at'], inplace=True)

    month_map = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
    }

    q = question.lower()
    selected_month = None
    selected_year = None

    for m_name, m_num in month_map.items():
        if m_name in q:
            selected_month = m_num
            break

    year_match = re.search(r"(20\d{2})", q)
    if year_match:
        selected_year = int(year_match.group(1))

    if selected_month and selected_year:
        df_temp = df_temp[(df_temp['at'].dt.month == selected_month) & (df_temp['at'].dt.year == selected_year)]
    elif selected_month:
        df_temp = df_temp[df_temp['at'].dt.month == selected_month]
    elif selected_year:
        df_temp = df_temp[df_temp['at'].dt.year == selected_year]

    if 'by month' in q or 'monthly' in q:
        df_temp['month'] = df_temp['at'].dt.to_period('M').astype(str)
        return df_temp

    return df_temp

def detect_visualization_request(question):
    q = question.lower()
    if "sentiment" in q and ("distribution" in q or "chart" in q or "breakdown" in q):
        return "sentiment_distribution"
    elif "cluster" in q and ("topics" in q or "keywords" in q):
        return "cluster_keywords"
    elif "trend" in q or "over time" in q or "time series" in q or "timeline" in q or "timeseries" in q:
        return "trend_over_time"
    elif "rating" in q and ("distribution" in q or "breakdown" in q):
        return "rating_distribution"
    elif "sentiment" in q and ("time" in q or "trend" in q):
        return "sentiment_over_time"
    elif "heatmap" in q or "correlation" in q:
        return "correlation_heatmap"
    elif "distribution" in q and "month" in q:
        return "monthly_distribution"
    else:
        return None

# =========================
# Visualization Functions (not shown here for brevity, assume same as before)
# =========================

# =========================
# AI Analyst Function (same as before)
# =========================

# =========================
# Main App
# =========================

st.set_page_config(page_title="Uber Business Analyst Assistant", layout="wide", page_icon="🚗")

try:
    st.image("uber.png", width=100)
except:
    st.markdown("🚗")

# Title with black background
st.markdown(
    """
    <div class='title-section'>
        <h1 style='text-align: center; color: #fff;'>🤖 Uber Business Analyst Assistant Powered by Gen AI</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align: center; color: #666;'>Sentiment • Trends • KPI Dashboard • Visualization</p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

try:
    df = load_data()
    st.sidebar.success(f"✅ Loaded {len(df):,} reviews")
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.stop()

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("🔑 Groq API Key", type="password", placeholder="gsk_...")
groq_client = validate_groq_api(api_key) if api_key else None
if groq_client:
    st.sidebar.success("✅ API key validated!")
else:
    st.sidebar.info("💡 Enter API key to use AI Assistant")

# Dataset metadata
st.sidebar.header("📌 Dataset Metadata")
st.sidebar.write(f"**Total Reviews:** {len(df):,}")
if 'at' in df.columns:
    df['at'] = pd.to_datetime(df['at'], errors='coerce')
    st.sidebar.write(f"**Date Range:** {df['at'].min().date()} to {df['at'].max().date()}")
if 'score' in df.columns:
    avg_score = pd.to_numeric(df['score'], errors='coerce').mean()
    st.sidebar.write(f"**Average Rating:** {avg_score:.2f}/5")
if 'cluster' in df.columns:
    st.sidebar.write(f"**Total Clusters:** {df['cluster'].nunique()}")

# Main layout
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("💬 Ask Questions About Your Data")
    suggested_questions = [
        "Show me sentiment distribution",
        "Show me distribution by month",
        "Trend in June 2025",
        "Rating breakdown for March",
        "Cluster keywords",
        "Correlation heatmap",
        "Top clusters with negative reviews",
        "Average rating last 3 months",
        "Insights on driver issues",
        "Deeper business insights and recommendations"
    ]
    selected = st.selectbox("Quick Questions:", [""] + suggested_questions)
    user_question = st.text_area("Or type your own question:", value=selected if selected else "")

    if st.button("Ask AI"):  # No emoji, CSS will style it
        if not user_question:
            st.warning("⚠️ Please enter a question")
        else:
            with st.spinner("🤔 Analyzing data..."):
                df_filtered = filter_data_by_date(df, user_question)
                viz_type = detect_visualization_request(user_question)

                if viz_type == "sentiment_distribution":
                    st.info("📊 Generating sentiment distribution chart...")
                    generate_sentiment_distribution(df_filtered)
                elif viz_type == "cluster_keywords":
                    st.info("🎯 Generating cluster keywords...")
                    generate_cluster_keywords(df_filtered)
                elif viz_type == "rating_distribution":
                    st.info("⭐ Generating rating distribution chart...")
                    generate_rating_distribution(df_filtered)
                elif viz_type in ["sentiment_over_time", "trend_over_time"]:
                    st.info("📈 Generating time series chart...")
                    generate_timeseries(df_filtered)
                elif viz_type == "correlation_heatmap":
                    st.info("🔥 Generating correlation heatmap...")
                    generate_correlation_heatmap(df_filtered)
                elif viz_type == "monthly_distribution":
                    st.info("🗓️ Generating monthly sentiment distribution...")
                    generate_monthly_distribution(df_filtered)
                else:
                    answer = ask_ai_analyst(user_question, df_filtered, groq_client)
                    st.markdown("### 🤖 AI Insight & Recommendations")
                    st.markdown(answer)

                if len(df_filtered) != len(df):
                    st.success(f"📅 Filtered period: {len(df_filtered):,} reviews shown.")

with col2:
    st.subheader("📊 Dashboard KPIs")
    total_reviews = len(df)
    sentiment_counts = df['sentiment'].value_counts()
    avg = pd.to_numeric(df['score'], errors='coerce').mean()
    neg_pct = sentiment_counts.get('negative', 0) / total_reviews * 100

    total_kpi, avg_rating_kpi, neg_kpi, pos_kpi = st.columns(4)

    total_kpi.markdown(f"<div style='font-size:14px'>📝 Total Reviews<br><b>{total_reviews:,}</b></div>", unsafe_allow_html=True)
    avg_rating_kpi.markdown(f"<div style='font-size:14px'>⭐ Avg Rating<br><b>{avg:.2f}/5</b></div>", unsafe_allow_html=True)
    neg_kpi.markdown(f"<div style='font-size:14px'>❌ Negative %<br><b>{neg_pct:.1f}%</b></div>", unsafe_allow_html=True)
    pos_kpi.markdown(f"<div style='font-size:14px'>🟢 Positive %<br><b>{sentiment_counts.get('positive',0)/total_reviews*100:.1f}%</b></div>", unsafe_allow_html=True)

    # Sentiment breakdown horizontal bar chart
    st.markdown("### 🎭 Sentiment Breakdown")
    fig, ax = plt.subplots(figsize=(2, 1))
    sns.barplot(
        x=sentiment_counts.values,
        y=sentiment_counts.index,
        palette=['#10b981', '#ef4444', '#f59e0b'],
        ax=ax
    )
    ax.set_xlabel("Number of Reviews", fontsize=10)
    ax.set_ylabel("")
    for i, v in enumerate(sentiment_counts.values):
        ax.text(v + max(sentiment_counts.values)*0.01, i, str(v), va='center', fontweight='bold', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

# Footer image
st.markdown("---")
st.image("uber_passengers.jpg", use_container_width=True)
st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;'>Built with ❤️ by Pranali J</div>", unsafe_allow_html=True)