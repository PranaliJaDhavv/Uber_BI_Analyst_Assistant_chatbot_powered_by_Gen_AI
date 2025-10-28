import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq
import json
# Uncomment the following line if you want to add WordCloud visualization later
# from wordcloud import WordCloud

# =========================
# Helper Functions
# =========================

@st.cache_resource
def load_data():
    # Replace 'preprocessed_reviews_.pkl' with your actual data file path
    with open('preprocessed_reviews_.pkl', 'rb') as f:
        return pickle.load(f)

def validate_groq_api(api_key):
    try:
        client = Groq(api_key=api_key)
        return client
    except:
        return None

def detect_visualization_request(question):
    q = question.lower()
    if "sentiment" in q and ("distribution" in q or "chart" in q or "breakdown" in q):
        return "sentiment_distribution"
    elif "cluster" in q and ("topics" in q or "keywords" in q):
        return "cluster_keywords"
    elif "trend" in q or "over time" in q or "time series" in q or "timeline" in q:
        return "trend_over_time"
    elif "rating" in q and ("distribution" in q or "breakdown" in q):
        return "rating_distribution"
    elif "sentiment" in q and ("time" in q or "trend" in q):
        return "sentiment_over_time"
    elif "heatmap" in q or "correlation" in q:
        return "correlation_heatmap"
    elif "timeseries" in q or "over time" in q:
        return "timeseries"
    else:
        return None

# =========================
# Visualization Functions
# =========================

def generate_sentiment_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sentiment_counts = df['sentiment'].value_counts()
    colors = {'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#f59e0b'}
    bar_colors = [colors.get(s, '#6b7280') for s in sentiment_counts.index]
    sns.barplot(x=sentiment_counts.index,
                y=sentiment_counts.values,
                palette=bar_colors,
                ax=ax)
    ax.set_title("Sentiment Distribution", fontsize=16, fontweight='bold')
    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    for i, v in enumerate(sentiment_counts.values):
        ax.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
    st.pyplot(fig)

def generate_cluster_keywords(df):
    if 'cluster' not in df.columns:
        st.warning("⚠️ Cluster information not available in the dataset.")
        return
    clusters = sorted(df['cluster'].unique())
    st.markdown("### 🎯 Cluster Keywords")
    for c in clusters:
        texts = df[df['cluster'] == c]['content'].tolist()
        cluster_size = len(texts)
        if texts:
            try:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
                vectorizer.fit_transform(texts)
                keywords = vectorizer.get_feature_names_out().tolist()
                with st.expander(f"**Cluster {c}** ({cluster_size} reviews)"):
                    st.write(f"**Keywords:** {', '.join(keywords)}")
                    sentiment_counts = df[df['cluster'] == c]['sentiment'].value_counts()
                    st.write(f"**Sentiment:** {sentiment_counts.to_dict()}")
            except Exception as e:
                st.error(f"❌ Error generating keywords for cluster {c}: {str(e)}")

def generate_rating_distribution(df):
    if 'score' not in df.columns:
        st.warning("Rating data not available.")
        return
    ratings = pd.to_numeric(df['score'], errors='coerce').dropna()
    fig, ax = plt.subplots()
    sns.histplot(ratings, bins=5, kde=False, ax=ax)
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Reviews")
    st.pyplot(fig)

def generate_timeseries(df):
    if 'at' not in df.columns:
        st.warning("Date/time data not available.")
        return
    df_temp = df.copy()
    try:
        df_temp['at'] = pd.to_datetime(df_temp['at'], errors='coerce')
    except:
        df_temp['at'] = pd.to_datetime(df_temp['at'], errors='coerce')
    df_temp.dropna(subset=['at'], inplace=True)
    if df_temp.empty:
        st.warning("No valid date data for timeseries.")
        return
    df_temp.set_index('at', inplace=True)
    daily_counts = df_temp.resample('D').size()
    if daily_counts.sum() == 0:
        st.warning("No data after resampling.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily_counts.index, daily_counts.values, marker='o')
    ax.set_title("Reviews Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Reviews")
    st.pyplot(fig)

def generate_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric data for correlation heatmap.")
        return
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)

def ask_ai_analyst(question, df, groq_client):
    if not groq_client:
        return "⚠️ Please enter your Groq API key in the sidebar."
    analysis_summary = {
        "total_reviews": len(df),
        "sentiment_distribution": df['sentiment'].value_counts().to_dict()
    }
    # Add date range
    if 'at' in df.columns:
        try:
            df_temp = df.copy()
            df_temp['at'] = pd.to_datetime(df_temp['at'], errors='coerce')
            analysis_summary["date_range"] = {
                "start": str(df_temp['at'].min().date()),
                "end": str(df_temp['at'].max().date())
            }
        except:
            pass
    # Add rating info
    if 'score' in df.columns:
        try:
            df_temp = df.copy()
            df_temp['score'] = pd.to_numeric(df_temp['score'], errors='coerce')
            analysis_summary["average_rating"] = float(df_temp['score'].mean())
        except:
            pass
    # Add clusters info
    if 'cluster' in df.columns:
        analysis_summary["clusters"] = []
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster_id]
            texts = cluster_data['content'].tolist()
            keywords = []
            if texts:
                try:
                    vectorizer = TfidfVectorizer(stop_words='english', max_features=5)
                    vectorizer.fit_transform(texts)
                    keywords = vectorizer.get_feature_names_out().tolist()
                except:
                    keywords = []
            info = {
                "cluster_id": int(cluster_id),
                "size": len(cluster_data),
                "percentage": round(len(cluster_data)/len(df)*100, 1),
                "keywords": keywords,
                "sentiment_breakdown": cluster_data['sentiment'].value_counts().to_dict(),
                "sample_reviews": cluster_data['content'].head(2).tolist()
            }
            analysis_summary["clusters"].append(info)
    # Compose prompt
    system_instruction = "You are an AI Business Analyst providing insights on Uber reviews."
    user_prompt = f"""Analysis:
{json.dumps(analysis_summary, indent=2)}

Question: {question}

Provide a clear, concise, and data-driven answer."""
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
        return f"❌ Error calling Groq API: {str(e)}"

# =========================
# Main App
# =========================

st.set_page_config(
    page_title="Uber Business Analyst Assitant Powered by Gen AI",
    layout="wide",
    page_icon="🚗"
)

# Logo & Title
try:
    st.image("uber.png", width=100)
except:
    st.markdown("🚗")
st.markdown(
    """
    <h1 style='text-align: center; color: #000000;'>
        🤖 Uber Business Analyst Assitant Powered by Gen A
st.set_page_config(
    page_title="Uber Business Analyst Assitant Powered by Gen AI",
    layout="wide",
    page_icon="🚗"
)

    </h1>
    <p style='text-align: center; font-size: 18px;'>
        Powered by AI • Sentiment Analysis • Trends & Visualization
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Load Data
try:
    df = load_data()
except:
    st.error("❌ Error loading data. Make sure your data file exists.")
    st.stop()

# Show metadata in sidebar
def display_metadata(df):
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.write(f"**Total Reviews:** {len(df):,}")
    if 'at' in df.columns:
        try:
            df['at'] = pd.to_datetime(df['at'], errors='coerce')
            date_range = f"{df['at'].min().date()} to {df['at'].max().date()}"
            st.sidebar.write(f"**Date Range:** {date_range}")
        except:
            pass
    st.sidebar.write(f"**Columns:** {', '.join(df.columns)}")
    st.sidebar.write(f"**Data Source:** Uber Reviews (Kaggle)")

display_metadata(df)

# Sidebar - API Key
st.sidebar.header("⚙️ Configuration")
with st.sidebar.expander("📖 How to get Groq API key", expanded=False):
    st.markdown("""
    1. Visit [console.groq.com](https://console.groq.com)
    2. Sign up (free!)
    3. Generate API key
    4. Paste below
    """)
api_key_input = st.sidebar.text_input("🔑 Enter your Groq API Key", type="password", placeholder="gsk_...")

groq_client = None
if api_key_input:
    if api_key_input.startswith("gsk_"):
        groq_client = validate_groq_api(api_key_input)
        if groq_client:
            st.sidebar.success("✅ API key validated!")
        else:
            st.sidebar.error("❌ Invalid API key")
    else:
        st.sidebar.warning("API key should start with 'gsk_'")
else:
    st.sidebar.info("Enter API key to enable AI features.")

# Main layout with two columns
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("💬 Ask Questions about Data")
    suggestions = [
        "What are the main customer complaints?",
        "Show sentiment distribution",
        "What are the cluster keywords?",
        "Show reviews over time",
        "Show sentiment trend over time",
        "Give me 3 actionable recommendations"
    ]
    selected_q = st.selectbox("Quick questions:", [""] + suggestions)
    user_q = st.text_area("Or type your own question:", value=selected_q if selected_q else "", height=100)

    if st.button("🚀 Ask AI"):
        if not user_q:
            st.warning("⚠️ Please enter a question")
        elif not groq_client:
            st.error("❌ Please set your Groq API key in the sidebar")
        else:
            with st.spinner("🤖 Analyzing..."):
                viz_type = detect_visualization_request(user_q)
                if viz_type == "sentiment_distribution":
                    generate_sentiment_distribution(df)
                elif viz_type == "cluster_keywords":
                    generate_cluster_keywords(df)
                elif viz_type == "trend_over_time" or viz_type == "timeseries":
                    generate_timeseries(df)
                elif viz_type == "sentiment_over_time":
                    generate_timeseries(df)
                elif viz_type == "rating_distribution":
                    generate_rating_distribution(df)
                elif viz_type == "correlation_heatmap":
                    generate_correlation_heatmap(df)
                else:
                    answer = ask_ai_analyst(user_q, df, groq_client)
                    st.markdown(f"### 🤖 AI Response")
                    st.markdown(answer)

with col2:
    st.subheader("📊 Dashboard Overview")
    total_reviews = len(df)
    sentiment_counts = df['sentiment'].value_counts()

    metric1, metric2, metric3 = st.columns(3)
    with metric1:
        st.metric("📝 Total Reviews", f"{total_reviews:,}")
    with metric2:
        if 'score' in df.columns:
            mean_score = pd.to_numeric(df['score'], errors='coerce').mean()
            st.metric("⭐ Avg Rating", f"{mean_score:.2f}/5")
        else:
            st.metric("⭐ Avg Rating", "N/A")
    with metric3:
        neg_pct = sentiment_counts.get('negative', 0) / total_reviews * 100
        st.metric("❌ Negative %", f"{neg_pct:.1f}%")

    # Sentiment Breakdown Pie Chart
    st.markdown("### 🎭 Sentiment Breakdown")
    fig1, ax1 = plt.subplots()
    colors = ['#10b981', '#ef4444', '#f59e0b']
    ax1.pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors
    )
    ax1.axis('equal')
    st.pyplot(fig1)

    # Key insights (you can replace these with more detailed analysis)
    st.markdown("### Key Insights")
    if 'content' in df.columns:
        top_negative = df[df['sentiment']=='negative']['content'].head(1).values
        top_positive = df[df['sentiment']=='positive']['content'].head(1).values
        st.write(f"**Most Negative Review:** {top_negative[0] if top_negative else 'N/A'}")
        st.write(f"**Most Positive Review:** {top_positive[0] if top_positive else 'N/A'}")
    else:
        st.write("No review content available.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 12px; color: #888; padding: 10px;'>
        Made by **Pranali J (pranalipjadhav2024@gmail.com)** — my personal project.
    </div>
    """,
    unsafe_allow_html=True
)
