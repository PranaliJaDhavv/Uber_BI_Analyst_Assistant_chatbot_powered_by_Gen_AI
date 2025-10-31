import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq
import json

# =========================
# Helper Functions
# =========================

@st.cache_resource
def load_data():
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
    elif "trend" in q or "over time" in q or "time series" in q or "timeline" in q or "timeseries" in q:
        return "trend_over_time"
    elif "rating" in q and ("distribution" in q or "breakdown" in q):
        return "rating_distribution"
    elif "sentiment" in q and ("time" in q or "trend" in q):
        return "sentiment_over_time"
    elif "heatmap" in q or "correlation" in q:
        return "correlation_heatmap"
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
    fig, ax = plt.subplots(figsize=(10, 6))
    rating_counts = ratings.value_counts().sort_index()
    colors = ['#ef4444', '#f97316', '#f59e0b', '#84cc16', '#10b981']
    bar_colors = [colors[int(r)-1] if r <= 5 else '#6b7280' for r in rating_counts.index]
    sns.barplot(x=rating_counts.index, y=rating_counts.values, palette=bar_colors, ax=ax)
    ax.set_title("Rating Distribution", fontsize=16, fontweight='bold')
    ax.set_xlabel("Rating (Stars)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    for i, v in enumerate(rating_counts.values):
        ax.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
    st.pyplot(fig)

def generate_timeseries(df):
    if 'at' not in df.columns:
        st.warning("Date/time data not available.")
        return
    df_temp = df.copy()
    try:
        df_temp['at'] = pd.to_datetime(df_temp['at'], unit='ms', errors='coerce')
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
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_counts.index, daily_counts.values, marker='o', linewidth=2, color='#3b82f6')
    ax.fill_between(daily_counts.index, daily_counts.values, alpha=0.3, color='#3b82f6')
    ax.set_title("Reviews Over Time", fontsize=16, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of Reviews", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def generate_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric data for correlation heatmap.")
        return
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, square=True, linewidths=1, ax=ax, fmt='.2f')
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight='bold')
    plt.tight_layout()
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
            df_temp['at'] = pd.to_datetime(df_temp['at'], unit='ms', errors='coerce')
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
    system_instruction = """You are an expert Business Analyst specializing in customer feedback analysis for Uber.

You have access to comprehensive analysis of Uber ride reviews including sentiment analysis and clustering results.

Your responsibilities:
- Provide data-driven insights with specific numbers and percentages
- Identify patterns, trends, and root causes
- Offer actionable business recommendations
- Prioritize issues by business impact
- Be concise yet thorough

Always reference specific data points from the analysis."""

    user_prompt = f"""Here is the complete analysis of Uber reviews:

{json.dumps(analysis_summary, indent=2)}

User Question: {question}

Provide a clear, data-driven answer with specific insights and actionable recommendations."""

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
        return f"❌ Error calling Groq API: {str(e)}\n\nPlease check:\n- Your API key is valid\n- You have internet connection\n- Groq service is available"

# =========================
# Main App
# =========================

st.set_page_config(
    page_title="Uber Business Analyst Assistant Powered by Gen AI",
    layout="wide",
    page_icon="🚗"
)

# Logo & Title
try:
    st.image("uber.png", width=100)
except:
    st.markdown("🚗")

st.markdown("""
<h1 style='text-align: center; color: #000000;'>
    🤖 Uber Business Analyst Assistant Powered by Gen AI
</h1>
<p style='text-align: center; color: #666; font-size: 18px;'>
    Powered by AI • Sentiment Analysis • Trends & Visualization
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Load Data
try:
    df = load_data()
    st.sidebar.success(f"✅ Loaded {len(df):,} reviews")
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.info("💡 Make sure 'preprocessed_reviews_.pkl' file is in the same directory")
    st.stop()

# Sidebar - API Key
st.sidebar.header("⚙️ Configuration")

with st.sidebar.expander("📖 How to get Groq API key", expanded=False):
    st.markdown("""
    **Step 1:** Visit [console.groq.com](https://console.groq.com)
    
    **Step 2:** Sign up (free!)
    
    **Step 3:** Create an API key
    
    **Step 4:** Paste it below
    
    ⚡ Free tier: 14,400 requests/day
    """)

api_key = st.sidebar.text_input(
    "🔑 Enter your Groq API Key",
    type="password",
    placeholder="gsk_..."
)

groq_client = None
if api_key:
    if api_key.startswith("gsk_"):
        groq_client = validate_groq_api(api_key)
        if groq_client:
            st.sidebar.success("✅ API key validated!")
        else:
            st.sidebar.error("❌ Invalid API key")
    else:
        st.sidebar.warning("⚠️ API key should start with 'gsk_'")
else:
    st.sidebar.info("💡 Enter API key to use AI Assistant")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Info")
st.sidebar.write(f"**Total Reviews:** {len(df):,}")

# Fix date column
if 'at' in df.columns:
    try:
        df['at'] = pd.to_datetime(df['at'], unit='ms', errors='coerce')
        st.sidebar.write(f"**Date Range:** {df['at'].min().date()} to {df['at'].max().date()}")
    except:
        try:
            df['at'] = pd.to_datetime(df['at'], errors='coerce')
            st.sidebar.write(f"**Date Range:** {df['at'].min().date()} to {df['at'].max().date()}")
        except:
            st.sidebar.write(f"**Date Range:** Available")

# Fix score column
if 'score' in df.columns:
    df['score'] = pd.to_numeric(df['score'], errors='coerce')

st.sidebar.write(f"**Columns:** {', '.join(df.columns.tolist())}")

# Main layout
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("💬 Ask Questions About Your Data")
    
    st.markdown("**💡 Try these:**")
    suggested_questions = [
        "What are the main customer complaints?",
        "Show me sentiment distribution",
        "What are the cluster keywords?",
        "Give me 3 actionable recommendations",
        "Show sentiment trends over time",
        "Display rating distribution",
        "Show review volume over time",
        "Generate a correlation heatmap"
    ]
    
    selected_suggestion = st.selectbox(
        "Quick questions:",
        [""] + suggested_questions,
        format_func=lambda x: "Select a question..." if x == "" else x
    )
    
    user_question = st.text_area(
        "Or type your own question:",
        value=selected_suggestion if selected_suggestion else "",
        placeholder="e.g., What patterns do you see in negative reviews?",
        height=100
    )

    if st.button("🚀 Ask AI", type="primary"):
        if not user_question:
            st.warning("⚠️ Please enter a question")
        elif not groq_client:
            st.error("❌ Please enter your Groq API key in the sidebar")
        else:
            with st.spinner("🤔 Analyzing data..."):
                viz_type = detect_visualization_request(user_question)
                
                if viz_type == "sentiment_distribution":
                    st.info("📊 Generating sentiment distribution chart...")
                    generate_sentiment_distribution(df)
                    
                elif viz_type == "cluster_keywords":
                    st.info("🎯 Generating cluster keywords...")
                    generate_cluster_keywords(df)
                    
                elif viz_type == "rating_distribution":
                    st.info("⭐ Generating rating distribution chart...")
                    generate_rating_distribution(df)
                    
                elif viz_type == "sentiment_over_time" or viz_type == "trend_over_time":
                    st.info("📈 Generating time series chart...")
                    generate_timeseries(df)
                    
                elif viz_type == "correlation_heatmap":
                    st.info("🔥 Generating correlation heatmap...")
                    generate_correlation_heatmap(df)
                    
                else:
                    answer = ask_ai_analyst(user_question, df, groq_client)
                    st.markdown("### 🤖 AI Response")
                    st.markdown(answer)

with col2:
    st.subheader("📊 Dashboard Overview")
    
    total_reviews = len(df)
    sentiment_counts = df['sentiment'].value_counts()
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric("📝 Total Reviews", f"{total_reviews:,}")
    
    with metric_col2:
        if 'score' in df.columns:
            score_numeric = pd.to_numeric(df['score'], errors='coerce')
            avg_rating = score_numeric.mean()
            if pd.notna(avg_rating):
                st.metric("⭐ Avg Rating", f"{avg_rating:.2f}/5")
            else:
                st.metric("⭐ Avg Rating", "N/A")
        else:
            st.metric("⭐ Avg Rating", "N/A")
    
    with metric_col3:
        negative_pct = (sentiment_counts.get('negative', 0) / total_reviews * 100)
        st.metric("❌ Negative %", f"{negative_pct:.1f}%")
    
    # Sentiment breakdown pie chart
    st.markdown("### 🎭 Sentiment Breakdown")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#10b981', '#ef4444', '#f59e0b']
    ax.pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors
    )
    ax.axis('equal')
    st.pyplot(fig)
    
    # Sentiment stats
    for sentiment, count in sentiment_counts.items():
        percentage = count / total_reviews * 100
        st.progress(percentage / 100, text=f"{sentiment.title()}: {count:,} ({percentage:.1f}%)")
    
    # Sample reviews
    st.markdown("### 📄 Sample Reviews")
    
    if 'content' in df.columns:
        sample_df = df[['content', 'sentiment']].sample(min(5, len(df)))
        
        for idx, row in sample_df.iterrows():
            sentiment_emoji = {
                'positive': '✅',
                'negative': '❌',
                'neutral': '➖'
            }.get(row['sentiment'], '•')
            
            with st.expander(f"{sentiment_emoji} {row['sentiment'].title()} Review"):
                st.write(row['content'])

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Built with ❤️ using Streamlit • Powered by Groq AI</p>
    <p style='font-size: 12px;'>Made by <strong>Pranali J</strong> (pranalipjadhav2024@gmail.com) — my personal project</p>
</div>
""", unsafe_allow_html=True)