# ========================================
# PROCESS 146K UBER REVIEWS - FROM GOOGLE DRIVE LINK
# ========================================

!pip install gdown transformers torch scikit-learn pandas -q

import pandas as pd
from transformers import pipeline
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from google.colab import files
import time
import gdown

start_time = time.time()

# Download from Google Drive
print("📥 Downloading CSV from Google Drive...")
file_id = '1dk8jH60u1dgBwpL868bm_mm-SkutyYoN'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'uber_reviews_2025.csv'
gdown.download(url, output, quiet=False)

print("\n📊 Loading reviews...")
df = pd.read_csv(output)

print(f"✅ Loaded {len(df):,} reviews")
print(f"📅 Date range: {df['at'].min()} to {df['at'].max()}")

# Clean data
print("\n🧹 Cleaning data...")
initial_count = len(df)
df = df.dropna(subset=['content'])
df['content'] = df['content'].astype(str)
df = df[df['content'].str.len() > 10]
df = df.drop_duplicates(subset=['content'])

print(f"   Removed {initial_count - len(df):,} invalid/duplicate reviews")
print(f"✅ Clean dataset: {len(df):,} reviews")

# Sentiment Analysis
print("\n" + "="*70)
print("😊 SENTIMENT ANALYSIS STARTING")
print("="*70)
print("⏱️  Estimated time: 20-30 minutes for 146k reviews")
print("🔥 Using GPU" if torch.cuda.is_available() else "💻 Using CPU")
print()

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1
)

def analyze_sentiments(texts, batch_size=100):
    sentiments = []
    total = len(texts)
    start = time.time()
    
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        results = sentiment_analyzer(batch, truncation=True, max_length=256)
        sentiments.extend(results)
        
        if (i + batch_size) % 5000 == 0 or i + batch_size >= total:
            progress = min(i + batch_size, total)
            elapsed = time.time() - start
            reviews_per_sec = progress / elapsed
            remaining = (total - progress) / reviews_per_sec if reviews_per_sec > 0 else 0
            
            print(f"   {progress:,}/{total:,} ({progress/total*100:.1f}%) | "
                  f"Speed: {reviews_per_sec:.0f} reviews/sec | "
                  f"ETA: {remaining/60:.1f} min")
    
    return sentiments

sentiments = analyze_sentiments(df['content'].tolist())

# Map sentiment
sentiment_map = {'POSITIVE': 'positive', 'NEGATIVE': 'negative', 'NEUTRAL': 'neutral'}
df['sentiment'] = [sentiment_map.get(s['label'], s['label'].lower()) for s in sentiments]
df['sentiment_score'] = [s['score'] for s in sentiments]

print(f"\n✅ Sentiment Analysis Complete!")
print(f"\n📊 Sentiment Distribution:")
for sentiment, count in df['sentiment'].value_counts().items():
    pct = count / len(df) * 100
    print(f"   {sentiment.title()}: {count:,} ({pct:.1f}%)")

# Clustering
print("\n" + "="*70)
print("🎯 TOPIC CLUSTERING STARTING")
print("="*70)

print("   Creating TF-IDF vectors...")
vectorizer = TfidfVectorizer(max_features=100, stop_words='english', min_df=20, max_df=0.7)
tfidf_matrix = vectorizer.fit_transform(df['content'])
print(f"   ✅ Created {tfidf_matrix.shape[0]:,} × {tfidf_matrix.shape[1]} matrix")

print("   Running MiniBatch K-Means...")
kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=5000, n_init=3, max_iter=100, verbose=1)
df['cluster'] = kmeans.fit_predict(tfidf_matrix)

print(f"\n✅ Clustering Complete!")
print(f"\n📊 Cluster Distribution:")
for cluster_id, count in df['cluster'].value_counts().sort_index().items():
    pct = count / len(df) * 100
    print(f"   Cluster {cluster_id}: {count:,} ({pct:.1f}%)")

# Show keywords
print(f"\n🔑 Top 10 Keywords per Cluster:")
feature_names = vectorizer.get_feature_names_out()
for i in range(5):
    center = kmeans.cluster_centers_[i]
    top_idx = center.argsort()[-10:][::-1]
    keywords = [feature_names[idx] for idx in top_idx]
    print(f"\n   Cluster {i}: {', '.join(keywords)}")
    cluster_sentiments = df[df['cluster'] == i]['sentiment'].value_counts()
    sentiment_str = ', '.join([f"{s}: {c}" for s, c in cluster_sentiments.items()])
    print(f"   Sentiments: {sentiment_str}")

# Save
output_file = 'preprocessed_reviews_2025_full.pkl'
print(f"\n💾 Saving to {output_file}...")
df.to_pickle(output_file)

import os
size_mb = os.path.getsize(output_file) / (1024 * 1024)
elapsed = time.time() - start_time

pos_count = (df['sentiment']=='positive').sum()
neg_count = (df['sentiment']=='negative').sum()
neu_count = (df['sentiment']=='neutral').sum()

print(f"\n" + "="*70)
print("✅ PROCESSING COMPLETE!")
print("="*70)
print(f"📦 Total Reviews: {len(df):,}")
print(f"📅 Date Range: {df['at'].min()} to {df['at'].max()}")
print(f"⏱️  Processing Time: {elapsed/60:.1f} minutes")
print(f"💾 File Size: {size_mb:.2f} MB")
print(f"\n📊 Final Statistics:")
print(f"   Positive: {pos_count:,} ({pos_count/len(df)*100:.1f}%)")
print(f"   Negative: {neg_count:,} ({neg_count/len(df)*100:.1f}%)")
print(f"   Neutral: {neu_count:,} ({neu_count/len(df)*100:.1f}%)")

if size_mb > 100:
    print(f"\n⚠️  File is {size_mb:.2f}MB - You'll need Git LFS")
else:
    print(f"\n✅ File size is good for GitHub!")

print("="*70)

print(f"\n⬇️  Downloading pickle file...")
files.download(output_file)
print("✅ Download complete! Check your downloads folder.")
print("\n🎉 All done! Replace your old pickle file and push to GitHub!")