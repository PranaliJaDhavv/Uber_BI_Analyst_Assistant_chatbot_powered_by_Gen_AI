import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pickle

# Load your data
df = pd.read_csv('uber_reviews.csv', encoding='latin1')  # adjust path and encoding

# Initialize tokenizer and model
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Setup sentiment pipeline with GPU if available
device = 0 if torch.cuda.is_available() else -1
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
    max_length=512,
    device=device
)

# Prepare texts
texts = df['content'].astype(str).tolist()

# Set batch size
batch_size = 64

results = []
total_texts = len(texts)

# Process in batches with progress logging
for i in range(0, total_texts, batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_results = sentiment_pipeline(batch_texts, batch_size=batch_size)
    results.extend(batch_results)
    print(f"Processed {i + len(batch_texts)} / {total_texts} reviews...")

# Add sentiment results to DataFrame
df['sentiment'] = [res['label'].lower() for res in results]

# Save the DataFrame with sentiment to a pickle file
with open('preprocessed_reviews.pkl', 'wb') as f:
    pickle.dump(df, f)

print("Sentiment analysis complete. Results saved to 'preprocessed_reviews.pkl'.")