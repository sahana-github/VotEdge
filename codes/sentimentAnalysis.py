import pandas as pd
from textblob import TextBlob

# Load the CSV
news_df = pd.read_csv("data/raw/news.csv")

# Make sure the column with news titles is named 'text'
# Apply sentiment analysis
def get_sentiment(text):
    analysis = TextBlob(str(text))
    if analysis.polarity > 0:
        return "Positive"
    elif analysis.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

news_df["sentiment"] = news_df["text"].apply(get_sentiment)

# Save to a new CSV
news_df.to_csv("data/raw/news_sentiment.csv", index=False)
print(news_df.head())
print("✅ Sentiment analysis done. Saved to data/raw/news_sentiment.csv")
