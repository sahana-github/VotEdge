import pandas as pd
from textblob import TextBlob

def get_sentiment(text):
    """Determine sentiment of text using TextBlob"""
    analysis = TextBlob(str(text))
    if analysis.polarity > 0:
        return "Positive"
    elif analysis.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def perform_sentiment_analysis():
    """Perform sentiment analysis on news data"""
    try:
        # Load the CSV
        news_df = pd.read_csv("data/raw/news.csv")
        
        # Make sure the column with news titles is named 'text'
        # Apply sentiment analysis
        news_df["sentiment"] = news_df["text"].apply(get_sentiment)
        
        # Save to a new CSV
        news_df.to_csv("data/raw/news_sentiment.csv", index=False)
        print(news_df.head())
        print("✅ Sentiment analysis done. Saved to data/raw/news_sentiment.csv")
        return True
    except FileNotFoundError:
        print("❌ Error: data/raw/news.csv not found. Please fetch news data first.")
        return False
    except Exception as e:
        print(f"❌ Error during sentiment analysis: {e}")
        return False

if __name__ == "__main__":
    perform_sentiment_analysis()
