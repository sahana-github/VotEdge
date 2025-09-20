import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sns.set(style="whitegrid")

def load_analysis_results(folder=".", pattern="analysis_results_"):
    """
    Load all analysis JSON files from the folder
    Returns a DataFrame with columns: party, datetime, overall_sentiment
    """
    records = []
    for file in os.listdir(folder):
        if file.startswith(pattern) and file.endswith(".json"):
            try:
                with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    records.append({
                        "party": data.get("party"),
                        "datetime": datetime.fromisoformat(data.get("generated_at")),
                        "overall_sentiment": data.get("overall_sentiment", {}).get("combined_sentiment", 0.0)
                    })
            except Exception as e:
                print(f"Error reading {file}: {e}")
    return pd.DataFrame(records)

def plot_sentiment_distribution(news_analysis, tweets_analysis, party_name):
    """
    Plot bar charts for positive, negative, neutral counts in news and tweets
    """
    categories = ['Positive', 'Negative', 'Neutral']
    counts_news = [news_analysis['positive'], news_analysis['negative'], news_analysis['neutral']]
    counts_tweets = [tweets_analysis['positive'], tweets_analysis['negative'], tweets_analysis['neutral']]

    x = range(len(categories))
    width = 0.35

    plt.figure(figsize=(8,5))
    plt.bar(x, counts_news, width, label='News', color='skyblue')
    plt.bar([i + width for i in x], counts_tweets, width, label='Tweets', color='orange')
    plt.xticks([i + width/2 for i in x], categories)
    plt.ylabel('Count')
    plt.title(f"Sentiment Distribution for {party_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_comparison_between_parties(overall_sentiments_df):
    """
    Plot a bar chart comparing average overall sentiment between parties
    """
    plt.figure(figsize=(8,5))
    sns.barplot(data=overall_sentiments_df, x='party', y='overall_sentiment', palette='viridis')
    plt.ylabel('Average Sentiment')
    plt.title("Comparison of Average Sentiment Between Parties")
    plt.ylim(-1,1)
    plt.tight_layout()
    plt.show()

def plot_sentiment_trends(folder="."):
    """
    Plot trend of overall sentiment over time for all parties
    """
    df = load_analysis_results(folder)
    if df.empty:
        print("No analysis files found for trend visualization.")
        return

    plt.figure(figsize=(10,6))
    sns.lineplot(data=df, x='datetime', y='overall_sentiment', hue='party', marker='o')
    plt.ylabel('Overall Sentiment')
    plt.title('Sentiment Trend Over Time')
    plt.xticks(rotation=45)
    plt.ylim(-1,1)
    plt.tight_layout()
    plt.legend(title='Party')
    plt.show()
