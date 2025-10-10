import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from .prediction_engine import get_all_party_predictions, get_prediction_trends

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


def plot_election_predictions():
    """
    Plot bar chart showing election prediction probabilities for all parties
    """
    from .prediction_engine import get_all_party_predictions
    predictions = get_all_party_predictions()
    
    if not predictions:
        print("No prediction data available for visualization.")
        return
    
    parties = list(predictions.keys())
    probabilities = [predictions[party] * 100 for party in parties]  # Convert to percentage
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(parties, probabilities, color=['skyblue', 'orange', 'lightgreen', 'pink'][:len(parties)])
    plt.ylabel('Win Probability (%)')
    plt.title('Election Prediction: Winning Probability by Party')
    plt.ylim(0, 100)
    
    # Add percentage labels on bars
    for bar, prob in zip(bars, probabilities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{prob:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def plot_prediction_trends(party_name):
    """
    Plot historical prediction trends for a specific party
    """
    from .prediction_engine import get_prediction_trends
    import matplotlib.dates as mdates
    from datetime import datetime
    
    history = get_prediction_trends(party_name)
    
    if not history:
        print(f"No prediction history available for {party_name}.")
        return
    
    # Extract dates and prediction probabilities
    dates = [datetime.fromisoformat(entry['date'].replace('Z', '+00:00')) for entry in history]
    probabilities = [entry['prediction_data']['win_probability'] * 100 for entry in history]  # Convert to percentage
    
    plt.figure(figsize=(10, 6))
    plt.plot(dates, probabilities, marker='o', linewidth=2, markersize=6)
    plt.ylabel('Win Probability (%)')
    plt.title(f'Prediction Trend for {party_name}')
    plt.ylim(0, 100)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
