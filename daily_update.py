#!/usr/bin/env python3
"""
Daily Update Script for VotEdge Election Prediction Engine
This script runs daily to update election predictions based on fresh data
"""
import os
import sys
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from votedge.logger import logger
from votedge.news_fetcher import fetch_news_titles
from votedge.social_media_fetcher import scrape_last_10_tweets_by_clicking
from votedge.sentimentAnalysis import perform_sentiment_analysis
from votedge.data_processor import analyze_news_sentiment, analyze_tweets_sentiment, save_analysis_results, load_party_twitter_handles
from votedge.prediction_engine import get_all_party_predictions
from votedge.visualization import plot_election_predictions

def run_daily_analysis():
    """Run daily analysis for all configured parties"""
    # Load environment variables
    load_dotenv()
    newsapi_key = os.getenv("NEWSAPI_KEY")
    if not newsapi_key:
        logger.critical("NEWSAPI_KEY is missing in .env file")
        return

    logger.info("Starting daily election prediction update...")
    
    # Load party Twitter handles
    party_handles = load_party_twitter_handles()
    logger.info(f"Loaded configuration for {len(party_handles)} parties")
    
    # Process each party
    for party_name, twitter_handles in party_handles.items():
        logger.info(f"Processing daily update for {party_name}")
        
        # Collect data
        news_data = collect_news_data(party_name, newsapi_key)
        tweets_data = collect_tweets_data(party_name, twitter_handles)
        
        # Perform sentiment analysis
        logger.info("Performing sentiment analysis...")
        news_analysis = analyze_news_sentiment(news_data)
        tweets_analysis = analyze_tweets_sentiment(tweets_data)

        logger.info(f"News Analysis: Total={news_analysis['total']}, Positive={news_analysis['positive']}, Negative={news_analysis['negative']}, Neutral={news_analysis['neutral']}, AvgPolarity={news_analysis['average_polarity']:.3f}")
        logger.info(f"Tweets Analysis: Total={tweets_analysis['total']}, Positive={tweets_analysis['positive']}, Negative={tweets_analysis['negative']}, Neutral={tweets_analysis['neutral']}, AvgPolarity={tweets_analysis['average_polarity']:.3f}")

        # Save results
        try:
            filename = save_analysis_results(party_name, news_analysis, tweets_analysis)
            logger.info(f"Analysis results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    # Generate updated predictions
    logger.info("Generating updated election predictions...")
    predictions = get_all_party_predictions()
    
    if predictions:
        logger.info("Election predictions updated:")
        for party, prob in predictions.items():
            logger.info(f"  {party}: {prob*100:.1f}% chance to win")
        
        # Save predictions to a summary file
        summary_data = {
            "last_updated": datetime.now().isoformat(),
            "predictions": predictions,
            "method": "VotEdge Election Prediction Engine v1.0"
        }
        
        with open("election_forecast_summary.json", "w") as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info("Election forecast summary saved to election_forecast_summary.json")
    
    logger.info("Daily update completed successfully!")


def collect_news_data(party_name: str, api_key: str) -> list:
    """Collect news data for a party and return as list of texts"""
    logger.info(f"Fetching news for '{party_name}'...")
    news_list = fetch_news_titles(party_name, api_key, retries=3, delay=5)

    if not news_list:
        logger.warning(f"No news articles found for {party_name}")
    else:
        os.makedirs("data/raw", exist_ok=True)
        news_df = pd.DataFrame({"text": news_list})
        news_df.to_csv("data/raw/news.csv", index=False)
        logger.info(f"Saved {len(news_df)} news articles to data/raw/news.csv")
    return news_list


def collect_tweets_data(party_name: str, twitter_handles: list, retries=3, backoff=5) -> list:
    """Collect tweets data for a party"""
    all_tweets = []

    if not twitter_handles:
        logger.warning(f"No Twitter handles configured for {party_name}")
        return all_tweets

    logger.info(f"Fetching tweets for '{party_name}' from {len(twitter_handles)} accounts...")

    for username in twitter_handles:
        for attempt in range(retries):
            try:
                tweet_results = scrape_last_10_tweets_by_clicking(username)
                if tweet_results:
                    all_tweets.extend(tweet_results)
                    logger.info(f"Fetching {len(tweet_results)} tweets from @{username}")
                    break
                else:
                    logger.warning(f"No tweets fetched from @{username}. Retry {attempt + 1}")
            except Exception as e:
                logger.error(f"Error fetching tweets from @{username}: {e}")
            time.sleep(backoff * (attempt + 1))
        else:
            logger.critical(f"Failed to fetch tweets from @{username} after {retries} attempts")

    if all_tweets:
        os.makedirs("data/raw", exist_ok=True)
        rows = [{"text": tweet.get("text", str(tweet)) if isinstance(tweet, dict) else str(tweet)} for tweet in all_tweets]
        tweets_df = pd.DataFrame(rows)
        tweets_df.to_csv("data/raw/tweets.csv", index=False)
        logger.info(f"Saved {len(all_tweets)} total tweets to data/raw/tweets.csv")
    else:
        logger.warning("No tweets found")

    return all_tweets


def display_current_predictions():
    """Display current election predictions"""
    predictions = get_all_party_predictions()
    
    if predictions:
        print("\n🗳️ Current Election Forecast:")
        print("=" * 40)
        
        # Sort predictions by probability (descending)
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        for i, (party, prob) in enumerate(sorted_predictions, 1):
            status = "🟢 LEADING!" if i == 1 else ""
            print(f"{i}. {party}: {prob*100:.1f}% {status}")
        
        print("\nDetailed Breakdown:")
        for party, prob in predictions.items():
            print(f"  {party}: {prob*100:.1f}% chance to win")
        
        # Identify predicted winner
        winner = max(predictions.items(), key=lambda x: x[1])
        print(f"\n🔮 PREDICTED WINNER: {winner[0]} with {winner[1]*100:.1f}% chance!")
        
        return predictions
    else:
        print("No prediction data available. Run the analysis first.")
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        # Just show current predictions without running new analysis
        display_current_predictions()
    else:
        # Run full daily update
        run_daily_analysis()
        print("\nDaily update completed. To see current predictions, run: python daily_update.py predict")
        display_current_predictions()