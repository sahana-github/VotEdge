#!/usr/bin/env python3
"""
VotEdge - Political Data Collection, Sentiment Analysis, and Visualization
"""

import os
import sys
import json
import time
from dotenv import load_dotenv
import pandas as pd
from io import StringIO
import sys as sys_io

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from votedge.logger import logger
from votedge.news_fetcher import fetch_news_titles
from votedge.social_media_fetcher import scrape_last_10_tweets_by_clicking
from votedge.sentimentAnalysis import perform_sentiment_analysis
from votedge.data_processor import analyze_news_sentiment, analyze_tweets_sentiment, save_analysis_results, load_party_twitter_handles
from votedge.visualization import plot_sentiment_distribution, plot_comparison_between_parties, plot_sentiment_trends, load_analysis_results
from votedge.prediction_engine import calculate_election_prediction, update_prediction_history, get_all_party_predictions


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
                    logger.info(f"Fetched {len(tweet_results)} tweets from @{username}")
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


def main():
    # Load environment variables
    load_dotenv()
    newsapi_key = os.getenv("NEWSAPI_KEY")
    if not newsapi_key:
        logger.critical("NEWSAPI_KEY is missing in .env file")
        return

    print("Welcome to VotEdge - Political Data Collection, Sentiment Analysis & Visualization")
    print("=" * 80)

    # Load party Twitter handles
    party_handles = load_party_twitter_handles()
    logger.info(f"Loaded configuration for {len(party_handles)} parties")

    # Display available parties
    print("\nAvailable parties:")
    for i, party in enumerate(party_handles.keys(), 1):
        print(f"{i}. {party}")

    # Get user input
    party_name = input("\nEnter political party name to analyze: ").strip()
    if not party_name:
        print("Party name is required!")
        return

    # Collect data
    news_data = collect_news_data(party_name, newsapi_key)
    twitter_handles = party_handles.get(party_name, [])
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

        overall_sentiment = (
            (news_analysis['average_polarity'] + tweets_analysis['average_polarity']) / 2
            if (news_analysis['total'] > 0 and tweets_analysis['total'] > 0)
            else (news_analysis['average_polarity'] if news_analysis['total'] > 0 else tweets_analysis['average_polarity'])
        )
        print(f"\nOverall Sentiment for {party_name}: {overall_sentiment:.3f}")
        if overall_sentiment > 0.1:
            print("   😊 Generally Positive")
        elif overall_sentiment < -0.1:
            print("   😞 Generally Negative")
        else:
            print("   😐 Generally Neutral")

    except Exception as e:
        logger.error(f"Error saving results: {e}")
        print("❌ Error saving analysis results. Check logs for details.")

    # ---------------- Visualization ----------------
    print("\n📊 Generating visualizations...")
    try:
        # 1️⃣ Sentiment distribution (News vs Tweets)
        plot_sentiment_distribution(news_analysis, tweets_analysis, party_name)

        # 2️⃣ Comparison between parties (average sentiment)
        df_all = load_analysis_results(".")
        if not df_all.empty:
            plot_comparison_between_parties(df_all)

        # 3️⃣ Trend over time
        plot_sentiment_trends(".")

        # 4️⃣ Election prediction bar chart
        print("\n🗳️ Generating election prediction...")
        plot_election_predictions()

    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        print("❌ Error generating visualizations. Check logs for details.")

    print("\nAnalysis & Visualization complete! Check logs/votedge.log for detailed logs and results in the current directory.")


if __name__ == "__main__":
    main()
