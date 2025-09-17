#!/usr/bin/env python3
"""
VotEdge - Political Data Collection and Analysis

This project collects political data from news and social media sources
and performs sentiment analysis to understand public opinion.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from votedge.news_fetcher import fetch_news_titles
from votedge.social_media_fetcher import scrape_last_10_tweets_by_clicking
from votedge.sentimentAnalysis import perform_sentiment_analysis
from votedge.data_processor import analyze_news_sentiment, analyze_tweets_sentiment, save_analysis_results, load_party_twitter_handles
import pandas as pd
from io import StringIO
import sys as sys_io

def collect_news_data(party_name: str, api_key: str) -> list:
    """Collect news data for a party and return as list of texts"""
    print(f"\n🔍 Fetching news for '{party_name}'...")
    
    # Capture printed output from news fetcher
    old_stdout = sys_io.stdout
    sys_io.stdout = mystdout = StringIO()
    
    fetch_news_titles(party_name, api_key)
    
    sys_io.stdout = old_stdout
    
    # Process news data
    lines = mystdout.getvalue().splitlines()
    news_list = [line.split(". ", 1)[-1] for line in lines if line and not line.startswith("📰")]
    
    if news_list:
        # Save to CSV for reference
        os.makedirs("data/raw", exist_ok=True)
        news_df = pd.DataFrame({"text": news_list})
        news_df.to_csv("data/raw/news.csv", index=False)
        print(f"✅ Saved {len(news_df)} news articles to data/raw/news.csv")
    else:
        print("❌ No news articles found")
    
    return news_list

def collect_tweets_data(party_name: str, twitter_handles: list) -> list:
    """Collect tweets data for a party and return as list of tweet objects"""
    all_tweets = []
    
    if not twitter_handles:
        print("No Twitter handles configured for this party")
        return all_tweets
    
    print(f"\n🔍 Fetching tweets for '{party_name}' from {len(twitter_handles)} accounts...")
    
    for username in twitter_handles:
        print(f"  - Fetching tweets from @{username}...")
        try:
            tweet_results = scrape_last_10_tweets_by_clicking(username)
            all_tweets.extend(tweet_results)
            print(f"    ✅ Fetched {len(tweet_results)} tweets")
        except Exception as e:
            print(f"    ❌ Error fetching tweets from @{username}: {e}")
    
    if all_tweets:
        # Save to CSV for reference
        os.makedirs("data/raw", exist_ok=True)
        rows = []
        for tweet in all_tweets:
            if isinstance(tweet, dict):
                text = tweet.get("text", "")
            else:
                text = str(tweet)
            rows.append({"text": text})
        
        tweets_df = pd.DataFrame(rows)
        tweets_df.to_csv("data/raw/tweets.csv", index=False)
        print(f"✅ Saved {len(all_tweets)} total tweets to data/raw/tweets.csv")
    else:
        print("❌ No tweets found")
    
    return all_tweets

def main():
    # Load environment variables
    load_dotenv()
    newsapi_key = os.getenv("NEWSAPI_KEY")
    
    if not newsapi_key:
        print("Please set NEWSAPI_KEY in your .env file")
        return
    
    print("Welcome to VotEdge - Political Data Collection and Analysis")
    print("=" * 60)
    
    # Load party Twitter handles configuration
    party_handles = load_party_twitter_handles()
    print(f"Loaded configuration for {len(party_handles)} parties")
    
    # Display available parties
    print("\nAvailable parties:")
    for i, party in enumerate(party_handles.keys(), 1):
        print(f"{i}. {party}")
    
    # Get user input
    party_name = input("\nEnter political party name to analyze: ").strip()
    if not party_name:
        print("Party name is required!")
        return
    
    # Collect news data
    news_data = collect_news_data(party_name, newsapi_key)
    
    # Collect tweets data
    twitter_handles = party_handles.get(party_name, [])
    tweets_data = collect_tweets_data(party_name, twitter_handles)
    
    # Perform sentiment analysis
    print("\n" + "=" * 60)
    print("🧠 Performing sentiment analysis...")
    
    # Analyze news sentiment
    news_analysis = analyze_news_sentiment(news_data)
    print(f"📰 News Analysis:")
    print(f"   Total articles: {news_analysis['total']}")
    print(f"   Positive: {news_analysis['positive']}, Negative: {news_analysis['negative']}, Neutral: {news_analysis['neutral']}")
    print(f"   Average polarity: {news_analysis['average_polarity']:.3f}")
    
    # Analyze tweets sentiment
    tweets_analysis = analyze_tweets_sentiment(tweets_data)
    print(f"🐦 Tweets Analysis:")
    print(f"   Total tweets: {tweets_analysis['total']}")
    print(f"   Positive: {tweets_analysis['positive']}, Negative: {tweets_analysis['negative']}, Neutral: {tweets_analysis['neutral']}")
    print(f"   Average polarity: {tweets_analysis['average_polarity']:.3f}")
    
    # Save combined results
    print("\n💾 Saving analysis results...")
    try:
        filename = save_analysis_results(party_name, news_analysis, tweets_analysis)
        print(f"✅ Analysis results saved to {filename}")
        
        # Display overall sentiment
        overall_sentiment = (news_analysis['average_polarity'] + tweets_analysis['average_polarity']) / 2 if (news_analysis['total'] > 0 and tweets_analysis['total'] > 0) else (news_analysis['average_polarity'] if news_analysis['total'] > 0 else tweets_analysis['average_polarity'])
        print(f"\n📊 Overall Sentiment for {party_name}: {overall_sentiment:.3f}")
        if overall_sentiment > 0.1:
            print("   😊 Generally Positive")
        elif overall_sentiment < -0.1:
            print("   😞 Generally Negative")
        else:
            print("   😐 Generally Neutral")
            
    except Exception as e:
        print(f"❌ Error saving results: {e}")
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check the current directory for results.")

if __name__ == "__main__":
    main()