"""
Unified data processor for VotEdge
Handles collection and sentiment analysis of both news and social media data
"""
import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple
from textblob import TextBlob

def get_sentiment(text: str) -> Dict[str, any]:
    """
    Determine sentiment of text using TextBlob
    Returns a dictionary with sentiment label and polarity score
    """
    analysis = TextBlob(str(text))
    polarity = analysis.polarity
    
    if polarity > 0:
        sentiment_label = "Positive"
    elif polarity < 0:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"
    
    return {
        "label": sentiment_label,
        "polarity": polarity
    }

def analyze_news_sentiment(news_data: List[str]) -> Dict[str, any]:
    """
    Perform sentiment analysis on news data
    Returns a dictionary with sentiment statistics
    """
    if not news_data:
        return {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "average_polarity": 0.0,
            "details": []
        }
    
    sentiments = []
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    total_polarity = 0.0
    
    for news_item in news_data:
        sentiment = get_sentiment(news_item)
        sentiments.append({
            "text": news_item,
            "sentiment": sentiment["label"],
            "polarity": sentiment["polarity"]
        })
        
        total_polarity += sentiment["polarity"]
        
        if sentiment["label"] == "Positive":
            positive_count += 1
        elif sentiment["label"] == "Negative":
            negative_count += 1
        else:
            neutral_count += 1
    
    return {
        "total": len(news_data),
        "positive": positive_count,
        "negative": negative_count,
        "neutral": neutral_count,
        "average_polarity": total_polarity / len(news_data) if news_data else 0.0,
        "details": sentiments
    }

def analyze_tweets_sentiment(tweets_data: List[Dict]) -> Dict[str, any]:
    """
    Perform sentiment analysis on tweets data
    Returns a dictionary with sentiment statistics
    """
    if not tweets_data:
        return {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "average_polarity": 0.0,
            "details": []
        }
    
    sentiments = []
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    total_polarity = 0.0
    
    for tweet in tweets_data:
        # Extract text from tweet
        if isinstance(tweet, dict):
            text = tweet.get("text", "")
        else:
            text = str(tweet)
        
        # Skip empty tweets
        if not text:
            continue
            
        sentiment = get_sentiment(text)
        sentiments.append({
            "text": text,
            "sentiment": sentiment["label"],
            "polarity": sentiment["polarity"]
        })
        
        total_polarity += sentiment["polarity"]
        
        if sentiment["label"] == "Positive":
            positive_count += 1
        elif sentiment["label"] == "Negative":
            negative_count += 1
        else:
            neutral_count += 1
    
    # Handle case where all tweets might be empty
    total_tweets = len(sentiments)
    if total_tweets == 0:
        return {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "average_polarity": 0.0,
            "details": []
        }
    
    return {
        "total": total_tweets,
        "positive": positive_count,
        "negative": negative_count,
        "neutral": neutral_count,
        "average_polarity": total_polarity / total_tweets,
        "details": sentiments
    }

def save_analysis_results(party_name: str, news_results: Dict, tweets_results: Dict, filename: str = None) -> str:
    """
    Save analysis results to a JSON file
    Returns the path to the saved file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_results_{party_name}_{timestamp}.json"
    
    results = {
        "party": party_name,
        "generated_at": datetime.now().isoformat(),
        "news_analysis": news_results,
        "tweets_analysis": tweets_results,
        "overall_sentiment": {
            "news_sentiment": news_results["average_polarity"],
            "tweets_sentiment": tweets_results["average_polarity"],
            "combined_sentiment": (news_results["average_polarity"] + tweets_results["average_polarity"]) / 2 
                                  if (news_results["total"] > 0 and tweets_results["total"] > 0) 
                                  else (news_results["average_polarity"] if news_results["total"] > 0 else tweets_results["average_polarity"])
        }
    }
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return filename

def load_party_twitter_handles() -> Dict[str, List[str]]:
    """
    Load political party Twitter handles from a configuration file
    Returns a dictionary mapping party names to lists of Twitter handles
    """
    config_file = "party_twitter_handles.json"
    
    # Default configuration
    default_config = {
        "BJP": ["narendramodi", "AmitShah", "BJP4India"],
        "INC": ["RahulGandhi", "INCIndia", "priyankagandhi"],
        "AAP": ["ArvindKejriwal", "AamAadmiParty"],
        "TMC": ["MamataOfficial", "AITCofficial"]
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {config_file}: {e}. Using default configuration.")
            return default_config
    else:
        # Create default config file
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        return default_config