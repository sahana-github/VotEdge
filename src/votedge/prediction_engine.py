"""
Election Prediction Engine
Transforms sentiment analysis results into election prediction probabilities
"""
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List
import os


def calculate_election_prediction(news_sentiment: float, tweets_sentiment: float, 
                                news_volume: int, tweet_volume: int) -> Dict[str, float]:
    """
    Calculate election prediction based on sentiment scores and volume
    
    Args:
        news_sentiment: Average sentiment score from news (between -1 and 1)
        tweets_sentiment: Average sentiment score from tweets (between -1 and 1)
        news_volume: Number of articles processed
        tweet_volume: Number of tweets processed
    
    Returns:
        Dictionary with prediction probability and supporting metrics
    """
    # Weighted combination of news and social media sentiment
    # News might have different weight than social media
    news_weight = 0.4
    social_weight = 0.6
    
    # Normalize volume to prevent bias toward parties with more mentions
    max_volume = max(news_volume, tweet_volume, 100)  # Use 100 as baseline
    volume_factor = min((news_volume + tweet_volume) / max_volume, 2.0)  # Cap at 2x
    
    # Calculate weighted sentiment score
    combined_sentiment = (news_sentiment * news_weight) + (tweets_sentiment * social_weight)
    
    # Adjust based on volume and other factors
    # Higher positive sentiment and volume should increase winning probability
    base_probability = 0.5  # Neutral starting point
    
    # Convert sentiment to probability range (0-1)
    # Positive sentiment increases probability, negative decreases
    sentiment_impact = combined_sentiment * 0.4  # Limit impact to 40%
    
    # Volume can amplify the sentiment effect
    amplified_impact = sentiment_impact * volume_factor
    
    # Calculate final probability
    prediction_probability = base_probability + amplified_impact
    
    # Ensure probability stays within reasonable bounds (10% to 90%)
    prediction_probability = max(0.1, min(0.9, prediction_probability))
    
    return {
        "win_probability": prediction_probability,
        "sentiment_score": combined_sentiment,
        "volume_factor": volume_factor,
        "raw_sentiment": {
            "news": news_sentiment,
            "tweets": tweets_sentiment
        },
        "raw_volume": {
            "news": news_volume,
            "tweets": tweet_volume
        }
    }


def update_prediction_history(party_name: str, prediction_data: Dict[str, float]):
    """
    Save prediction data to history file for trend analysis
    """
    history_file = f"election_predictions_{party_name.lower()}.json"
    
    # Load existing history
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except:
            history = []
    
    # Add new prediction with timestamp
    new_entry = {
        "date": datetime.now().isoformat(),
        "prediction_data": prediction_data
    }
    
    history.append(new_entry)
    
    # Keep only last 30 days of data
    if len(history) > 30:
        history = history[-30:]
    
    # Save updated history
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)


def get_all_party_predictions() -> Dict[str, float]:
    """
    Calculate predictions for all parties based on recent analysis data
    """
    # Find all analysis result files
    analysis_files = []
    for file in os.listdir('.'):
        if file.startswith('analysis_results_') and file.endswith('.json'):
            analysis_files.append(file)
    
    party_predictions = {}
    
    for file in analysis_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            party_name = data.get('party', '').upper()
            news_analysis = data['news_analysis']
            tweets_analysis = data['tweets_analysis']
            
            # Calculate prediction for this party
            prediction = calculate_election_prediction(
                news_sentiment=news_analysis.get('average_polarity', 0),
                tweets_sentiment=tweets_analysis.get('average_polarity', 0),
                news_volume=news_analysis.get('total', 0),
                tweet_volume=tweets_analysis.get('total', 0)
            )
            
            party_predictions[party_name] = prediction['win_probability']
            
            # Update history
            update_prediction_history(party_name, prediction)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    return party_predictions


def get_prediction_trends(party_name: str) -> List[Dict]:
    """
    Get historical prediction data for trend analysis
    """
    history_file = f"election_predictions_{party_name.lower()}.json"
    
    if not os.path.exists(history_file):
        return []
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        return history
    except:
        return []


if __name__ == "__main__":
    # Test the prediction engine
    print("Testing election prediction engine...")
    
    # Example calculation
    result = calculate_election_prediction(
        news_sentiment=0.2,
        tweets_sentiment=0.15,
        news_volume=50,
        tweet_volume=100
    )
    
    print(f"Prediction result: {result}")
    
    # Get all party predictions
    all_predictions = get_all_party_predictions()
    print(f"All party predictions: {all_predictions}")