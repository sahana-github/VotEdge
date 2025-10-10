"""
VotEdge - Political Data Collection and Analysis
"""
__version__ = "0.1.0"

from .news_fetcher import fetch_news_titles
from .social_media_fetcher import scrape_last_10_tweets_by_clicking
from .sentimentAnalysis import perform_sentiment_analysis, get_sentiment
from .data_processor import analyze_news_sentiment, analyze_tweets_sentiment, save_analysis_results, load_party_twitter_handles, get_sentiment
from .prediction_engine import calculate_election_prediction, get_all_party_predictions
from .visualization import plot_election_predictions, plot_prediction_trends