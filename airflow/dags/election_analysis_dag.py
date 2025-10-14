"""
Election Analysis DAG for VotEdge
This DAG orchestrates the daily election prediction workflow
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from votedge.news_fetcher import fetch_news_titles
from votedge.social_media_fetcher import scrape_last_10_tweets_by_clicking
from votedge.sentimentAnalysis import perform_sentiment_analysis
from votedge.data_processor import analyze_news_sentiment, analyze_tweets_sentiment, save_analysis_results, load_party_twitter_handles
from votedge.prediction_engine import get_all_party_predictions
from votedge.visualization import plot_election_predictions

# Default arguments for the DAG
default_args = {
    'owner': 'VotEdge Team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'election_analysis_dag',
    default_args=default_args,
    description='VotEdge Election Analysis Workflow',
    schedule_interval=timedelta(days=1),  # Run daily
    start_date=days_ago(1),
    catchup=False,
    tags=['election', 'analysis', 'prediction'],
)

def collect_news_data(party_name: str, api_key: str) -> list:
    """Collect news data for a party and return as list of texts"""
    from votedge.logger import logger
    logger.info(f"Fetching news for '{party_name}'...")
    news_list = fetch_news_titles(party_name, api_key, retries=3, delay=5)

    if not news_list:
        logger.warning(f"No news articles found for {party_name}")
    else:
        os.makedirs("data/raw", exist_ok=True)
        import pandas as pd
        news_df = pd.DataFrame({"text": news_list})
        news_df.to_csv("data/raw/news.csv", index=False)
        logger.info(f"Saved {len(news_df)} news articles to data/raw/news.csv")
    return news_list


def collect_tweets_data(party_name: str, twitter_handles: list, retries=3, backoff=5) -> list:
    """Collect tweets data for a party"""
    import time
    from votedge.logger import logger
    
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
                import traceback
                logger.error(traceback.format_exc())
            time.sleep(backoff * (attempt + 1))
        else:
            logger.critical(f"Failed to fetch tweets from @{username} after {retries} attempts")

    if all_tweets:
        os.makedirs("data/raw", exist_ok=True)
        import pandas as pd
        rows = [{"text": tweet.get("text", str(tweet)) if isinstance(tweet, dict) else str(tweet)} for tweet in all_tweets]
        tweets_df = pd.DataFrame(rows)
        tweets_df.to_csv("data/raw/tweets.csv", index=False)
        logger.info(f"Saved {len(all_tweets)} total tweets to data/raw/tweets.csv")
    else:
        logger.warning("No tweets found")

    return all_tweets


def run_party_analysis(**context):
    """Run analysis for all parties"""
    from dotenv import load_dotenv
    import os
    from votedge.logger import logger
    
    # Load environment variables
    load_dotenv()
    newsapi_key = os.getenv("NEWSAPI_KEY")
    if not newsapi_key:
        raise ValueError("NEWSAPI_KEY is missing in .env file")

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
            raise


def update_election_predictions(**context):
    """Update election predictions"""
    from datetime import datetime
    import json
    from votedge.logger import logger
    
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
    else:
        logger.warning("No predictions generated")
        raise ValueError("No predictions were generated")


def generate_visualizations(**context):
    """Generate visualizations"""
    from votedge.logger import logger
    from votedge.visualization import plot_election_predictions
    
    logger.info("Generating election prediction visualizations...")
    plot_election_predictions()
    logger.info("Visualization generation completed")


# Define tasks
collect_and_analyze_task = PythonOperator(
    task_id='collect_and_analyze_data',
    python_callable=run_party_analysis,
    dag=dag,
)

update_predictions_task = PythonOperator(
    task_id='update_election_predictions',
    python_callable=update_election_predictions,
    dag=dag,
)

generate_visualizations_task = PythonOperator(
    task_id='generate_visualizations',
    python_callable=generate_visualizations,
    dag=dag,
)

# Set task dependencies
collect_and_analyze_task >> update_predictions_task >> generate_visualizations_task