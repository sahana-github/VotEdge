import requests
import os
import time
from dotenv import load_dotenv
from .logger import logger

load_dotenv()
api_key = os.getenv("NEWSAPI_KEY")

def fetch_news_titles(party_name, api_key, retries=3, delay=2):
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': party_name,
        'language': 'en',
        'pageSize': 30,
        'sortBy': 'publishedAt',
        'apiKey': api_key
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if response.status_code != 200:
                logger.warning(f"Attempt {attempt}: NewsAPI Error: {data.get('message', 'Unknown')}")
                time.sleep(delay)
                continue

            articles = data.get("articles", [])
            if not articles:
                logger.info(f"No articles found for {party_name}")
                return []

            logger.info(f"Fetched {len(articles)} articles for {party_name}")
            return [article['title'] for article in articles]

        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt}: Network error: {e}")
            time.sleep(delay)

    logger.error(f"Failed to fetch news for {party_name} after {retries} attempts")
    return []
