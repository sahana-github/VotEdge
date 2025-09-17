import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("NEWSAPI_KEY")

def fetch_news_titles(party_name, api_key):
    url = "https://newsapi.org/v2/everything"
    
    params = {
        'q': party_name,
        'language': 'en',
        'pageSize': 30,
        'sortBy': 'publishedAt',
        'apiKey': api_key
    }

    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code != 200:
        print("Error:", data.get("message", "Unknown error"))
        return

    articles = data.get("articles", [])
    if not articles:
        print("No articles found.")
        return

    print(f"\n📰 Latest News Titles for '{party_name}':\n")
    for i, article in enumerate(articles, start=1):
        print(f"{i}. {article['title']}")

if __name__ == "__main__":
    party_name = input("Enter political party name: ").strip()
    fetch_news_titles(party_name, api_key)
