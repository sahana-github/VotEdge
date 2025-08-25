import os
import pandas as pd
from io import StringIO
import sys

from news_fetcher import fetch_news_titles
from social_media_fetcher import scrape_last_10_tweets_by_clicking

os.makedirs("data/raw", exist_ok=True)

# -----------------------------
# 1️⃣ Capture news titles
# -----------------------------
party_name = input("Enter political party name: ").strip()

# Capture printed output
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

fetch_news_titles(party_name, os.getenv("NEWSAPI_KEY"))

sys.stdout = old_stdout

# Read lines and save
lines = mystdout.getvalue().splitlines()
news_list = [line.split(". ", 1)[-1] for line in lines if line and not line.startswith("📰")]

news_df = pd.DataFrame({"text": news_list})
news_df.to_csv("data/raw/news.csv", index=False)
print(f"Saved {len(news_df)} news rows to data/raw/news.csv")

# -----------------------------
# 2️⃣ Scrape tweets (optional)
# -----------------------------
username = input("Enter Twitter username to scrape: ").strip()
tweet_results = scrape_last_10_tweets_by_clicking(username)

rows = []
for r in tweet_results:
    text = r.get("text") if isinstance(r, dict) and "text" in r else str(r)
    rows.append({
        "source_type": "tweet",
        "username": username,
        "text": text,
        "url": None,
        "published_at": None
    })

if rows:
    tweets_df = pd.DataFrame(rows)
    tweets_df.to_csv("data/raw/tweets.csv", index=False)
    print("Saved data/raw/tweets.csv rows:", len(tweets_df))
else:
    print("No tweets found.")

print("✅ Data fetching complete. CSVs are in data/raw/")
