from playwright.sync_api import sync_playwright

def scrape_last_10_tweets_by_clicking(username: str) -> list:
    """
    Scrape the last 10 tweets from a user's profile on X.com by clicking on each post
    and scraping the individual tweet page.
    """
    all_tweet_data = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        # Navigate to the user's profile page
        profile_url = f"https://x.com/{username}"
        page.goto(profile_url)
        page.wait_for_selector("[data-testid='tweet']")

        # Get all tweet elements and find their links
        tweet_elements = page.locator("[data-testid='tweet']").all()
        tweet_links = []
        for tweet_element in tweet_elements[:10]: # Take only the first 10 tweets displayed
            try:
                # Find the link within the tweet element
                link = tweet_element.locator("a[href*='/status/']").first.get_attribute("href")
                if link:
                    # Construct the full URL
                    full_url = f"https://x.com{link}"
                    tweet_links.append(full_url)
            except Exception as e:
                print(f"Could not get link from tweet element: {e}")

        # Use a new context for each tweet to ensure clean scraping
        for link in tweet_links:
            _xhr_calls = []
            
            def intercept_response(response):
                if response.request.resource_type == "xhr":
                    _xhr_calls.append(response)

            new_page = context.new_page()
            new_page.on("response", intercept_response)
            new_page.goto(link)
            new_page.wait_for_selector("[data-testid='tweet']")

            # Find the tweet-related XHR call
            tweet_calls = [f for f in _xhr_calls if "TweetResultByRestId" in f.url]

            for xhr in tweet_calls:
                try:
                    data = xhr.json()
                    result = data.get('data', {}).get('tweetResult', {}).get('result')
                    if result:
                        all_tweet_data.append(result)
                        break  # Found the main tweet data, move to the next link
                except Exception as e:
                    print(f"Could not parse JSON from XHR call: {e}")
            
            new_page.close() # Close the page after scraping
    
    return all_tweet_data

if __name__ == "__main__":
    username = "tetsuoai"
    tweets = scrape_last_10_tweets_by_clicking(username)
    for tweet in tweets:
        print(tweet)
        print("-" * 50)