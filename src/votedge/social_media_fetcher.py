from playwright.sync_api import sync_playwright
from .logger import logger
import time

def scrape_last_10_tweets_by_clicking(username: str, retries=3, delay=2) -> list:
    all_tweet_data = []

    for attempt in range(1, retries + 1):
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                context = browser.new_context(viewport={"width": 1920, "height": 1080})
                page = context.new_page()
                profile_url = f"https://x.com/{username}"
                page.goto(profile_url)
                page.wait_for_selector("[data-testid='tweet']")

                # Get all tweets and take first 10
                tweet_elements = page.locator("[data-testid='tweet']").all()[:10]

                for tweet_element in tweet_elements:
                    try:
                        # Safe way to get tweet link
                        link_el = tweet_element.locator("a[href*='/status/']").first
                        link = link_el.get_attribute("href") if link_el else None
                        if link:
                            full_url = f"https://x.com{link}"
                            new_page = context.new_page()
                            new_page.goto(full_url)
                            new_page.wait_for_selector("[data-testid='tweet']")
                            
                            # Get first matching tweet text to avoid strict mode violation
                            tweet_text_elements = new_page.locator("[data-testid='tweetText']").all()
                            if tweet_text_elements:
                                text = tweet_text_elements[0].inner_text()
                                all_tweet_data.append({'user': username, 'text': text})
                            new_page.close()
                    except Exception as e:
                        logger.warning(f"Failed to scrape one tweet for @{username}: {e}")

                browser.close()
                logger.info(f"Fetched {len(all_tweet_data)} tweets for @{username}")
                return all_tweet_data

        except Exception as e:
            logger.warning(f"Attempt {attempt} failed for @{username}: {e}")
            time.sleep(delay)

    logger.error(f"Failed to fetch tweets for @{username} after {retries} attempts")
    return all_tweet_data
