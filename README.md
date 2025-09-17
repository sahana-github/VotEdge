# VotEdge - Political Data Collection and Analysis

This project collects political data from news and social media sources and performs sentiment analysis to understand public opinion towards political parties.

## Setup

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Install NLTK data for TextBlob:
   ```bash
   python -m textblob.download_corpora
   ```

3. Create a `.env` file with your API keys:
   ```env
   NEWSAPI_KEY=your_newsapi_key
   ```

4. Install Playwright browsers (for social media scraping):
   ```bash
   playwright install chromium
   ```

## Usage

Run the main application:
```bash
python main.py
```

The application will:
1. Ask for a political party name to analyze
2. Fetch recent news articles about that party
3. Fetch recent tweets from official party/accounts associated with that party
4. Perform sentiment analysis on both news and tweets
5. Save all results in a unified JSON file

## Configuration

The `party_twitter_handles.json` file maps political parties to their official Twitter accounts:
```json
{
  "BJP": ["narendramodi", "AmitShah", "BJP4India"],
  "INC": ["RahulGandhi", "INCIndia", "priyankagandhi"],
  "AAP": ["ArvindKejriwal", "AamAadmiParty"],
  "TMC": ["MamataOfficial", "AITCofficial"]
}
```

You can modify this file to add more parties or accounts.

## Project Structure

- `src/votedge/`: Main source code
  - `news_fetcher.py`: Fetches news articles using NewsAPI
  - `social_media_fetcher.py`: Scrapes tweets from X.com
  - `sentimentAnalysis.py`: Performs sentiment analysis using TextBlob
  - `data_processor.py`: Unified data processing and analysis
- `data/raw/`: Raw and processed data files
- `party_twitter_handles.json`: Party to Twitter handles mapping
- `requirements.txt`: Python dependencies

## Output

- `data/raw/news.csv`: Raw news article titles
- `data/raw/tweets.csv`: Raw tweet data
- `analysis_results_{party}_{timestamp}.json`: Combined sentiment analysis results

## Recent Improvements

### Fixed Social Media Scraper Issues
- Updated Playwright to run in headless mode for better automated environments
- Modified scraper to extract clean tweet text instead of raw JSON data
- Improved data processing to handle complex JSON from social media scraper
- Enhanced error handling for empty or malformed tweets

## Future Improvements

### Error Handling & Reliability
- Add proper error handling for NewsAPI rate limits and connection issues
- Implement retry mechanisms for failed API requests
- Add better error messages for users when scraping fails

### Data Visualization
- Create charts and graphs for sentiment analysis results
- Implement trend visualization over time
- Add comparison charts between different parties

### Additional Social Media Support
- Add support for Facebook, Instagram, or other platforms
- Implement a unified interface for multiple social media sources

### Historical Data & Trend Analysis
- Store historical sentiment data for tracking changes over time
- Implement trend analysis to identify shifts in public opinion
- Add comparison features between different time periods

### Automated Reporting
- Generate PDF or HTML reports with analysis results
- Schedule automated runs for regular sentiment tracking
- Add email notification support for significant changes

