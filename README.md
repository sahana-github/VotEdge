# VotEdge - Political Data Collection and Analysis

This project collects political data from news and social media sources for election prediction, processes it with LLMs, and trains machine learning models to predict election outcomes.

## Setup

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API keys:
   ```
   NEWSAPI_KEY=your_newsapi_key
   TWITTER_API_KEY=your_twitter_api_key
   TWITTER_API_SECRET=your_twitter_api_secret
   TWITTER_ACCESS_TOKEN=your_twitter_access_token
   TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
   TWITTER_BEARER_TOKEN=your_twitter_bearer_token
   GEMINI_API_KEY=your_gemini_api_key
   ```

3. Initialize the Airflow database:
   ```
   airflow db init
   ```

4. Create an Airflow user:
   ```
   airflow users create \
       --username admin \
       --firstname admin \
       --lastname admin \
       --role Admin \
       --email admin@example.com
   ```

## Usage

1. Start the Airflow scheduler:
   ```
   airflow scheduler
   ```

2. In another terminal, start the Airflow webserver:
   ```
   airflow webserver
   ```

3. Open your browser to http://localhost:8080 to access the Airflow UI

4. Enable the `votedge_pipeline` DAG in the Airflow UI

## Project Structure

- `codes/news_fetcher.py`: Fetches news articles using NewsAPI
- `codes/social_media_fetcher.py`: Fetches tweets from political figures
- `codes/data_collector.py`: Main data collection script that combines both sources
- `codes/llm_processor.py`: Processes collected data with Google's Gemini LLM
- `codes/feature_engineer.py`: Engineers features for ML models
- `codes/model_trainer.py`: Trains ML models for prediction with MLflow tracking
- `codes/visualizer.py`: Creates visualizations of predictions and insights
- `airflow/dags/votedge_dag.py`: Airflow DAG defining the ETL pipeline