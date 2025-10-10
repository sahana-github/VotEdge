# chatbot/chatbot_app.py
import streamlit as st
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Import the prediction and visualization functionality
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.votedge.prediction_engine import get_all_party_predictions
from src.votedge.visualization import (
    plot_election_predictions as plot_election_bars,
    load_analysis_results,
    plot_sentiment_distribution,
    plot_comparison_between_parties,
    plot_sentiment_trends,
    plot_prediction_trends
)

# Additional imports for RAG functionality with LangChain and HuggingFace embeddings
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import json

# Load environment variables
load_dotenv()

# Load environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Import functions for full analysis
from src.votedge.news_fetcher import fetch_news_titles
from src.votedge.social_media_fetcher import scrape_last_10_tweets_by_clicking
from src.votedge.sentimentAnalysis import perform_sentiment_analysis
from src.votedge.data_processor import analyze_news_sentiment, analyze_tweets_sentiment, save_analysis_results, load_party_twitter_handles
from src.votedge.prediction_engine import get_all_party_predictions
from src.votedge.logger import logger
import time

st.set_page_config(page_title="VotEdge Election Prediction Chatbot", layout="wide")
st.title("VotEdge Election Prediction Chatbot 🤖")
st.markdown("Visualize political data and ask questions about the visualizations!")

# Load CSV data
data_folder = Path(".") / "data" / "raw"
csv_files = list(data_folder.glob("*.csv"))

# Check if any analysis results exist
analysis_files = []
for file in Path(".").glob("analysis_results_*.json"):
    analysis_files.append(file)

# Function to create vector store from data for RAG
def create_vectorstore(all_rows):
    """
    Creates a FAISS vector store from the provided text data
    """
    if not all_rows:
        return None
        
    # Extract text content from all_rows
    texts = []
    metadatas = []
    
    for row in all_rows:
        text_content = row["text"]
        texts.append(text_content)
        metadatas.append({"source_file": row["file"]})
    
    if texts:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents(texts, metadatas=metadatas)
        
        # Initialize HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore
    return None

def collect_news_data(party_name: str, api_key: str) -> list:
    """Collect news data for a party and return as list of texts"""
    logger.info(f"Fetching news for '{party_name}'...")
    news_list = fetch_news_titles(party_name, api_key, retries=3, delay=5)

    if not news_list:
        logger.warning(f"No news articles found for {party_name}")
    else:
        os.makedirs("data/raw", exist_ok=True)
        news_df = pd.DataFrame({"text": news_list})
        news_df.to_csv("data/raw/news.csv", index=False)
        logger.info(f"Saved {len(news_df)} news articles to data/raw/news.csv")
    return news_list

def collect_tweets_data(party_name: str, twitter_handles: list, retries=3, backoff=5) -> list:
    """Collect tweets data for a party"""
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
                    logger.info(f"Fetched {len(tweet_results)} tweets from @{username}")
                    break
                else:
                    logger.warning(f"No tweets fetched from @{username}. Retry {attempt + 1}")
            except Exception as e:
                logger.error(f"Error fetching tweets from @{username}: {e}")
            time.sleep(backoff * (attempt + 1))
        else:
            logger.critical(f"Failed to fetch tweets from @{username} after {retries} attempts")

    if all_tweets:
        os.makedirs("data/raw", exist_ok=True)
        rows = [{"text": tweet.get("text", str(tweet)) if isinstance(tweet, dict) else str(tweet)} for tweet in all_tweets]
        tweets_df = pd.DataFrame(rows)
        tweets_df.to_csv("data/raw/tweets.csv", index=False)
        logger.info(f"Saved {len(all_tweets)} total tweets to data/raw/tweets.csv")
    else:
        logger.warning("No tweets found")

    return all_tweets

def run_daily_analysis():
    """Run daily analysis for all configured parties"""
    # Load environment variables
    newsapi_key = os.getenv("NEWSAPI_KEY")
    if not newsapi_key:
        st.error("NEWSAPI_KEY is missing in .env file")
        return

    logger.info("Starting daily election prediction update...")
    
    # Load party Twitter handles
    party_handles = load_party_twitter_handles()
    logger.info(f"Loaded configuration for {len(party_handles)} parties")
    
    # Process each party
    for i, (party_name, twitter_handles) in enumerate(party_handles.items()):
        logger.info(f"Processing daily update for {party_name}")
        
        # Update progress in Streamlit
        st.sidebar.info(f"Processing {party_name} ({i+1}/{len(party_handles)})...")
        
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
    
    # Generate updated predictions
    logger.info("Generating updated election predictions...")
    predictions = get_all_party_predictions()
    
    if predictions:
        logger.info("Election predictions updated:")
        for party, prob in predictions.items():
            logger.info(f"  {party}: {prob*100:.1f}% chance to win")
        
        # Save predictions to a summary file
        from datetime import datetime
        summary_data = {
            "last_updated": datetime.now().isoformat(),
            "predictions": predictions,
            "method": "VotEdge Election Prediction Engine v1.0"
        }
        
        with open("election_forecast_summary.json", "w") as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info("Election forecast summary saved to election_forecast_summary.json")
    
    logger.info("Daily update completed successfully!")

# Sidebar for election predictions
with st.sidebar:
    st.header("🗳️ Election Prediction")
    
    # Button to refresh and display current predictions
    if st.button("Refresh Election Predictions"):
        # Show current election predictions
        predictions = get_all_party_predictions()
        
        if predictions:
            st.subheader("Winning Probabilities")
            
            # Create bar chart for election predictions
            parties = list(predictions.keys())
            probabilities = [predictions[party] * 100 for party in parties]  # Convert to percentage
            
            # Create a bar chart
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(parties, probabilities, color=['skyblue', 'orange', 'lightgreen', 'pink'][:len(parties)])
            ax.set_ylabel('Win Probability (%)')
            ax.set_title('Election Prediction: Winning Probability by Party')
            ax.set_ylim(0, 100)
            
            # Add percentage labels on bars
            for bar, prob in zip(bars, probabilities):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'{prob:.1f}%', ha='center', va='bottom')
            
            st.pyplot(fig)
            
            # Display as text too
            st.subheader("Prediction Details:")
            for party, prob in predictions.items():
                st.write(f"{party}: {prob*100:.1f}% chance to win")
        else:
            st.warning("No prediction data available. Run the main analysis first to generate predictions.")
    
    st.header("🔄 Full Analysis")
    
    # Button to run full daily analysis for all parties
    if st.button("Run Complete Analysis (All Parties)"):
        with st.spinner("Running complete analysis for all parties (news, tweets, sentiment, predictions)..."):
            try:
                # Run the complete analysis for all parties
                run_daily_analysis()
                st.success("✅ Complete analysis completed for all parties!")
                # Rerun the app to update the visualizations
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error running complete analysis: {str(e)}")
                # Fallback: run the main analysis
                try:
                    # Attempt to run main analysis for a single party
                    party_handles = load_party_twitter_handles()
                    if party_handles:
                        first_party = list(party_handles.keys())[0]
                        newsapi_key = os.getenv("NEWSAPI_KEY")
                        
                        # Collect data for first party
                        news_data = collect_news_data(first_party, newsapi_key)
                        twitter_handles = party_handles.get(first_party, [])
                        tweets_data = collect_tweets_data(first_party, twitter_handles)
                        
                        # Perform sentiment analysis
                        news_analysis = analyze_news_sentiment(news_data)
                        tweets_analysis = analyze_tweets_sentiment(tweets_data)
                        
                        # Save results
                        filename = save_analysis_results(first_party, news_analysis, tweets_analysis)
                        st.success(f"✅ Partial analysis completed for {first_party}!")
                    else:
                        st.error("❌ No party handles found in configuration.")
                except Exception as fallback_error:
                    st.error(f"❌ Fallback analysis also failed: {str(fallback_error)}")

# Main content area - Add more tabs for visualizations
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Election Predictions", 
    "📈 Sentiment Distribution", 
    "⚖️ Party Comparison", 
    "📉 Sentiment Trends", 
    "💬 Ask Questions"
])

with tab1:
    st.header("Election Forecast Dashboard")
    
    # Load and display all predictions
    all_predictions = get_all_party_predictions()
    
    if all_predictions:
        st.subheader("Current Election Forecast")
        
        # Create columns for each party
        cols = st.columns(len(all_predictions))
        for i, (party, prob) in enumerate(all_predictions.items()):
            with cols[i]:
                st.metric(label=party, value=f"{prob*100:.1f}%", delta=None)
        
        # Create visual bar chart
        st.subheader("Visual Forecast")
        if st.button("Generate Prediction Chart", key="gen_pred_chart_tab1"):
            try:
                plot_election_bars()
            except:
                # Fallback to matplotlib chart
                parties = list(all_predictions.keys())
                probabilities = [all_predictions[party] * 100 for party in parties]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(parties, probabilities, color=['skyblue', 'orange', 'lightgreen', 'pink'][:len(parties)])
                ax.set_ylabel('Win Probability (%)')
                ax.set_title('Election Prediction: Winning Probability by Party')
                ax.set_ylim(0, 100)
                
                # Add percentage labels on bars
                for bar, prob in zip(bars, probabilities):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           f'{prob:.1f}%', ha='center', va='bottom')
                
                st.pyplot(fig)
    else:
        st.warning("No election predictions available. Please run the main analysis first.")

# Tab for sentiment distribution
with tab2:
    st.header("Sentiment Distribution")
    
    if analysis_files:
        # Parse party name from analysis files to get one for sample visualization
        available_parties = set()
        for file in analysis_files:
            # Extract party from filename like "analysis_results_BJP_20250920_135101.json"
            name_parts = file.name.split('_')
            if len(name_parts) >= 3:
                party = name_parts[2].upper() if name_parts[2] != 'results' else name_parts[1].upper()
                available_parties.add(party)
        
        if available_parties:
            selected_party = st.selectbox("Select a party to visualize sentiment distribution:", list(available_parties))
            
            # Find the latest analysis file for the selected party
            latest_file = None
            for file in analysis_files:
                if selected_party.lower() in file.name.lower():
                    if latest_file is None or file.stat().st_mtime > latest_file.stat().st_mtime:
                        latest_file = file
            
            if latest_file:
                try:
                    with open(latest_file, 'r') as f:
                        data = json.load(f)
                    
                    news_analysis = data.get('news_analysis', {})
                    tweets_analysis = data.get('tweets_analysis', {})
                    
                    if news_analysis and tweets_analysis:
                        # Show summary statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader(f"News Sentiment for {selected_party}")
                            st.metric("Total Articles", news_analysis.get('total', 0))
                            st.metric("Positive", news_analysis.get('positive', 0))
                            st.metric("Negative", news_analysis.get('negative', 0))
                            st.metric("Neutral", news_analysis.get('neutral', 0))
                            avg_sentiment = news_analysis.get('average_polarity', 0)
                            st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
                        
                        with col2:
                            st.subheader(f"Social Media Sentiment for {selected_party}")
                            st.metric("Total Tweets", tweets_analysis.get('total', 0))
                            st.metric("Positive", tweets_analysis.get('positive', 0))
                            st.metric("Negative", tweets_analysis.get('negative', 0))
                            st.metric("Neutral", tweets_analysis.get('neutral', 0))
                            avg_sentiment = tweets_analysis.get('average_polarity', 0)
                            st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
                        
                        # Create sentiment distribution chart
                        if st.button("Generate Sentiment Distribution Chart", key="gen_sent_dist_chart"):
                            try:
                                # Create a matplotlib chart manually since we can't call the original function directly
                                categories = ['Positive', 'Negative', 'Neutral']
                                counts_news = [
                                    news_analysis.get('positive', 0),
                                    news_analysis.get('negative', 0),
                                    news_analysis.get('neutral', 0)
                                ]
                                counts_tweets = [
                                    tweets_analysis.get('positive', 0),
                                    tweets_analysis.get('negative', 0),
                                    tweets_analysis.get('neutral', 0)
                                ]

                                x = range(len(categories))
                                width = 0.35

                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.bar(x, counts_news, width, label='News', color='skyblue')
                                ax.bar([i + width for i in x], counts_tweets, width, label='Tweets', color='orange')
                                ax.set_xticks([i + width/2 for i in x])
                                ax.set_xticklabels(categories)
                                ax.set_ylabel('Count')
                                ax.set_title(f"Sentiment Distribution for {selected_party}")
                                ax.legend()
                                plt.tight_layout()
                                
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error generating sentiment distribution chart: {e}")
                    else:
                        st.warning("No analysis data found in the selected file.")
                except Exception as e:
                    st.error(f"Error reading analysis file: {e}")
            else:
                st.warning(f"No analysis file found for {selected_party}.")
        else:
            st.warning("No parties found in analysis files.")
    else:
        st.warning("No analysis results available. Please run the main analysis first.")

# Tab for party comparison
with tab3:
    st.header("Party Comparison Dashboard")
    
    if analysis_files:
        df_all = load_analysis_results(".", pattern="analysis_results_")
        if not df_all.empty:
            st.subheader("Average Sentiment Comparison Between Parties")
            
            # Create comparison chart
            if st.button("Generate Party Comparison Chart", key="gen_party_comp_chart"):
                try:
                    # Group by party and calculate mean sentiment
                    avg_sentiments = df_all.groupby('party')['overall_sentiment'].mean().reset_index()
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(avg_sentiments['party'], avg_sentiments['overall_sentiment'], 
                                 color=['skyblue', 'orange', 'lightgreen', 'pink'][:len(avg_sentiments)])
                    ax.set_ylabel('Average Sentiment')
                    ax.set_title('Comparison of Average Sentiment Between Parties')
                    ax.set_ylim(-1, 1)
                    
                    # Add sentiment labels on bars
                    for bar, sentiment in zip(bars, avg_sentiments['overall_sentiment']):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                               f'{sentiment:.2f}', ha='center', va='bottom')
                    
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Also show as a table
                    st.subheader("Detailed Comparison Table")
                    st.dataframe(avg_sentiments)
                except Exception as e:
                    st.error(f"Error generating party comparison chart: {e}")
        else:
            st.warning("No analysis results loaded for comparison.")
    else:
        st.warning("No analysis results available. Please run the main analysis first.")

# Tab for sentiment trends
with tab4:
    st.header("Sentiment Trends Over Time")
    
    if analysis_files:
        st.subheader("Historical Sentiment Trends")
        
        # Create trends chart
        if st.button("Generate Sentiment Trends Chart", key="gen_sent_trends_chart"):
            try:
                df = load_analysis_results(".", pattern="analysis_results_")
                if not df.empty:
                    # Ensure datetime column is properly formatted
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Create line plot for each party
                    for party in df['party'].unique():
                        party_data = df[df['party'] == party].sort_values('datetime')
                        ax.plot(party_data['datetime'], party_data['overall_sentiment'], 
                               marker='o', label=party, linewidth=2, markersize=6)
                    
                    ax.set_ylabel('Overall Sentiment')
                    ax.set_title('Sentiment Trend Over Time')
                    ax.set_ylim(-1, 1)
                    ax.legend(title='Party')
                    ax.grid(True, linestyle='--', alpha=0.6)
                    
                    # Rotate x-axis labels
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Show the raw trend data
                    st.subheader("Raw Trend Data")
                    st.dataframe(df.sort_values('datetime', ascending=False))
                else:
                    st.warning("No trend data available.")
            except Exception as e:
                st.error(f"Error generating sentiment trends chart: {e}")
    else:
        st.warning("No analysis results available. Please run the main analysis first.")

# Tab for asking questions about the visualizations
with tab5:
    st.header("Ask Questions About Visualizations")
    
    if not csv_files and not analysis_files:
        st.warning("No data files found! Run the main analysis first to generate data.")
    else:
        # Combine CSVs into a list of rows for simple keyword search
        all_rows = []
        
        # Load CSV data
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                for idx, row in df.iterrows():
                    all_rows.append({"file": csv_file.name, "text": " ".join(row.astype(str).tolist())})
            except:
                pass  # Skip if CSV file can't be read
        
        # Also load analysis results for more detailed info
        for analysis_file in analysis_files:
            try:
                with open(analysis_file, 'r') as f:
                    data = json.load(f)
                
                party = data.get('party', 'Unknown')
                news_analysis = data.get('news_analysis', {})
                tweets_analysis = data.get('tweets_analysis', {})
                
                # Add summary info as searchable text
                summary = (
                    f"Party: {party}. News: {news_analysis.get('total', 0)} articles, "
                    f"Pos: {news_analysis.get('positive', 0)}, Neg: {news_analysis.get('negative', 0)}, "
                    f"Neutral: {news_analysis.get('neutral', 0)}. "
                    f"Avg sentiment: {news_analysis.get('average_polarity', 0):.3f}. "
                    f"Tweets: {tweets_analysis.get('total', 0)} tweets, "
                    f"Pos: {tweets_analysis.get('positive', 0)}, Neg: {tweets_analysis.get('negative', 0)}, "
                    f"Neutral: {tweets_analysis.get('neutral', 0)}. "
                    f"Avg sentiment: {tweets_analysis.get('average_polarity', 0):.3f}. "
                    f"Overall sentiment: {data.get('overall_sentiment', {}).get('combined_sentiment', 0):.3f}."
                )
                
                all_rows.append({"file": analysis_file.name, "text": summary})
            except:
                pass  # Skip if JSON file can't be read

        # Create vector store for RAG
        vectorstore = create_vectorstore(all_rows)
        
        if vectorstore:
            # Initialize LLM if API key is available
            if gemini_api_key:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_api_key)
                    
                    # Create a QA chain with the vector store and LLM
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(k=5),
                        return_source_documents=True
                    )
                    
                    # Streamlit input
                    user_question = st.text_input("Ask your question about political data, visualizations, or election forecast:", 
                                                placeholder="e.g., 'Which party has the most positive sentiment?', 'Show me BJP trends', 'Compare sentiment between parties'")
                    
                    if st.button("Get Answer", key="get_answer_tab5") and user_question:
                        with st.spinner("Processing your question with Gemini RAG..."):
                            try:
                                response = qa_chain.invoke({"query": user_question})
                                
                                if "result" in response:
                                    st.markdown("**Answer (from your data):**")
                                    st.markdown(response["result"])
                                    
                                    # Show source documents if available
                                    if "source_documents" in response:
                                        with st.expander("Show source documents"):
                                            for i, doc in enumerate(response["source_documents"]):
                                                st.markdown(f"**Source {i+1}:** {doc.metadata.get('source_file', 'Unknown')}")
                                                st.markdown(f"{doc.page_content[:500]}...")
                                else:
                                    st.error("No response generated. Please try again.")
                                    
                            except Exception as e:
                                st.error(f"Error processing your question: {str(e)}")
                                # Fallback to manual search if RAG fails
                                st.markdown("**Raw search results:**")
                                for r in all_rows[:5]:
                                    st.markdown(f"- {r['file']}: {r['text'][:200]}...")
                except Exception as e:
                    st.error(f"Gemini API initialization failed: {str(e)}")
                    # Fallback to basic search
                    st.markdown("**Using basic search instead:**")
                    user_question = st.text_input("Ask your question about political data, visualizations, or election forecast:", 
                                                placeholder="e.g., 'Which party has the most positive sentiment?', 'Show me BJP trends', 'Compare sentiment between parties'")
                    
                    if st.button("Get Answer", key="get_answer_tab5_fallback") and user_question:
                        with st.spinner("Searching for answer..."):
                            # Simple keyword search as fallback
                            question_lower = user_question.lower()
                            matches = []
                            for r in all_rows:
                                if any(keyword in r["text"].lower() for keyword in question_lower.split()):
                                    matches.append(f"{r['file']}: {r['text'][:200]}...")
                            
                            if matches:
                                st.markdown("**Answer (from your data):**")
                                for m in matches[:5]:
                                    st.markdown(f"- {m}")
                            else:
                                st.markdown("No matching data found.")
            else:
                st.warning("GEMINI_API_KEY not found in environment. Using basic search instead.")
                # Basic search functionality
                user_question = st.text_input("Ask your question about political data, visualizations, or election forecast:", 
                                            placeholder="e.g., 'Which party has the most positive sentiment?', 'Show me BJP trends', 'Compare sentiment between parties'")
                
                if st.button("Get Answer", key="get_answer_tab5_basic") and user_question:
                    with st.spinner("Searching for answer..."):
                        # Simple keyword search
                        question_lower = user_question.lower()
                        matches = []
                        for r in all_rows:
                            if any(keyword in r["text"].lower() for keyword in question_lower.split()):
                                matches.append(f"{r['file']}: {r['text'][:200]}...")
                        
                        if matches:
                            st.markdown("**Answer (from your data):**")
                            for m in matches[:5]:
                                st.markdown(f"- {m}")
                        else:
                            st.markdown("No matching data found.")
        else:
            st.warning("No data available for RAG. Please run the main analysis first to generate data.")
    
    if not csv_files and not analysis_files:
        st.warning("No data files found! Run the main analysis first to generate data.")
    else:
        # Combine CSVs into a list of rows for simple keyword search
        all_rows = []
        
        # Load CSV data
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                for idx, row in df.iterrows():
                    all_rows.append({"file": csv_file.name, "text": " ".join(row.astype(str).tolist())})
            except:
                pass  # Skip if CSV file can't be read
        
        # Also load analysis results for more detailed info
        for analysis_file in analysis_files:
            try:
                with open(analysis_file, 'r') as f:
                    data = json.load(f)
                
                party = data.get('party', 'Unknown')
                news_analysis = data.get('news_analysis', {})
                tweets_analysis = data.get('tweets_analysis', {})
                
                # Add summary info as searchable text
                summary = ("Party: {party}. News: {total_articles} articles, "
                         "Pos: {pos_articles}, Neg: {neg_articles}, "
                         "Neutral: {neut_articles}. "
                         "Tweets: {total_tweets} tweets, "
                         "Pos: {pos_tweets}, Neg: {neg_tweets}, "
                         "Neutral: {neut_tweets}.").format(
                            party=party,
                            total_articles=news_analysis.get("total", 0),
                            pos_articles=news_analysis.get("positive", 0),
                            neg_articles=news_analysis.get("negative", 0),
                            neut_articles=news_analysis.get("neutral", 0),
                            total_tweets=tweets_analysis.get("total", 0),
                            pos_tweets=tweets_analysis.get("positive", 0),
                            neg_tweets=tweets_analysis.get("negative", 0),
                            neut_tweets=tweets_analysis.get("neutral", 0)
                        )
                
                all_rows.append({"file": analysis_file.name, "text": summary})
            except:
                pass  # Skip if JSON file can't be read

        # Create vector store for RAG
        vectorstore = create_vectorstore(all_rows)
        
        if vectorstore:
            # Initialize LLM if API key is available
            if gemini_api_key:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_api_key)
                    
                    # Create a QA chain with the vector store and LLM
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(k=5),
                        return_source_documents=True
                    )
                    
                    # Streamlit input
                    user_question = st.text_input("Ask your question about political data, visualizations, or election forecast:")

                    if st.button("Get Answer", key="get_answer_tab2_old") and user_question:
                        with st.spinner("Processing your question with Gemini RAG..."):
                            try:
                                response = qa_chain.invoke({"query": user_question})
                                
                                if "result" in response:
                                    st.markdown("**Answer (from your data):**")
                                    st.markdown(response["result"])
                                    
                                    # Show source documents if available
                                    if "source_documents" in response:
                                        with st.expander("Show source documents"):
                                            for i, doc in enumerate(response["source_documents"]):
                                                st.markdown(f"**Source {i+1}:** {doc.metadata.get('source_file', 'Unknown')}")
                                                st.markdown(f"{doc.page_content[:500]}...")
                                else:
                                    st.error("No response generated. Please try again.")
                                    
                            except Exception as e:
                                st.error(f"Error processing your question: {str(e)}")
                                # Fallback to manual search if RAG fails
                                st.markdown("**Raw search results:**")
                                for r in all_rows[:5]:
                                    st.markdown(f"- {r['file']}: {r['text'][:200]}...")
                except Exception as e:
                    st.error(f"Gemini API initialization failed: {str(e)}")
                    # Fallback to basic search
                    st.markdown("**Using basic search instead:**")
                    user_question = st.text_input("Ask your question about political data, visualizations, or election forecast:")
                    
                    if st.button("Get Answer", key="get_answer_tab2_old_fallback") and user_question:
                        with st.spinner("Searching for answer..."):
                            # Simple keyword search as fallback
                            question_lower = user_question.lower()
                            matches = []
                            for r in all_rows:
                                if any(keyword in r["text"].lower() for keyword in question_lower.split()):
                                    matches.append(f"{r['file']}: {r['text'][:200]}...")
                            
                            if matches:
                                st.markdown("**Answer (from your data):**")
                                for m in matches[:5]:
                                    st.markdown(f"- {m}")
                            else:
                                st.markdown("No matching data found.")
            else:
                st.warning("GEMINI_API_KEY not found in environment. Using basic search instead.")
                # Basic search functionality
                user_question = st.text_input("Ask your question about political data, visualizations, or election forecast:")
                
                if st.button("Get Answer", key="get_answer_tab2_old_basic") and user_question:
                    with st.spinner("Searching for answer..."):
                        # Simple keyword search
                        question_lower = user_question.lower()
                        matches = []
                        for r in all_rows:
                            if any(keyword in r["text"].lower() for keyword in question_lower.split()):
                                matches.append(f"{r['file']}: {r['text'][:200]}...")
                        
                        if matches:
                            st.markdown("**Answer (from your data):**")
                            for m in matches[:5]:
                                st.markdown(f"- {m}")
                        else:
                            st.markdown("No matching data found.")
        else:
            st.warning("No data available for RAG. Please run the main analysis first to generate data.")
