# Airflow Integration for VotEdge

This directory contains the Apache Airflow workflow orchestration for the VotEdge election prediction system.

## Overview

Apache Airflow is used to automate the daily election prediction workflow, including:
- Collecting news data for all configured political parties
- Scraping tweets from party social media accounts
- Performing sentiment analysis on collected data
- Updating election predictions based on sentiment analysis
- Generating visualizations of the predictions

## Directory Structure

```
airflow/
├── dags/                    # DAG definition files
│   └── election_analysis_dag.py
├── logs/                    # Airflow execution logs
├── airflow.cfg              # Airflow configuration
├── requirements.txt         # Airflow dependencies
└── README.md               # This file
```

## Setting Up Airflow

1. **Install Airflow and dependencies:**
   ```bash
   cd airflow
   pip install -r requirements.txt
   ```

2. **Initialize the Airflow database:**
   ```bash
   airflow db init
   ```

3. **Create an Airflow user (optional, for web UI access):**
   ```bash
   airflow users create \
       --username admin \
       --firstname Admin \
       --lastname User \
       --role Admin \
       --email admin@example.com
   ```

4. **Start the Airflow services:**
   - Terminal 1 (Scheduler):
     ```bash
     airflow scheduler
     ```
   - Terminal 2 (Web Server):
     ```bash
     airflow webserver --port 8080
     ```

5. **Access the Airflow UI:**
   - Open your browser and go to `http://localhost:8080`
   - Use the credentials created in step 3

## DAG Details

The `election_analysis_dag.py` file defines the workflow with the following tasks:

1. **collect_and_analyze_data**: Collects news and social media data for all configured parties and performs sentiment analysis
2. **update_election_predictions**: Updates the election predictions based on the sentiment analysis
3. **generate_visualizations**: Creates visualizations of the updated predictions

The DAG is scheduled to run daily at the specified interval.

## Configuration

- Ensure your `.env` file contains the necessary API keys (NEWSAPI_KEY)
- Make sure the `src/` directory with the VotEdge modules is accessible
- The DAG will use the party configuration from the existing VotEdge system

## Monitoring

- Monitor the workflow execution through the Airflow web UI
- Check logs in the `logs/` directory for detailed execution information
- Set up email notifications in the Airflow configuration if needed