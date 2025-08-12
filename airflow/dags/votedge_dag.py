from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from codes.data_collector import collect_all_data
from codes.llm_processor import process_data_with_llm
from codes.feature_engineer import engineer_features
from codes.model_trainer import train_model
from codes.visualizer import create_visualizations

# Default arguments for the DAG
default_args = {
    'owner': 'votedge',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'votedge_pipeline',
    default_args=default_args,
    description='Election prediction pipeline',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['votedge', 'election', 'prediction'],
)

# Define tasks
start_pipeline = DummyOperator(
    task_id='start_pipeline',
    dag=dag,
)

collect_data = PythonOperator(
    task_id='collect_data',
    python_callable=collect_all_data,
    dag=dag,
)

process_data = PythonOperator(
    task_id='process_data',
    python_callable=process_data_with_llm,
    dag=dag,
)

engineer_features = PythonOperator(
    task_id='engineer_features',
    python_callable=engineer_features,
    dag=dag,
)

train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

create_visualizations = PythonOperator(
    task_id='create_visualizations',
    python_callable=create_visualizations,
    dag=dag,
)

end_pipeline = DummyOperator(
    task_id='end_pipeline',
    dag=dag,
)

# Set task dependencies
start_pipeline >> collect_data >> process_data >> engineer_features >> train_model >> create_visualizations >> end_pipeline