from datetime import datetime

from airflow.operators.empty import EmptyOperator

from airflow import DAG

# Define the default arguments
default_args = {
    # the owner attribute will display on home screen. Recommend using your group name or your own name for dags on the server.
    "owner": "airflow",
    # Depends on past will allow your dag to backfill historical data. Leave this as false for this project.
    "depends_on_past": False,
    # Start data is the start date of your dag's history. It will define when the backfill will start.
    # Due to some intracacies with the way scheduling works this needs to be more than one interval in the past.
    # For this project leaving it as 01/01/2024 is recommended
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    # Global setting for the number of retry attempts for failed tasks.
    "retries": 1,
}

# Instantiate the DAG
with DAG(
    dag_id="my_test_dag",
    default_args=default_args,
    description="A simple test DAG with dummy operators",
    schedule_interval=None,
) as dag:

    # Define the start task
    start = EmptyOperator(task_id="start")

    # Define an empty task
    empty = EmptyOperator(task_id="parrallel_empty_1")

    empty_2 = EmptyOperator(task_id="parrallel_empty_2")

    # Define the finish task
    finish = EmptyOperator(task_id="finish")

    # Set the task dependencies
    start >> [empty, empty_2] >> finish
