from datetime import datetime, timedelta

from airflow.operators.python_operator import PythonOperator

from airflow import DAG

# Define the default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 6, 18),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# Define the function to be executed
def trigger_training():
    import logging

    # logging.basicConfig(level=logging.warning)

    logging.warning("RUNNING THE SCRIPT")
    logging.warning("IMPORTING LIBRARIES")

    from azure.ai.ml import Input, MLClient, dsl
    from azure.ai.ml.sweep import Choice, Uniform
    from azure.identity import ClientSecretCredential

    logging.warning("IMPORTED LIBRARIES")

    subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
    resource_group = "buas-y2"
    workspace_name = "CV3"
    tenant_id = "0a33589b-0036-4fe8-a829-3ed0926af886"
    client_id = "27157a5a-3927-4895-8478-9d4554697d25"
    client_secret = "stf8Q~mP2cB923Mvz5K91ITcoYgvRXs4J1lysbfb"

    credential = ClientSecretCredential(tenant_id, client_id, client_secret)
    logging.warning("Got the credential")

    # Create an MLClient using the credential and workspace details
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
    logging.warning("Got the MLClient")

    # Load the registered components
    model_training_component = ml_client.components.get("model_training")
    model_evaluation_component = ml_client.components.get("model_evaluation")
    model_registration_component = ml_client.components.get("model_registration")
    logging.warning("GOT ALL COMPONENTS")

    @dsl.pipeline(
        name="Scheduled training pipeline",
        instance_type="defaultinstancetype",
        compute="adsai1",
    )
    def model_training_pipeline(
        model_name: str,
        preprocessed_dataset_path: str = None,
        threshold: float = 0.5,
    ):
        model_training_step = model_training_component(
            dataset_path=preprocessed_dataset_path,
            learning_rate=Uniform(min_value=1e-5, max_value=1e-2),
            batch_size=Choice([16, 32, 64, 128]),
        ).sweep(
            sampling_algorithm="bayesian",
            primary_metric="val_loss",
            goal="minimize",
        )
        model_training_step.set_limits(
            max_total_trials=3, max_concurrent_trials=3, timeout=7200
        )

        model_evaluation_step = model_evaluation_component(
            model_path=model_training_step.outputs["model_path"],
            dataset_path=preprocessed_dataset_path,
        )

        model_registration_component(
            model_path=model_training_step.outputs["model_path"],
            model_name=model_name,
            metrics_path=model_evaluation_step.outputs["evaluation_results"],
            threshold=threshold,
        )

        return {
            "model_path": model_training_step.outputs["model_path"],
            "history": model_training_step.outputs["history"],
        }

    logging.warning("CREATED PIPELINE")

    # Define your pipeline instance and parameters here
    model_training_pipeline_instance = model_training_pipeline(
        model_name="model_from_scheduled_training",
        preprocessed_dataset_path=Input(
            path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/buas-y2/workspaces/CV3/datastores/workspaceblobstore/paths/data/testing_dataset/"
        ),
        threshold=0.01,
    )
    # Submit the pipeline
    ml_client.jobs.create_or_update(model_training_pipeline_instance)
    logging.warning("SUBMITTED PIPELINE JOB")


# Define the DAG
dag = DAG(
    "scheduled_training_dag",
    default_args=default_args,
    description="DAG to schedule the training pipeline",
    schedule_interval=timedelta(days=1),
)

# Define the task using PythonOperator
schedule_training_task = PythonOperator(
    task_id="trigger_training",
    python_callable=trigger_training,
    dag=dag,
)

# Define the task dependencies (if any)
schedule_training_task
