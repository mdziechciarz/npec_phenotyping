import os
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command, dsl
from azure.ai.ml.sweep import Choice, Uniform
from azure.identity import InteractiveBrowserCredential

# Create a browser credential
credential = InteractiveBrowserCredential()

# subscription_id, resource_group, and workspace_name are strings
subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
resource_group = "buas-y2"
workspace_name = "CV3"

# Create an MLClient using the credential and workspace details
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

# Get environment
env = ml_client.environments.get("NewCV3environment3", 16)

data_preprocessing_component = command(
    name="data_preprocessing",
    display_name="Data preprocessing component",
    description="Preprocesses images - crops, resizes, patchifies and splits into train/test/val",
    inputs={
        "dataset_path": Input(type="uri_folder", description="Path to the dataset"),
        "test_size": Input(type="number", description="Test size"),
        "val_size": Input(type="number", description="Validation size"),
    },
    outputs={
        "preprocessed_dataset_path": Output(
            type="uri_folder",
            mode="rw_mount",
            description="Path to the preprocessed dataset",
        ),
    },
    code="src/",
    command="python data_preprocessing.py --dataset-path ${{inputs.dataset_path}} --preprocessed-dataset-path ${{outputs.preprocessed_dataset_path}} --test-size ${{inputs.test_size}} --val-size ${{inputs.val_size}}",
    environment=env,
)


model_creation_component = command(
    name="model_creation",
    display_name="Model creation component",
    description="Builds a new U-Net model",
    inputs={
        "model_path": Input(type="uri_folder", description="Path to save the model"),
        "learning_rate": Input(
            type="number", description="Learning rate", default=1e-3
        ),
    },
    outputs={
        "model": Output(
            type="uri_folder", mode="rw_mount", description="Path to the model"
        ),
    },
    code="src/",
    command="python model_creation.py --model-path ${{inputs.model_path}} --learning-rate ${{inputs.learning_rate}} --model ${{outputs.model}}",
    environment=env,
)


model_training_component = command(
    name="model_training",
    display_name="Model training component",
    description="Trains a model",
    inputs={
        "dataset_path": Input(type="uri_folder", description="Path to the dataset"),
        "input_model_path": Input(
            type="uri_folder", optional=True, description="Path of model to retrain"
        ),
        "learning_rate": Input(
            type="number", description="Learning rate", default=1e-3
        ),
        "batch_size": Input(type="number", description="Batch size", default=16),
    },
    outputs={
        "model_path": Output(
            type="uri_folder", mode="rw_mount", description="Path to the model"
        ),
        "history": Output(
            type="uri_folder", mode="rw_mount", description="Training history"
        ),
    },
    code="src/",
    command="python model_training.py --dataset-path ${{inputs.dataset_path}} --learning-rate ${{inputs.learning_rate}} --history-path ${{outputs.history}} --model-path ${{outputs.model_path}} --batch-size ${{inputs.batch_size}} $[[--input-model-path ${{inputs.input_model_path}}]]",
    environment=env,
)


model_evaluation_component = command(
    name="model_evaluation",
    display_name="Model evaluation component",
    description="Evaluates a model on test data",
    inputs={
        "model_path": Input(type="uri_folder", description="Path to the model"),
        "dataset_path": Input(type="uri_folder"),
    },
    outputs={
        "evaluation_results": Output(
            type="uri_folder", description="Path to the evaluation results"
        ),
    },
    code="src/",
    command="python model_evaluation.py --model-path ${{inputs.model_path}} --dataset-path ${{inputs.dataset_path}} --evaluation-results ${{outputs.evaluation_results}}",
    environment=env,
)

model_registration_component = command(
    name="model_registration",
    display_name="Model registration component",
    description="Registers a model in the model registry",
    inputs={
        "model_path": Input(type="uri_folder", description="Path to the model"),
        "model_name": Input(
            type="string",
            description="Name of the model",
        ),
        "metrics_path": Input(type="uri_folder", description="Path to the metrics"),
        "threshold": Input(type="number", description="Threshold", default=None),
    },
    outputs={},
    code="src/",
    command="python model_registration.py --model-path ${{inputs.model_path}} --metrics-path ${{inputs.metrics_path}} --model-name ${{inputs.model_name}} --threshold ${{inputs.threshold}}",
    environment=env,
)


@dsl.pipeline(
    name="Model training pipeline after including optional input model (WITHOUT INPUT MODEL)",
    instance_type="gpu",
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


# Example usage
# model_training_pipeline_instance = model_training_pipeline(
#     model_name="Retrained_model",
#     preprocessed_dataset_path=Input(
#         path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/buas-y2/workspaces/CV3/datastores/workspaceblobstore/paths/data/testing_dataset/"
#     ),
#     threshold=0.1,
# )

# # Submit the pipeline.
# pipeline_run = ml_client.jobs.create_or_update(model_training_pipeline_instance)
