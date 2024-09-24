# Components that need to be created and registered:


# - Data loading component (Loads data generators, default or custom from datastore)
# Michał

# - Data preprocessing component (Preprocesses images - crops, resizes, patchifies and splits into train/tesst/val)

# - Data registration component (Uploads preprocessed data into the datastore )

# - Model creation component (Builds a new U-Net model)

# - Model training component (Trains a model, # either on default or custom data if provided) # and either with default or custom hyperparameters set if provided)

# - Hyperparameter search component (Trains multiple models with different hyperparameters and returns the best one)

# - Model evaluation component (Evaluates a model on test data) Stijn

# - Model registration component (Registers a model in the model registry) Stijn

# - Mask prediction component (Given image and model, predicts mask for the image)

# - Instance segmentation component (Given mask, segments instances) Stijn

# - Landmarks detection component (Given segmented mask, detects landmarks for each plant)

# - Root length detection component (Given segmented mask, detects primary and lateral root lengths for each plant)


import os
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command, dsl
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
env = ml_client.environments.get("NewCV3environment3", 15)


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
        # "input_model_path": Input(type="uri_folder", description="Path to save the model"),
        "learning_rate": Input(
            type="number", description="Learning rate", default=1e-3
        ),
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
    command="python model_training.py --dataset-path ${{inputs.dataset_path}} --learning-rate ${{inputs.learning_rate}} --history-path ${{outputs.history}} --model-path ${{outputs.model_path}}",
    environment=env,
)


mask_prediction_component = command(
    name="mask_prediction",
    display_name="Mask prediction component",
    description="Given image and model, predicts mask for the image",
    inputs={
        "model_path": Input(type="uri_folder", description="Path to the model"),
        "input_img_path": Input(
            type="uri_folder", description="Path to the input image"
        ),
    },
    outputs={
        "output_mask_path": Output(
            type="uri_folder", mode="rw_mount", description="Path to the output mask"
        ),
    },
    code="src/",
    command="python mask_prediction.py --model-path ${{inputs.model_path}} --input-img-path ${{inputs.input_img_path}} --output-mask-path ${{outputs.output_mask_path}}",
    environment=env,
)


landmarks_detection_component = command(
    name="landmarks_detection",
    display_name="Landmarks detection component",
    description="Given segmented mask, detects landmarks for each plant",
    inputs={
        "mask_path": Input(type="uri_folder", description="Path to the mask folder"),
    },
    outputs={
        "landmarks_path": Output(
            type="uri_folder", description="Path to the output landmarks folder"
        ),
    },
    code="src/",
    command="python landmarks_detection.py --mask-path ${{inputs.mask_path}} --landmarks-path ${{outputs.landmarks_path}}",
    environment=env,
)

root_length_measurement_component = command(
    name="root_length_measurement",
    display_name="Root length measurement component",
    description="Given segmented mask, detects primary and lateral root lengths for each plant",
    inputs={
        "mask_path": Input(type="uri_folder", description="Path to the mask folder"),
    },
    outputs={
        "output_path": Output(
            type="uri_folder", description="Path to the output root lengths folder"
        ),
    },
    code="src/",
    command="python root_length_measurement.py --mask-path ${{inputs.mask_path}} --output-path ${{outputs.output_path}}",
    environment=env,
)

model_evaluation_component = command(
    name="model_evaluation",
    display_name="Model evaluation component",
    description="Evaluates a model on test data",
    inputs={
        "model_path": Input(type="uri_folder", description="Path to the model"),
        "test_dataset_path": Input(type="uri_folder"),
    },
    outputs={
        "evaluation_results": Output(
            type="uri_folder", description="Path to the evaluation results"
        ),
    },
    code="src/",
    command="python model_evaluation.py --model-path ${{inputs.model_path}} --test-dataset-path ${{inputs.test_dataset_path}} --evaluation-results ${{outputs.evaluation_results}}",
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
        # "metrics_path": Input(type="uri_folder", description="Path to the metrics"),
        # "threshold": Input(type="number", description="Threshold", default=None),
    },
    outputs={},
    code="src/",
    # command="python model_registration.py --model-path ${{inputs.model_path}} --metrics-path ${{inputs.metrics_path}} --model-name ${{inputs.model_name}} --threshold ${{inputs.threshold}}",
    command="python model_registration.py --model-path ${{inputs.model_path}} --model-name ${{inputs.model_name}}",
    environment=env,
)


@dsl.pipeline(
    name="Inference pipeline",
    instance_type="defaultinstancetype",
    compute="adsai1",
)
def inference_pipeline(
    model_path: str,
    input_img_path: str,
):
    mask_prediction_step = mask_prediction_component(
        model_path=model_path,
        input_img_path=input_img_path,
    )

    landmarks_detection_step = landmarks_detection_component(
        mask_path=mask_prediction_step.outputs["output_mask_path"],
    )

    root_length_measurement_step = root_length_measurement_component(
        mask_path=mask_prediction_step.outputs["output_mask_path"],
    )

    return {
        "landmarks": landmarks_detection_step.outputs["landmarks_path"],
        "root_lengths": root_length_measurement_step.outputs["output_path"],
    }


@dsl.pipeline(
    name="Model training pipeline",
    instance_type="defaultinstancetype",
    compute="adsai1",
)
def model_training_pipeline(
    model_name: str,
    # raw_dataset_path: str = None,
    preprocessed_dataset_path: str = None,
    learning_rate: float = 1e-3,
):
    # if raw_dataset_path is None and preprocessed_dataset_path is None:
    #     raise ValueError("Either raw or preprocessed dataset path must be provided")

    # if raw_dataset_path:
    #     data_preprocessing_step = data_preprocessing_component(
    #         dataset_path=raw_dataset_path,
    #         test_size=0.2,
    #         val_size=0.1,
    #     )
    #     preprocessed_dataset_path = data_preprocessing_step.outputs[
    #         "preprocessed_dataset_path"
    #     ]

    model_training_step = model_training_component(
        dataset_path=preprocessed_dataset_path,
        learning_rate=learning_rate,
    )

    model_registration_component(
        model_path=model_training_step.outputs["model_path"],
        model_name=model_name,
    )

    return {
        "model_path": model_training_step.outputs["model_path"],
        "history": model_training_step.outputs["history"],
    }


# # # Model_name (string)
# model_name = "TEST_MODEL_1"


# model_training_pipeline_instance = model_training_pipeline(
#     model_name=model_name,
#     preprocessed_dataset_path=Input(
#         path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/buas-y2/workspaces/CV3/datastores/workspaceblobstore/paths/data/testing_dataset/"
#     ),
#     learning_rate=1e-3,
# )

# # Submit the pipeline.
# pipeline_run = ml_client.jobs.create_or_update(model_training_pipeline_instance)


# inference_pipeline_instance = inference_pipeline(
#     model_path=Input(
#         path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/buas-y2/workspaces/CV3/datastores/workspaceblobstore/paths/models/primary_model/"
#     ),
#     input_img_path=Input(
#         path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/buas-y2/workspaces/CV3/datastores/workspaceblobstore/paths/UI/2024-06-10_104533_UTC/example_image/"
#     ),
# )

# # Submit the pipeline.
# pipeline_run = ml_client.jobs.create_or_update(inference_pipeline_instance)


# Pipeline to test the evaluation component
@dsl.pipeline(
    name="Model evaluation pipeline",
    instance_type="defaultinstancetype",
    compute="adsai1",
)
def model_evaluation_pipeline(
    model_path: str,
    test_dataset_path: str,
):
    model_evaluation_step = model_evaluation_component(
        model_path=model_path,
        test_dataset_path=test_dataset_path,
    )

    return {
        "evaluation_results": model_evaluation_step.outputs["evaluation_results"],
    }


model_evaluation_pipeline_instance = model_evaluation_pipeline(
    model_path=Input(
        path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/buas-y2/workspaces/CV3/datastores/workspaceblobstore/paths/models/primary_model/"
    ),
    test_dataset_path=Input(
        path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/buas-y2/workspaces/CV3/datastores/workspaceblobstore/paths/data/testing_dataset/test/"
    ),
)

pipeline_run = ml_client.jobs.create_or_update(model_evaluation_pipeline_instance)
