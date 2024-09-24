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
env = ml_client.environments.get("NewCV3environment3", 11)


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


# USAGE EXAMPLE

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
