import argparse
import json
import logging
import os

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import ClientSecretCredential, InteractiveBrowserCredential

# Create a browser credential
credential = InteractiveBrowserCredential()

# subscription_id, resource_group, and workspace_name are strings
subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
resource_group = "buas-y2"
workspace_name = "CV3"
tenant_id = "0a33589b-0036-4fe8-a829-3ed0926af886"
client_id = "27157a5a-3927-4895-8478-9d4554697d25"
client_secret = "stf8Q~mP2cB923Mvz5K91ITcoYgvRXs4J1lysbfb"


# Register model in AzureML model registry
def register_model(model_path, model_name):
    """
    Register a model in the AzureML model registry.

    This function registers a model in the AzureML model registry using the provided model path and name.
    It uses `ClientSecretCredential` for authentication and `MLClient` to interact with the AzureML workspace.

    :param model_path: Path to the model file.
    :type model_path: str
    :param model_name: Name to register the model under.
    :type model_name: str
    :return: None
    :rtype: None
    :author: Lea Banovac

    :Example:

    .. code-block:: python

        register_model("/path/to/model", "my_model_name")

    :Note:
        Ensure that the model file exists at the specified path and the credentials are correctly set up.
    """
    credential = ClientSecretCredential(tenant_id, client_id, client_secret)

    # Create an MLClient using the credential and workspace details
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

    model = Model(
        path=model_path,
        name=model_name,
        type=AssetTypes.CUSTOM_MODEL,
        description="Model created from pipeline",
    )

    model = ml_client.models.create_or_update(model)

    logging.info(
        f"Model succesfully registered in Azure Model Registry, model_name: {model_name}"
    )


def register_if_IOU_above_threshold(model_path, model_name, metrics_path, threshold):
    with open(os.path.join(metrics_path, "metrics.json"), "r") as file:
        metrics = json.load(file)

        # Check if the IOU is above the threshold
        if metrics["iou"] > threshold:
            register_model(model_path, model_name)
        else:
            logging.warning("IOU is below the threshold, model not registered")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--metrics-path", type=str)
    parser.add_argument("--threshold", type=float)

    args = parser.parse_args()

    if args.threshold:
        register_if_IOU_above_threshold(
            args.model_path, args.model_name, args.metrics_path, args.threshold
        )
    else:
        register_model(args.model_path, args.model_name)
