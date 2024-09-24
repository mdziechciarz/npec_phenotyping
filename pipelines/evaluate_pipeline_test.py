from azure.ai.ml import MLClient, Input, Output, dsl
from azure.identity import InteractiveBrowserCredential

# Create a browser credential
credential = InteractiveBrowserCredential()

# Define your Azure subscription details
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

# Get the registered component
model_evaluation_component = ml_client.components.get(name="model_evaluation", version="1")

# Define the pipeline
@dsl.pipeline(
    name="model-evaluation-pipeline",
    description="A pipeline to evaluate a model",
)
def evaluate_model_pipeline(
    model: Input,
    test_data: Input,
):
    evaluation_step = model_evaluation_component(
        model=model,
        test_data=test_data,
        evaluation_results=Output(type="uri_folder", path="azureml://datastores/workspaceblobstore/paths/evaluation_results/")
    )

    return {"evaluation_results": evaluation_step.outputs.evaluation_results}

# Create the pipeline
pipeline = evaluate_model_pipeline(
    model=Input(type="uri_file", path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/buas-y2/workspaces/CV3/datastores/workspaceblobstore/paths/models/primary.h5"),
    test_data=Input(type="uri_folder", path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/buas-y2/workspaces/CV3/datastores/workspaceblobstore/paths/data/testing_dataset/"),
)

# Submit the pipeline
pipeline_run = ml_client.jobs.create_or_update(pipeline)
print(f"Pipeline submitted with ID: {pipeline_run.id}")
