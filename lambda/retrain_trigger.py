import boto3, os

def lambda_handler(event, context):
    sagemaker_client = boto3.client("sagemaker")
    pipeline_name = os.environ.get("PIPELINE_NAME", "PyTorchMLOpsPipeline")

    response = sagemaker_client.start_pipeline_execution(PipelineName=pipeline_name)
    print("Triggered pipeline execution:", response["PipelineExecutionArn"])
    return {"status": "success", "executionArn": response["PipelineExecutionArn"]}