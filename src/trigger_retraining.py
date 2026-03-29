import boto3
import os

REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
PIPELINE_NAME = os.getenv("PIPELINE_NAME", "PyTorchMLOpsPipeline")

sm = boto3.client("sagemaker", region_name=REGION)

def lambda_handler(event, context):
    response = sm.start_pipeline_execution(
        PipelineName=PIPELINE_NAME,
        PipelineParameters=[
            {
                "Name": "AccuracyThreshold",
                "Value": "0.50"
            }
        ],
    )
    return {
        "statusCode": 200,
        "pipeline_execution_arn": response["PipelineExecutionArn"],
    }