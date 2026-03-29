import os
import boto3

REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
PIPELINE_NAME = os.getenv("PIPELINE_NAME", "PyTorchMLOpsPipeline")
ACCURACY_THRESHOLD = os.getenv("ACCURACY_THRESHOLD", "0.50")

sm = boto3.client("sagemaker", region_name=REGION)


def lambda_handler(event, context):
    response = sm.start_pipeline_execution(
        PipelineName=PIPELINE_NAME,
        PipelineParameters=[
            {
                "Name": "AccuracyThreshold",
                "Value": ACCURACY_THRESHOLD,
            }
        ],
    )

    return {
        "statusCode": 200,
        "pipeline_execution_arn": response["PipelineExecutionArn"],
        "event": event,
    }