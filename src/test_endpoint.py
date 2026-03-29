import json
import boto3

REGION = "us-west-2"
ENDPOINT_NAME = "pytorch-mlops-registry-endpoint"

runtime = boto3.client("sagemaker-runtime", region_name=REGION)

with open("payload.json", "r") as f:
    payload = f.read()

response = runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Body=payload,
)

body = response["Body"].read().decode("utf-8")
print(body)