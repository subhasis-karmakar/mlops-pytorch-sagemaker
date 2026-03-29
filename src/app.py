from fastapi import FastAPI
from pydantic import BaseModel
import boto3
import json

app = FastAPI()
runtime = boto3.client("sagemaker-runtime", region_name="us-west-2")

ENDPOINT_NAME = "pytorch-mlops-registry-endpoint"


class PredictRequest(BaseModel):
    instances: list


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(req.dict()),
    )
    return json.loads(response["Body"].read().decode("utf-8"))