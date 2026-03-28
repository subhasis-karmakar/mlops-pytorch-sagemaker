import os
import boto3
from botocore.exceptions import ClientError

from sagemaker.session import Session
from sagemaker.pytorch.model import PyTorchModel


REGION = "us-west-2"
ROLE_ARN = "arn:aws:iam::628479576048:role/SageMakerExecutionRole"

ENDPOINT_NAME = "pytorch-mlops-endpoint-direct"
INSTANCE_TYPE = "ml.m5.large"
INITIAL_INSTANCE_COUNT = 1

MODEL_DATA = os.getenv("MODEL_DATA")
FRAMEWORK_VERSION = "1.12"
PY_VERSION = "py38"

boto_session = boto3.Session(region_name=REGION)
sm = boto_session.client("sagemaker")
sm_session = Session(boto_session=boto_session)


def delete_endpoint_if_exists(endpoint_name: str) -> None:
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        print(f"Endpoint exists: {endpoint_name}. Deleting.")
        sm.delete_endpoint(EndpointName=endpoint_name)

        waiter = sm.get_waiter("endpoint_deleted")
        waiter.wait(EndpointName=endpoint_name)
        print(f"Deleted endpoint: {endpoint_name}")
    except ClientError as e:
        if "Could not find endpoint" not in str(e):
            raise


def delete_endpoint_config_if_exists(endpoint_config_name: str) -> None:
    try:
        sm.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        print(f"Endpoint config exists: {endpoint_config_name}. Deleting.")
        sm.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    except ClientError as e:
        if "Could not find endpoint configuration" not in str(e):
            raise


def delete_model_if_exists(model_name: str) -> None:
    try:
        sm.describe_model(ModelName=model_name)
        print(f"Model exists: {model_name}. Deleting.")
        sm.delete_model(ModelName=model_name)
    except ClientError as e:
        if "Could not find model" not in str(e):
            raise


if __name__ == "__main__":
    if not MODEL_DATA:
        raise ValueError(
            "MODEL_DATA environment variable is required. "
            "Set it to the S3 path of model.tar.gz from a successful training job."
        )

    model_name = f"{ENDPOINT_NAME}-model"

    delete_endpoint_if_exists(ENDPOINT_NAME)
    delete_endpoint_config_if_exists(f"{ENDPOINT_NAME}-config")
    delete_model_if_exists(model_name)

    model = PyTorchModel(
        model_data=MODEL_DATA,
        role=ROLE_ARN,
        entry_point="src/inference.py",
        source_dir=".",
        framework_version=FRAMEWORK_VERSION,
        py_version=PY_VERSION,
        sagemaker_session=sm_session,
        name=model_name,
    )

    predictor = model.deploy(
        initial_instance_count=INITIAL_INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
    )

    print(f"Deployment complete. Endpoint name: {ENDPOINT_NAME}")