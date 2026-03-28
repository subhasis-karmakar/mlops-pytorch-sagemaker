import os
import boto3
from botocore.exceptions import ClientError

REGION = "us-west-2"
ROLE_ARN = "arn:aws:iam::628479576048:role/SageMakerExecutionRole"

MODEL_PACKAGE_GROUP_NAME = "PyTorchMLOpsModelGroup"

MODEL_NAME = "pytorch-mlops-registry-model"
ENDPOINT_CONFIG_NAME = "pytorch-mlops-registry-endpoint-config"
ENDPOINT_NAME = "pytorch-mlops-registry-endpoint"

INSTANCE_TYPE = "ml.m5.large"
INITIAL_INSTANCE_COUNT = 1

# Optional selectors
MODEL_PACKAGE_ARN = os.getenv("MODEL_PACKAGE_ARN")
MODEL_VERSION = os.getenv("MODEL_VERSION")
ONLY_APPROVED = True

sm = boto3.client("sagemaker", region_name=REGION)


def get_model_package_by_version(model_package_group_name: str, version: int) -> str:
    paginator = sm.get_paginator("list_model_packages")
    for page in paginator.paginate(
        ModelPackageGroupName=model_package_group_name,
        SortBy="CreationTime",
        SortOrder="Descending",
    ):
        for pkg in page.get("ModelPackageSummaryList", []):
            desc = sm.describe_model_package(ModelPackageName=pkg["ModelPackageArn"])
            if desc.get("ModelPackageVersion") == version:
                return pkg["ModelPackageArn"]
    raise RuntimeError(f"Version {version} not found in group {model_package_group_name}")


def get_latest_model_package(model_package_group_name: str, only_approved: bool = True) -> str:
    kwargs = {
        "ModelPackageGroupName": model_package_group_name,
        "SortBy": "CreationTime",
        "SortOrder": "Descending",
        "MaxResults": 1,
    }
    if only_approved:
        kwargs["ModelApprovalStatus"] = "Approved"

    response = sm.list_model_packages(**kwargs)
    packages = response.get("ModelPackageSummaryList", [])
    if not packages:
        status_text = "approved " if only_approved else ""
        raise RuntimeError(f"No {status_text}model packages found in group: {model_package_group_name}")

    return packages[0]["ModelPackageArn"]


def resolve_model_package_arn() -> str:
    if MODEL_PACKAGE_ARN:
        return MODEL_PACKAGE_ARN
    if MODEL_VERSION:
        return get_model_package_by_version(MODEL_PACKAGE_GROUP_NAME, int(MODEL_VERSION))
    return get_latest_model_package(MODEL_PACKAGE_GROUP_NAME, ONLY_APPROVED)


def ensure_model(model_name: str, model_package_arn: str) -> None:
    try:
        sm.describe_model(ModelName=model_name)
        print(f"Model already exists: {model_name}. Recreating.")
        sm.delete_model(ModelName=model_name)
    except ClientError as e:
        if "Could not find model" not in str(e):
            raise

    sm.create_model(
        ModelName=model_name,
        Containers=[{"ModelPackageName": model_package_arn}],
        ExecutionRoleArn=ROLE_ARN,
    )
    print(f"Created model: {model_name}")


def ensure_endpoint_config(endpoint_config_name: str, model_name: str) -> None:
    try:
        sm.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        print(f"Endpoint config already exists: {endpoint_config_name}. Recreating.")
        sm.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    except ClientError as e:
        if "Could not find endpoint configuration" not in str(e):
            raise

    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": INITIAL_INSTANCE_COUNT,
                "InstanceType": INSTANCE_TYPE,
                "InitialVariantWeight": 1.0,
            }
        ],
    )
    print(f"Created endpoint config: {endpoint_config_name}")


def create_or_update_endpoint(endpoint_name: str, endpoint_config_name: str) -> None:
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        print(f"Endpoint exists: {endpoint_name}. Updating.")
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
    except ClientError as e:
        if "Could not find endpoint" in str(e):
            print(f"Endpoint does not exist: {endpoint_name}. Creating.")
            sm.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
            )
        else:
            raise

    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)
    print(f"Endpoint is InService: {endpoint_name}")


if __name__ == "__main__":
    model_package_arn = resolve_model_package_arn()
    print(f"Using model package: {model_package_arn}")

    desc = sm.describe_model_package(ModelPackageName=model_package_arn)
    print(f"Selected version: {desc.get('ModelPackageVersion')}")
    print(f"Approval status: {desc.get('ModelApprovalStatus')}")

    ensure_model(MODEL_NAME, model_package_arn)
    ensure_endpoint_config(ENDPOINT_CONFIG_NAME, MODEL_NAME)
    create_or_update_endpoint(ENDPOINT_NAME, ENDPOINT_CONFIG_NAME)

    print(f"Deployment complete. Endpoint name: {ENDPOINT_NAME}")