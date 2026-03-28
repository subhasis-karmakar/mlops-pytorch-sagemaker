from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

import src.deploy_model as deploy_model


# -------------------------
# Shared test constants
# -------------------------
MODEL_GROUP = "PyTorchMLOpsModelGroup"
MODEL_NAME = "test-model"
ENDPOINT_CONFIG_NAME = "test-endpoint-config"
ENDPOINT_NAME = "test-endpoint"

MODEL_PACKAGE_ARN_EXPLICIT = "arn:explicit"
MODEL_PACKAGE_ARN_LATEST = "arn:aws:sagemaker:us-west-2:123:model-package/latest-approved"
MODEL_PACKAGE_ARN_VERSION_1 = "arn:one"
MODEL_PACKAGE_ARN_VERSION_3 = "arn:two"
MODEL_PACKAGE_ARN_GENERIC = "arn:model-package"

MODEL_VERSION_FOUND = 3
MODEL_VERSION_MISSING = 99

ROLE_ARN = deploy_model.ROLE_ARN
INSTANCE_TYPE = deploy_model.INSTANCE_TYPE
INITIAL_INSTANCE_COUNT = deploy_model.INITIAL_INSTANCE_COUNT


def make_client_error(message: str, code: str = "ValidationException") -> ClientError:
    return ClientError(
        error_response={
            "Error": {
                "Code": code,
                "Message": message,
            }
        },
        operation_name="TestOperation",
    )


@pytest.fixture
def mock_sm():
    with patch.object(deploy_model, "sm", autospec=True) as mock_client:
        yield mock_client


def test_get_latest_model_package_returns_latest_approved(mock_sm):
    mock_sm.list_model_packages.return_value = {
        "ModelPackageSummaryList": [
            {"ModelPackageArn": MODEL_PACKAGE_ARN_LATEST}
        ]
    }

    result = deploy_model.get_latest_model_package(
        MODEL_GROUP,
        only_approved=True,
    )

    assert result == MODEL_PACKAGE_ARN_LATEST
    mock_sm.list_model_packages.assert_called_once_with(
        ModelPackageGroupName=MODEL_GROUP,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
        ModelApprovalStatus="Approved",
    )


def test_get_latest_model_package_raises_when_empty(mock_sm):
    mock_sm.list_model_packages.return_value = {"ModelPackageSummaryList": []}

    with pytest.raises(RuntimeError, match="No approved model packages found"):
        deploy_model.get_latest_model_package(
            MODEL_GROUP,
            only_approved=True,
        )


def test_get_model_package_by_version_returns_matching_version(mock_sm):
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {
            "ModelPackageSummaryList": [
                {"ModelPackageArn": MODEL_PACKAGE_ARN_VERSION_1},
                {"ModelPackageArn": MODEL_PACKAGE_ARN_VERSION_3},
            ]
        }
    ]
    mock_sm.get_paginator.return_value = paginator

    def describe_side_effect(ModelPackageName):
        if ModelPackageName == MODEL_PACKAGE_ARN_VERSION_1:
            return {"ModelPackageVersion": 1}
        if ModelPackageName == MODEL_PACKAGE_ARN_VERSION_3:
            return {"ModelPackageVersion": MODEL_VERSION_FOUND}
        return {}

    mock_sm.describe_model_package.side_effect = describe_side_effect

    result = deploy_model.get_model_package_by_version(
        MODEL_GROUP,
        MODEL_VERSION_FOUND,
    )

    assert result == MODEL_PACKAGE_ARN_VERSION_3


def test_get_model_package_by_version_raises_when_not_found(mock_sm):
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"ModelPackageSummaryList": [{"ModelPackageArn": MODEL_PACKAGE_ARN_VERSION_1}]}
    ]
    mock_sm.get_paginator.return_value = paginator
    mock_sm.describe_model_package.return_value = {"ModelPackageVersion": 1}

    with pytest.raises(RuntimeError, match=f"Version {MODEL_VERSION_MISSING} not found"):
        deploy_model.get_model_package_by_version(
            MODEL_GROUP,
            MODEL_VERSION_MISSING,
        )


def test_resolve_model_package_arn_prefers_exact_arn(monkeypatch):
    monkeypatch.setattr(deploy_model, "MODEL_PACKAGE_ARN", MODEL_PACKAGE_ARN_EXPLICIT)
    monkeypatch.setattr(deploy_model, "MODEL_VERSION", None)

    result = deploy_model.resolve_model_package_arn()

    assert result == MODEL_PACKAGE_ARN_EXPLICIT


def test_resolve_model_package_arn_uses_version(monkeypatch):
    monkeypatch.setattr(deploy_model, "MODEL_PACKAGE_ARN", None)
    monkeypatch.setattr(deploy_model, "MODEL_VERSION", MODEL_VERSION_FOUND)

    with patch.object(
        deploy_model,
        "get_model_package_by_version",
        return_value=MODEL_PACKAGE_ARN_VERSION_3,
    ) as mock_get:
        result = deploy_model.resolve_model_package_arn()

    assert result == MODEL_PACKAGE_ARN_VERSION_3
    mock_get.assert_called_once_with(MODEL_GROUP, MODEL_VERSION_FOUND)


def test_resolve_model_package_arn_uses_latest_when_no_override(monkeypatch):
    monkeypatch.setattr(deploy_model, "MODEL_PACKAGE_ARN", None)
    monkeypatch.setattr(deploy_model, "MODEL_VERSION", None)
    monkeypatch.setattr(deploy_model, "ONLY_APPROVED", True)

    with patch.object(
        deploy_model,
        "get_latest_model_package",
        return_value=MODEL_PACKAGE_ARN_LATEST,
    ) as mock_get:
        result = deploy_model.resolve_model_package_arn()

    assert result == MODEL_PACKAGE_ARN_LATEST
    mock_get.assert_called_once_with(MODEL_GROUP, True)


def test_ensure_model_creates_when_missing(mock_sm):
    mock_sm.describe_model.side_effect = make_client_error("Could not find model")

    deploy_model.ensure_model(MODEL_NAME, MODEL_PACKAGE_ARN_GENERIC)

    mock_sm.create_model.assert_called_once_with(
        ModelName=MODEL_NAME,
        Containers=[{"ModelPackageName": MODEL_PACKAGE_ARN_GENERIC}],
        ExecutionRoleArn=ROLE_ARN,
    )


def test_ensure_model_recreates_when_exists(mock_sm):
    mock_sm.describe_model.return_value = {"ModelName": MODEL_NAME}

    deploy_model.ensure_model(MODEL_NAME, MODEL_PACKAGE_ARN_GENERIC)

    mock_sm.delete_model.assert_called_once_with(ModelName=MODEL_NAME)
    mock_sm.create_model.assert_called_once_with(
        ModelName=MODEL_NAME,
        Containers=[{"ModelPackageName": MODEL_PACKAGE_ARN_GENERIC}],
        ExecutionRoleArn=ROLE_ARN,
    )


def test_ensure_model_reraises_unexpected_client_error(mock_sm):
    mock_sm.describe_model.side_effect = make_client_error(
        "Access denied",
        code="AccessDeniedException",
    )

    with pytest.raises(ClientError):
        deploy_model.ensure_model(MODEL_NAME, MODEL_PACKAGE_ARN_GENERIC)


def test_ensure_endpoint_config_creates_when_missing(mock_sm):
    mock_sm.describe_endpoint_config.side_effect = make_client_error(
        "Could not find endpoint configuration"
    )

    deploy_model.ensure_endpoint_config(ENDPOINT_CONFIG_NAME, MODEL_NAME)

    mock_sm.create_endpoint_config.assert_called_once_with(
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": MODEL_NAME,
                "InitialInstanceCount": INITIAL_INSTANCE_COUNT,
                "InstanceType": INSTANCE_TYPE,
                "InitialVariantWeight": 1.0,
            }
        ],
    )


def test_ensure_endpoint_config_recreates_when_exists(mock_sm):
    mock_sm.describe_endpoint_config.return_value = {
        "EndpointConfigName": ENDPOINT_CONFIG_NAME
    }

    deploy_model.ensure_endpoint_config(ENDPOINT_CONFIG_NAME, MODEL_NAME)

    mock_sm.delete_endpoint_config.assert_called_once_with(
        EndpointConfigName=ENDPOINT_CONFIG_NAME
    )
    mock_sm.create_endpoint_config.assert_called_once()


def test_create_or_update_endpoint_creates_when_missing(mock_sm):
    mock_sm.describe_endpoint.side_effect = make_client_error("Could not find endpoint")
    waiter = MagicMock()
    mock_sm.get_waiter.return_value = waiter

    deploy_model.create_or_update_endpoint(ENDPOINT_NAME, ENDPOINT_CONFIG_NAME)

    mock_sm.create_endpoint.assert_called_once_with(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
    )
    waiter.wait.assert_called_once_with(EndpointName=ENDPOINT_NAME)


def test_create_or_update_endpoint_updates_when_exists(mock_sm):
    mock_sm.describe_endpoint.return_value = {"EndpointName": ENDPOINT_NAME}
    waiter = MagicMock()
    mock_sm.get_waiter.return_value = waiter

    deploy_model.create_or_update_endpoint(ENDPOINT_NAME, ENDPOINT_CONFIG_NAME)

    mock_sm.update_endpoint.assert_called_once_with(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
    )
    waiter.wait.assert_called_once_with(EndpointName=ENDPOINT_NAME)