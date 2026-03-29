import os
import boto3
from botocore.exceptions import ClientError
from sagemaker.session import Session
from sagemaker.model_monitor import ModelQualityMonitor, EndpointInput

REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
ROLE_ARN = "arn:aws:iam::628479576048:role/SageMakerExecutionRole"
ENDPOINT_NAME = "pytorch-mlops-registry-endpoint"

GROUND_TRUTH_S3_URI = "s3://mlops-monitoring-bucket-b9c36351/groundtruth/"
MONITOR_OUTPUT_S3_URI = "s3://mlops-monitoring-bucket-b9c36351/monitoring/model-quality"

MONITOR_SCHEDULE_NAME = "pytorch-mlops-model-quality-monitor"
MONITOR_INSTANCE_TYPE = os.getenv("MONITOR_INSTANCE_TYPE", "ml.m5.xlarge")

boto_session = boto3.Session(region_name=REGION)
sagemaker_session = Session(boto_session=boto_session)


def main():
    monitor = ModelQualityMonitor(
        role=ROLE_ARN,
        instance_count=1,
        instance_type=MONITOR_INSTANCE_TYPE,
        volume_size_in_gb=20,
        max_runtime_in_seconds=1800,
        sagemaker_session=sagemaker_session,
    )

    print(f"Creating model quality monitoring schedule with instance type: {MONITOR_INSTANCE_TYPE}")

    try:
        monitor.create_monitoring_schedule(
            monitor_schedule_name=MONITOR_SCHEDULE_NAME,
            endpoint_input=EndpointInput(
                endpoint_name=ENDPOINT_NAME,
                destination="/opt/ml/processing/input/endpoint",
                start_time_offset="-P2D",
                end_time_offset="-P1D",
                inference_attribute="predictions",
            ),
            ground_truth_input=GROUND_TRUTH_S3_URI,
            output_s3_uri=MONITOR_OUTPUT_S3_URI,
            problem_type="MulticlassClassification",
            schedule_cron_expression="cron(30 * ? * * *)",
            enable_cloudwatch_metrics=True,
        )
    except ClientError as e:
        raise RuntimeError(
            f"Failed to create model quality monitor. "
            f"Check SageMaker processing quota, endpoint data capture, and ground truth setup. "
            f"Original error: {e}"
        ) from e

    print(f"Model quality monitor created: {MONITOR_SCHEDULE_NAME}")
    print(f"Ground truth location expected under: {GROUND_TRUTH_S3_URI}")


if __name__ == "__main__":
    main()