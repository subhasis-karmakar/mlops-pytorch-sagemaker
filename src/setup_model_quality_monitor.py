import os
import boto3
from botocore.exceptions import ClientError
from sagemaker.session import Session
from sagemaker.model_monitor import ModelQualityMonitor

REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
ROLE_ARN = "arn:aws:iam::628479576048:role/SageMakerExecutionRole"
ENDPOINT_NAME = "pytorch-mlops-registry-endpoint"

GROUND_TRUTH_S3_URI = "s3://mlops-monitoring-bucket-b9c36351/groundtruth/"
MONITOR_OUTPUT_S3_URI = "s3://mlops-monitoring-bucket-b9c36351/monitoring/model-quality"

MONITOR_SCHEDULE_NAME = "pytorch-mlops-model-quality-monitor"
MONITOR_INSTANCE_TYPE = os.getenv("MONITOR_INSTANCE_TYPE", "ml.t3.medium")

boto_session = boto3.Session(region_name=REGION)
sagemaker_session = Session(boto_session=boto_session)


def main():
    monitor = ModelQualityMonitor(
        role=ROLE_ARN,
        instance_count=1,
        instance_type=MONITOR_INSTANCE_TYPE,
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
        sagemaker_session=sagemaker_session,
    )

    print(f"Creating model quality monitoring schedule with instance type: {MONITOR_INSTANCE_TYPE}")

    try:
        monitor.create_monitoring_schedule(
            monitor_schedule_name=MONITOR_SCHEDULE_NAME,
            endpoint_input=ENDPOINT_NAME,
            output_s3_uri=MONITOR_OUTPUT_S3_URI,
            problem_type="MulticlassClassification",
            inference_attribute="0",
            ground_truth_attribute="0",
            schedule_cron_expression="cron(30 * ? * * *)",
            enable_cloudwatch_metrics=True,
        )
    except ClientError as e:
        raise RuntimeError(
            f"Failed to create model quality monitor. "
            f"Likely cause: no SageMaker processing quota for instance type {MONITOR_INSTANCE_TYPE}. "
            f"Original error: {e}"
        ) from e

    print(f"Model quality monitor created: {MONITOR_SCHEDULE_NAME}")
    print(f"Ground truth location expected under: {GROUND_TRUTH_S3_URI}")


if __name__ == "__main__":
    main()