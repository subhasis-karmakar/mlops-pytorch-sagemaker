import os
import boto3
from sagemaker.session import Session
from sagemaker.model_monitor import DefaultModelMonitor, DatasetFormat

REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
ROLE_ARN = "arn:aws:iam::628479576048:role/SageMakerExecutionRole"
ENDPOINT_NAME = "pytorch-mlops-registry-endpoint"

BASELINE_DATASET_S3_URI = "s3://mlops-monitoring-bucket-b9c36351/baseline/baseline-input.jsonl"
BASELINE_OUTPUT_S3_URI = "s3://mlops-monitoring-bucket-b9c36351/baselines/data-quality"
MONITOR_OUTPUT_S3_URI = "s3://mlops-monitoring-bucket-b9c36351/monitoring/data-quality"

MONITOR_SCHEDULE_NAME = "pytorch-mlops-data-quality-monitor"
MONITOR_INSTANCE_TYPE = os.getenv("MONITOR_INSTANCE_TYPE", "ml.t3.medium")

boto_session = boto3.Session(region_name=REGION)
sagemaker_session = Session(boto_session=boto_session)


def main():
    monitor = DefaultModelMonitor(
        role=ROLE_ARN,
        instance_count=1,
        instance_type=MONITOR_INSTANCE_TYPE,
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
        sagemaker_session=sagemaker_session,
    )

    print(f"Creating baseline using instance type: {MONITOR_INSTANCE_TYPE}")
    monitor.suggest_baseline(
        baseline_dataset=BASELINE_DATASET_S3_URI,
        dataset_format=DatasetFormat.json(lines=True),
        output_s3_uri=BASELINE_OUTPUT_S3_URI,
        wait=True,
        logs=False,
    )

    print("Creating monitoring schedule...")
    monitor.create_monitoring_schedule(
        monitor_schedule_name=MONITOR_SCHEDULE_NAME,
        endpoint_input=ENDPOINT_NAME,
        output_s3_uri=MONITOR_OUTPUT_S3_URI,
        statistics=monitor.baseline_statistics(),
        constraints=monitor.baseline_constraints(),
        schedule_cron_expression="cron(0 * ? * * *)",
        enable_cloudwatch_metrics=True,
    )

    print(f"Data quality monitor created: {MONITOR_SCHEDULE_NAME}")


if __name__ == "__main__":
    main()