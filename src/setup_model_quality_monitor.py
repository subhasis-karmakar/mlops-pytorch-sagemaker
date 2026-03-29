import boto3
from sagemaker.session import Session
from sagemaker.model_monitor import ModelQualityMonitor

REGION = "us-west-2"
ROLE_ARN = "arn:aws:iam::628479576048:role/SageMakerExecutionRole"
ENDPOINT_NAME = "pytorch-mlops-registry-endpoint"

GROUND_TRUTH_S3_URI = "s3://mlops-monitoring-bucket-b9c36351/groundtruth/"
MONITOR_OUTPUT_S3_URI = "s3://mlops-monitoring-bucket-b9c36351/monitoring/model-quality"

MONITOR_SCHEDULE_NAME = "pytorch-mlops-model-quality-monitor"

boto_session = boto3.Session(region_name=REGION)
sagemaker_session = Session(boto_session=boto_session)


def main():
    monitor = ModelQualityMonitor(
        role=ROLE_ARN,
        instance_count=1,
        instance_type="ml.m5.large",
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
        sagemaker_session=sagemaker_session,
    )

    print("Creating model quality monitoring schedule...")
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

    print(f"Model quality monitor created: {MONITOR_SCHEDULE_NAME}")
    print(f"Ground truth location expected under: {GROUND_TRUTH_S3_URI}")


if __name__ == "__main__":
    main()