import boto3
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.session import Session

REGION = "us-west-2"
ROLE_ARN = "arn:aws:iam::628479576048:role/SageMakerExecutionRole"
ENDPOINT_NAME = "pytorch-mlops-registry-endpoint"

BASELINE_RESULTS_S3_URI = "s3://mlops-monitoring-bucket-b9c36351/baselines/data-quality"
MONITOR_OUTPUT_S3_URI = "s3://mlops-monitoring-bucket-b9c36351/monitoring/data-quality"
CAPTURE_S3_URI = "s3://mlops-monitoring-bucket-b9c36351/datacapture"

boto_session = boto3.Session(region_name=REGION)
sagemaker_session = Session(boto_session=boto_session)

monitor = DefaultModelMonitor(
    role=ROLE_ARN,
    instance_count=1,
    instance_type="ml.m5.large",
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
    sagemaker_session=sagemaker_session,
)

# Use a small representative inference payload dataset in S3 as baseline input
# Example: s3://.../baseline/baseline-input.jsonl
baseline_dataset = "s3://mlops-monitoring-bucket-b9c36351/baseline/baseline-input.jsonl"

monitor.suggest_baseline(
    baseline_dataset=baseline_dataset,
    dataset_format={"json": {"lines": True}},
    output_s3_uri=BASELINE_RESULTS_S3_URI,
    wait=True,
    logs=False,
)

monitor.create_monitoring_schedule(
    monitor_schedule_name="pytorch-mlops-data-quality-monitor",
    endpoint_input=ENDPOINT_NAME,
    output_s3_uri=MONITOR_OUTPUT_S3_URI,
    statistics=monitor.baseline_statistics(),
    constraints=monitor.baseline_constraints(),
    schedule_cron_expression="cron(0 * ? * * *)",
    enable_cloudwatch_metrics=True,
)