import boto3
from sagemaker.model_monitor import ModelQualityMonitor
from sagemaker.session import Session

REGION = "us-west-2"
ROLE_ARN = "arn:aws:iam::628479576048:role/SageMakerExecutionRole"
ENDPOINT_NAME = "pytorch-mlops-registry-endpoint"

GROUND_TRUTH_S3_URI = "s3://mlops-monitoring-bucket-b9c36351/groundtruth/"
OUTPUT_S3_URI = "s3://mlops-monitoring-bucket-b9c36351/monitoring/model-quality"

boto_session = boto3.Session(region_name=REGION)
sagemaker_session = Session(boto_session=boto_session)

monitor = ModelQualityMonitor(
    role=ROLE_ARN,
    instance_count=1,
    instance_type="ml.m5.large",
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
    sagemaker_session=sagemaker_session,
)

monitor.create_monitoring_schedule(
    monitor_schedule_name="pytorch-mlops-model-quality-monitor",
    endpoint_input=ENDPOINT_NAME,
    output_s3_uri=OUTPUT_S3_URI,
    problem_type="MulticlassClassification",
    inference_attribute="0",
    ground_truth_attribute="0",
    schedule_cron_expression="cron(30 * ? * * *)",
    enable_cloudwatch_metrics=True,
)