# CloudWatch Alarms for PyTorch MLOps

Create alarms for these metrics:

## Endpoint alarms
- Invocation5XXErrors > 0 for 5 minutes
- ModelLatency p95 above threshold
- OverheadLatency above threshold

## Model Monitor alarms
- Data quality violations detected
- Model quality violations detected
- Monitoring job failures

## Pipeline alarms
- SageMaker pipeline execution failed
- Retraining pipeline execution failed

## Suggested alarm names
- pytorch-mlops-endpoint-5xx
- pytorch-mlops-endpoint-latency
- pytorch-mlops-data-drift
- pytorch-mlops-model-drift
- pytorch-mlops-retraining-failed