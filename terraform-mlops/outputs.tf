output "checkpoint_s3_uri" {
  description = "S3 URI for SageMaker checkpoints"
  value       = "s3://${aws_s3_bucket.mlops_checkpoints.bucket}/"
}

output "data_s3_uri" {
  description = "S3 URI for training data"
  value       = "s3://${aws_s3_bucket.mlops_data.bucket}/"
}

output "monitoring_s3_uri" {
  description = "S3 URI for monitoring outputs"
  value       = "s3://${aws_s3_bucket.mlops_monitoring.bucket}/"
}