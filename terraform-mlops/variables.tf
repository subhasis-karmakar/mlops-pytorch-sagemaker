variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Name prefix for all resources"
  type        = string
  default     = "mlops-pytorch-sagemaker"
}

variable "checkpoint_bucket_name" {
  description = "S3 bucket name for checkpoints"
  type        = string
  default     = "mlops-checkpoints-bucket"
}

variable "data_bucket_name" {
  description = "S3 bucket name for training data"
  type        = string
  default     = "mlops-data-bucket"
}

variable "monitoring_bucket_name" {
  description = "S3 bucket name for monitoring outputs"
  type        = string
  default     = "mlops-monitoring-bucket"
}
