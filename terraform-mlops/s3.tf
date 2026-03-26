# Random suffix for mlops_data bucket
resource "random_id" "data_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket" "mlops_data" {
  bucket = "mlops-data-bucket-${random_id.data_suffix.hex}"
  tags = {
    Name        = "mlops-data"
    Environment = "mlops"
  }
}

# Random suffix for mlops_monitoring bucket
resource "random_id" "monitoring_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket" "mlops_monitoring" {
  bucket = "mlops-monitoring-bucket-${random_id.monitoring_suffix.hex}"
  tags = {
    Name        = "mlops-monitoring"
    Environment = "mlops"
  }
}

# Random suffix for mlops_checkpoints bucket
resource "random_id" "checkpoints_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket" "mlops_checkpoints" {
  bucket = "mlops-checkpoints-bucket-${random_id.checkpoints_suffix.hex}"

  tags = {
    Name        = "mlops-checkpoints"
    Environment = "mlops"
  }
}

# Separate resource for versioning
resource "aws_s3_bucket_versioning" "mlops_checkpoints_versioning" {
  bucket = aws_s3_bucket.mlops_checkpoints.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Separate resource for lifecycle configuration
resource "aws_s3_bucket_lifecycle_configuration" "mlops_checkpoints_lifecycle" {
  bucket = aws_s3_bucket.mlops_checkpoints.id

  rule {
    id     = "expire-old-checkpoints"
    status = "Enabled"

    filter {
      prefix = "" # applies to all objects
    }

    expiration {
      days = 30
    }
  }
}
