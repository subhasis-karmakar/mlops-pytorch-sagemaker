resource "aws_s3_bucket" "mlops_data" {
  bucket = "mlops-data-bucket"
  tags = {
    Name        = "mlops-data"
    Environment = "mlops"
  }
}

resource "aws_s3_bucket" "mlops_monitoring" {
  bucket = "mlops-monitoring-bucket"
  tags = {
    Name        = "mlops-monitoring"
    Environment = "mlops"
  }
}

resource "aws_s3_bucket" "mlops_checkpoints" {
  bucket = "mlops-checkpoints-bucket"
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

    # Required filter/prefix block
    filter {
      prefix = "" # applies to all objects
    }

    expiration {
      days = 30
    }
  }
}
