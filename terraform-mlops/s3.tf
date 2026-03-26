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

  versioning {
    enabled = true
  }

  lifecycle_rule {
    id      = "expire-old-checkpoints"
    enabled = true
    expiration {
      days = 30
    }
  }

  tags = {
    Name        = "mlops-checkpoints"
    Environment = "mlops"
  }
}
