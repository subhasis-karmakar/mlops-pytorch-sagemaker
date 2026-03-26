resource "aws_sagemaker_pipeline" "mlops_pipeline" {
  pipeline_name         = "mlops-pytorch-pipeline"
  pipeline_display_name = "PyTorch-MLOps-Pipeline" # spaces replaced with hyphens

  role_arn = aws_iam_role.sagemaker_role.arn

  pipeline_definition = <<EOF
{
  "Version": "2020-12-01",
  "Steps": [
    {
      "Name": "TrainModel",
      "Type": "Training",
      "Arguments": {
        "TrainingJobName": "pytorch-training-job",
        "AlgorithmSpecification": {
          "TrainingImage": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12-cpu-py38",
          "TrainingInputMode": "File"
        },
        "OutputDataConfig": {
          "S3OutputPath": "s3://${aws_s3_bucket.mlops_data.bucket}/output/"
        },
        "CheckpointConfig": {
          "S3Uri": "s3://${aws_s3_bucket.mlops_checkpoints.bucket}/",
          "LocalPath": "/opt/ml/checkpoints"
        }
      }
    }
  ]
}
EOF
}
