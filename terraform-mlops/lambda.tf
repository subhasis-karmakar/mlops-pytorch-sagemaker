resource "aws_lambda_function" "retrain_lambda" {
  function_name = "RetrainTrigger"
  role          = aws_iam_role.lambda_role.arn
  runtime       = "python3.9"
  handler       = "retrain_trigger.lambda_handler"

  filename      = "${path.module}/../lambda/retrain_trigger.zip"

  environment {
    variables = {
      PIPELINE_NAME = aws_sagemaker_pipeline.mlops_pipeline.name
    }
  }
}
