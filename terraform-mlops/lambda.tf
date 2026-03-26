resource "aws_lambda_function" "retrain_lambda" {
  function_name = "mlops-retrain-lambda"
  role          = aws_iam_role.lambda_role.arn
  handler       = "retrain_trigger.lambda_handler"
  runtime       = "python3.9"
  filename      = "${path.module}/lambda.zip"

  environment {
    variables = {
      PIPELINE_NAME = aws_sagemaker_pipeline.mlops_pipeline.pipeline_name
    }
  }
}
