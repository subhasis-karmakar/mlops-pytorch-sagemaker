resource "aws_cloudwatch_metric_alarm" "model_drift_alarm" {
  alarm_name          = "ModelDriftAlarm"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "violations"
  namespace           = "AWS/SageMaker"
  period              = 3600
  statistic           = "Sum"
  threshold           = 1
  alarm_actions       = [aws_lambda_function.retrain_lambda.arn]
}