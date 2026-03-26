terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Include all resources
module "s3" {
  source = "./s3.tf"
}

module "sagemaker" {
  source = "./sagemaker.tf"
}

module "monitor" {
  source = "./monitor.tf"
}

module "lambda" {
  source = "./lambda.tf"
}

module "iam" {
  source = "./iam.tf"
}
