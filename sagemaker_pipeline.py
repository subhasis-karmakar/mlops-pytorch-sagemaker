import os
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.pytorch import PyTorch
from sagemaker.model import Model

# Pull role from environment (set in GitHub Actions secrets or hard-coded for now)
role = "arn:aws:iam::628479576048:role/SageMakerExecutionRole"

# Terraform outputs (replace with env vars if you prefer)
checkpoint_s3_uri = "s3://mlops-checkpoints-bucket-8a20dd98/"
data_s3_uri = "s3://mlops-data-bucket-ed1659d4/"
monitoring_s3_uri = "s3://mlops-monitoring-bucket-b9c36351/"

# Configure estimator with checkpointing
estimator = PyTorch(
    entry_point="src/train.py",
    role=role,
    instance_type="ml.m5.large",
    framework_version="1.12",
    py_version="py38",
    hyperparameters={"epochs": 5, "batch_size": 64, "lr": 0.001},
    checkpoint_s3_uri=checkpoint_s3_uri,
    checkpoint_local_path="/opt/ml/checkpoints"
)

# Training step
train_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={"training": f"{data_s3_uri}/cifar10/train"}
)

# Model definition
model = Model(
    image_uri=estimator.training_image_uri(),
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    entry_point="src/inference.py",
    source_dir="src"
)

# Build step_args from model.create()
model_step_args = model.create(instance_type="ml.m5.large")

# Model registration step
model_step = ModelStep(
    name="RegisterModel",
    step_args=model_step_args
)

# Pipeline definition
pipeline = Pipeline(
    name="PyTorchMLOpsPipeline",
    steps=[train_step, model_step]
)

if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    print("Pipeline execution started:", execution.arn)
