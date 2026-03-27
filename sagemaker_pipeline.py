import os
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.pytorch import PyTorch
from sagemaker.model import Model

role = "arn:aws:iam::628479576048:role/SageMakerExecutionRole"

checkpoint_s3_uri = "s3://mlops-checkpoints-bucket-8a20dd98/"
data_s3_uri = "s3://mlops-data-bucket-ed1659d4/"
monitoring_s3_uri = "s3://mlops-monitoring-bucket-b9c36351/"

# Pipeline session
pipeline_session = PipelineSession()

# Estimator with checkpointing
estimator = PyTorch(
    entry_point="src/train.py",
    role=role,
    instance_type="ml.m5.large",
    framework_version="1.12",
    py_version="py38",
    hyperparameters={"epochs": 5, "batch_size": 64, "lr": 0.001},
    checkpoint_s3_uri=checkpoint_s3_uri,
    checkpoint_local_path="/opt/ml/checkpoints",
    sagemaker_session=pipeline_session
)

train_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={"training": f"{data_s3_uri}/cifar10/train"}
)

model = Model(
    image_uri=estimator.training_image_uri(),
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    entry_point="src/inference.py",
    source_dir="src",
    sagemaker_session=pipeline_session
)

model_step_args = model.create(instance_type="ml.m5.large")

model_step = ModelStep(
    name="RegisterModel",
    step_args=model_step_args
)

pipeline = Pipeline(
    name="PyTorchMLOpsPipeline",
    steps=[train_step, model_step],
    sagemaker_session=pipeline_session
)

if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    print("Pipeline execution started:", execution.arn)
