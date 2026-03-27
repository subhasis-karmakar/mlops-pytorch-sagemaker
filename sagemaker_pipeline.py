from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ModelStep
from sagemaker.pytorch import PyTorch
from sagemaker.model import Model
from sagemaker import get_execution_role

# Define bucket URIs as variables
checkpoint_s3_uri = "s3://mlops-checkpoints-bucket-8a20dd98/"
data_s3_uri = "s3://mlops-data-bucket-ed1659d4/"

role = get_execution_role()

estimator = PyTorch(
    entry_point="src/train.py",
    role=role,
    instance_type="ml.m5.large",
    framework_version="1.12",
    py_version="py38",
    hyperparameters={"epochs": 5, "batch_size": 64, "lr": 0.001}
)

train_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={"training": f"{data_s3_uri}/cifar10/train"},
    checkpoint_config={
        "S3Uri": checkpoint_s3_uri,
        "LocalPath": "/opt/ml/checkpoints"
    }
)

model = Model(
    image_uri=estimator.training_image_uri(),
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    entry_point="src/inference.py",
    source_dir="src"
)

model_step = ModelStep(name="RegisterModel", model=model)

pipeline = Pipeline(name="PyTorchMLOpsPipeline", steps=[train_step, model_step])

if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    print("Pipeline execution started:", execution.arn)
