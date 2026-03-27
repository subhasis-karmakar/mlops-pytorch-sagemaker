import os
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.pytorch import PyTorch
from sagemaker.model import Model
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep

role = "arn:aws:iam::628479576048:role/SageMakerExecutionRole"

checkpoint_s3_uri = "s3://mlops-checkpoints-bucket-8a20dd98/"
data_s3_uri = "s3://mlops-data-bucket-ed1659d4/"
monitoring_s3_uri = "s3://mlops-monitoring-bucket-b9c36351/"

# Pipeline session
pipeline_session = PipelineSession()

# Estimator with checkpointing + output path
estimator = PyTorch(
    entry_point="src/train.py",
    source_dir=".",               # upload the whole repo
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    framework_version="1.12",
    py_version="py38",
    hyperparameters={"epochs": 5, "batch_size": 64, "lr": 0.001},
    checkpoint_s3_uri=checkpoint_s3_uri,
    checkpoint_local_path="/opt/ml/checkpoints",
    sagemaker_session=pipeline_session,
    output_path=monitoring_s3_uri   # ✅ model + evaluation JSONs land here
)

train_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={"training": f"{data_s3_uri}cifar10/train"}
)

# Evaluation step: run evaluate.py and produce evaluation.json
script_eval = ScriptProcessor(
    image_uri=estimator.training_image_uri(),
    command=["python3"],
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=pipeline_session
)

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation_output",
    path="evaluation.json"
)

eval_step = ProcessingStep(
    name="EvaluateModel",
    processor=script_eval,
    inputs=[
        ProcessingInput(
            source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/model",
            destination=monitoring_s3_uri,
            output_name="evaluation_output"
        )
    ],
    code="src/evaluate.py",
    property_files=[evaluation_report]
)

# Register model
model = Model(
    image_uri=estimator.training_image_uri(),
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    entry_point="src/inference.py",
    source_dir=".",
    sagemaker_session=pipeline_session
)

model_step_args = model.create(instance_type="ml.m5.large")
model_step = ModelStep(name="RegisterModel", step_args=model_step_args)

# Condition: only register if accuracy >= 0.8
cond_gte = ConditionGreaterThanOrEqualTo(
    left=eval_step.properties.PropertyFiles["EvaluationReport"]["accuracy"],
    right=0.2
)

cond_step = ConditionStep(
    name="CheckAccuracy",
    conditions=[cond_gte],
    if_steps=[model_step],
    else_steps=[]   # skip registration if accuracy < 0.2
)

# Deployment step: create a real-time endpoint
deploy_step_args = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="pytorch-mlops-endpoint"
)

# Pipeline definition
pipeline = Pipeline(
    name="PyTorchMLOpsPipeline",
    steps=[train_step, eval_step, cond_step, deploy_step_args],
    sagemaker_session=pipeline_session
)

if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    print("Pipeline execution started:", execution.arn)
