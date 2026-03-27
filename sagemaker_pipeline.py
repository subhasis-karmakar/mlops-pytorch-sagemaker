import os

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.parameters import ParameterFloat
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.step_collections import RegisterModel

from sagemaker.pytorch import PyTorch
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch.model import PyTorchModel


role = "arn:aws:iam::628479576048:role/SageMakerExecutionRole"

checkpoint_s3_uri = "s3://mlops-checkpoints-bucket-8a20dd98/"
data_s3_uri = "s3://mlops-data-bucket-ed1659d4/"
monitoring_s3_uri = "s3://mlops-monitoring-bucket-b9c36351/"

pipeline_session = PipelineSession()

accuracy_threshold = ParameterFloat(
    name="AccuracyThreshold",
    default_value=0.80,
)

# =========================
# Training
# =========================
estimator = PyTorch(
    entry_point="src/train.py",
    source_dir=".",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    framework_version="1.12",
    py_version="py38",
    hyperparameters={"epochs": 5},
    checkpoint_s3_uri=checkpoint_s3_uri,
    checkpoint_local_path="/opt/ml/checkpoints",
    sagemaker_session=pipeline_session,
)

train_args = estimator.fit(
    inputs={
        "training": TrainingInput(
            s3_data=f"{data_s3_uri}cifar10/train"
        )
    }
)

train_step = TrainingStep(
    name="TrainModel",
    step_args=train_args,
)

# =========================
# Evaluation
# =========================
script_eval = ScriptProcessor(
    image_uri=estimator.training_image_uri(),
    command=["python3"],
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=pipeline_session,
)

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json",
)

eval_args = script_eval.run(
    code="src/evaluate.py",
    inputs=[
        ProcessingInput(
            source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=f"{data_s3_uri}cifar10/test",
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
            destination=f"{monitoring_s3_uri}evaluation"
        )
    ],
)

eval_step = ProcessingStep(
    name="EvaluateModel",
    step_args=eval_args,
    property_files=[evaluation_report],
)

# =========================
# Register Model
# =========================
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=f"{monitoring_s3_uri}evaluation/evaluation.json",
        content_type="application/json"
    )
)

register_step = RegisterModel(
    name="RegisterModel",
    estimator=estimator,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    model_package_group_name="PyTorchMLOpsModelGroup",
    model_metrics=model_metrics,
    approval_status="PendingManualApproval",
)

# =========================
# Condition Step (FIXED)
# =========================
eval_accuracy = JsonGet(
    step_name=eval_step.name,
    property_file=evaluation_report,
    json_path="evaluation.accuracy"
)

cond_step = ConditionStep(
    name="CheckAccuracy",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=eval_accuracy,
            right=accuracy_threshold
        )
    ],
    if_steps=[register_step],
    else_steps=[]
)

# =========================
# Pipeline
# =========================
pipeline = Pipeline(
    name="PyTorchMLOpsPipeline",
    parameters=[accuracy_threshold],
    steps=[train_step, eval_step, cond_step],
    sagemaker_session=pipeline_session,
)

if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    print("Pipeline execution started:", execution.arn)