from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.parameters import ParameterFloat
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.fail_step import FailStep

from sagemaker.pytorch import PyTorch
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.inputs import TrainingInput


ROLE = "arn:aws:iam::628479576048:role/SageMakerExecutionRole"

CHECKPOINT_S3_URI = "s3://mlops-checkpoints-bucket-8a20dd98/"
DATA_S3_URI = "s3://mlops-data-bucket-ed1659d4/"
MONITORING_S3_URI = "s3://mlops-monitoring-bucket-b9c36351/"

PIPELINE_NAME = "PyTorchMLOpsPipeline"
MODEL_PACKAGE_GROUP_NAME = "PyTorchMLOpsModelGroup"

TRAIN_INSTANCE_TYPE = "ml.m5.large"
EVAL_INSTANCE_TYPE = "ml.t3.medium"
REGISTER_INFERENCE_INSTANCE = "ml.m5.large"
REGISTER_TRANSFORM_INSTANCE = "ml.m5.large"

pipeline_session = PipelineSession()

accuracy_threshold = ParameterFloat(
    name="AccuracyThreshold",
    default_value=0.20,
)


def build_pipeline() -> Pipeline:
    estimator = PyTorch(
        entry_point="src/train.py",
        source_dir=".",
        role=ROLE,
        instance_type=TRAIN_INSTANCE_TYPE,
        instance_count=1,
        framework_version="1.12",
        py_version="py38",
        hyperparameters={
            "epochs": 5,
            "batch_size": 64,
            "lr": 0.001,
        },
        checkpoint_s3_uri=CHECKPOINT_S3_URI,
        checkpoint_local_path="/opt/ml/checkpoints",
        sagemaker_session=pipeline_session,
    )

    train_args = estimator.fit(
        inputs={
            "training": TrainingInput(
                s3_data=f"{DATA_S3_URI}cifar10/train",
            )
        }
    )

    train_step = TrainingStep(
        name="TrainModel",
        step_args=train_args,
    )

    script_eval = ScriptProcessor(
        image_uri=estimator.training_image_uri(),
        command=["python3"],
        role=ROLE,
        instance_count=1,
        instance_type=EVAL_INSTANCE_TYPE,
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
                destination="/opt/ml/processing/model",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"{MONITORING_S3_URI}evaluation",
            )
        ],
    )

    eval_step = ProcessingStep(
        name="EvaluateModel",
        step_args=eval_args,
        property_files=[evaluation_report],
    )

    eval_accuracy = JsonGet(
        step_name=eval_step.name,
        property_file=evaluation_report,
        json_path="accuracy",
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    eval_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
                    "evaluation.json",
                ],
            ),
            content_type="application/json",
        )
    )

    register_step = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/x-image"],
        response_types=["application/json"],
        inference_instances=[REGISTER_INFERENCE_INSTANCE],
        transform_instances=[REGISTER_TRANSFORM_INSTANCE],
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
        model_metrics=model_metrics,
        approval_status="Approved",
    )

    fail_step = FailStep(
        name="AccuracyTooLow",
        error_message="Model accuracy did not meet threshold for registration.",
    )

    cond_step = ConditionStep(
        name="CheckAccuracy",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=eval_accuracy,
                right=accuracy_threshold,
            )
        ],
        if_steps=[register_step],
        else_steps=[fail_step],
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[accuracy_threshold],
        steps=[train_step, eval_step, cond_step],
        sagemaker_session=pipeline_session,
    )


if __name__ == "__main__":
    pipeline = build_pipeline()
    pipeline.upsert(role_arn=ROLE)
    execution = pipeline.start()
    print("Pipeline execution started:", execution.arn)