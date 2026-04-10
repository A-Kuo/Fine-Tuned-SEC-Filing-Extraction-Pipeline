"""AWS SageMaker Model Deployment for Financial LLM.

Registers the trained QLoRA model (stored in S3) in the SageMaker Model Registry
and deploys it to a real-time inference endpoint backed by a HuggingFace DLC.

Architecture:
    S3 (model.tar.gz)
        └── SageMaker Model Registry  (versioned, auditable)
                └── SageMaker Endpoint  (auto-scaling, real-time REST)
                        └── inference.py  (custom handler: vLLM-style batching)

Endpoint design:
    - Instance: ml.g5.2xlarge (24 GB A10G)
    - Min/max capacity: 1–4 instances (auto-scales on SageMakerVariantInvocationsPerInstance)
    - Invocation timeout: 60 s  (SEC filings can be long)
    - Container: HuggingFace Inference DLC (transformers + peft pre-installed)
    - Custom inference.py: loads adapter on top of quantized base, returns JSON

Usage:
    # Deploy the latest training output
    python serving/sagemaker_deploy.py --model-data s3://my-bucket/output/model.tar.gz

    # Deploy and immediately run a smoke test
    python serving/sagemaker_deploy.py --model-data s3://my-bucket/output/model.tar.gz --smoke-test

    # Delete an endpoint to stop billing
    python serving/sagemaker_deploy.py --delete-endpoint my-endpoint-name
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from loguru import logger


# ---------------------------------------------------------------------------
# Model registration
# ---------------------------------------------------------------------------

def register_model(
    model_data_uri: str,
    role: str,
    session,
    model_name: str | None = None,
) -> str:
    """
    Register a trained model artefact in SageMaker Model Registry.

    Parameters
    ----------
    model_data_uri : str
        S3 URI of the model.tar.gz produced by the training job.
    role : str
        SageMaker execution role ARN.
    session : sagemaker.Session
    model_name : str, optional
        Explicit name; auto-generated if omitted.

    Returns
    -------
    str
        The registered model name.
    """
    from sagemaker.huggingface import HuggingFaceModel

    if model_name is None:
        model_name = f"findoc-qlora-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

    model = HuggingFaceModel(
        model_data=model_data_uri,
        role=role,
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        sagemaker_session=session,
        name=model_name,
        # Point to our custom inference handler
        entry_point="inference.py",
        source_dir=str(Path(__file__).parent),
        env={
            "HF_TASK": "text-generation",
            "PYTHONIOENCODING": "utf-8",
            "MAX_NEW_TOKENS": "512",
        },
    )

    logger.info("Registering model '{}' with data at {}", model_name, model_data_uri)
    model.create(instance_type="ml.g5.2xlarge")
    logger.info("Model '{}' registered successfully.", model_name)
    return model_name


# ---------------------------------------------------------------------------
# Endpoint deployment
# ---------------------------------------------------------------------------

def deploy_endpoint(
    model_name: str,
    role: str,
    session,
    instance_type: str = "ml.g5.2xlarge",
    initial_instance_count: int = 1,
    endpoint_name: str | None = None,
) -> str:
    """
    Deploy a registered model to a SageMaker real-time inference endpoint.

    Auto-scaling is configured via Application Auto Scaling after creation:
    - Scale-out: SageMakerVariantInvocationsPerInstance > 100 (target)
    - Min=1, Max=4 instances
    - Cooldown: 60 s in / 300 s out

    Returns
    -------
    str
        The endpoint name.
    """
    import boto3
    from sagemaker.huggingface import HuggingFaceModel

    if endpoint_name is None:
        endpoint_name = f"findoc-endpoint-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

    sm_client = session.boto_session.client("sagemaker")
    model_config = sm_client.describe_model(ModelName=model_name)

    predictor = HuggingFaceModel(
        model_data=model_config["PrimaryContainer"]["ModelDataUrl"],
        role=role,
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        sagemaker_session=session,
        name=model_name,
    ).deploy(
        initial_instance_count=initial_instance_count,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        wait=True,
    )

    logger.info("Endpoint '{}' is live on {}.", endpoint_name, instance_type)

    # ---------------------------------------------------------------------------
    # Auto-scaling
    # ---------------------------------------------------------------------------
    aas = boto3.client(
        "application-autoscaling",
        region_name=session.boto_session.region_name,
    )
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

    aas.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=1,
        MaxCapacity=4,
    )
    aas.put_scaling_policy(
        PolicyName=f"{endpoint_name}-invocations-scaling",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 100.0,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
            },
            "ScaleInCooldown": 300,
            "ScaleOutCooldown": 60,
        },
    )
    logger.info("Auto-scaling configured: 1–4 instances, target 100 invocations/instance.")
    return endpoint_name


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test(endpoint_name: str, session) -> dict:
    """
    Invoke the deployed endpoint with a sample SEC filing snippet.

    Returns the raw JSON response from the model.
    """
    import boto3

    runtime = boto3.client(
        "sagemaker-runtime",
        region_name=session.boto_session.region_name,
    )

    sample_input = {
        "inputs": (
            "Extract financial fields from this 10-K excerpt:\n\n"
            "Net revenues for fiscal 2024 were $12.4 billion, up 8% year-over-year. "
            "Operating income was $2.1 billion with a 17% operating margin. "
            "Cash and cash equivalents stood at $3.8 billion as of December 31, 2024.\n\n"
            "Return JSON with keys: revenue, revenue_growth_pct, operating_income, "
            "operating_margin_pct, cash_equivalents, period_end_date."
        )
    }

    t0 = time.perf_counter()
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(sample_input).encode(),
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    body = json.loads(response["Body"].read().decode())
    logger.info("Smoke test OK  latency={:.0f}ms", latency_ms)
    logger.info("Response: {}", json.dumps(body, indent=2))
    return {"latency_ms": latency_ms, "response": body}


# ---------------------------------------------------------------------------
# Endpoint deletion (stop billing)
# ---------------------------------------------------------------------------

def delete_endpoint(endpoint_name: str, session) -> None:
    sm_client = session.boto_session.client("sagemaker")
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    logger.info("Endpoint '{}' deleted.", endpoint_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deploy a trained FinDocAnalyzer model to SageMaker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-data",      dest="model_data",
                   help="S3 URI of model.tar.gz from training job")
    p.add_argument("--model-name",      dest="model_name", default=None,
                   help="Explicit model name in registry (auto-generated if omitted)")
    p.add_argument("--endpoint-name",   dest="endpoint_name", default=None,
                   help="Explicit endpoint name (auto-generated if omitted)")
    p.add_argument("--instance-type",   default="ml.g5.2xlarge", dest="instance_type")
    p.add_argument("--region",          default=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    p.add_argument("--profile",         default=os.getenv("AWS_PROFILE"))
    p.add_argument("--role",            default=os.getenv("SAGEMAKER_ROLE"))
    p.add_argument("--smoke-test",      action="store_true", dest="smoke_test",
                   help="Run a smoke test after deploying")
    p.add_argument("--delete-endpoint", metavar="ENDPOINT_NAME", dest="delete_endpoint",
                   help="Delete a deployed endpoint (stops billing)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import boto3
    import sagemaker

    boto_session = boto3.Session(
        region_name=args.region,
        profile_name=args.profile or None,
    )
    session = sagemaker.Session(boto_session=boto_session)
    role = args.role or sagemaker.get_execution_role(session)

    if args.delete_endpoint:
        delete_endpoint(args.delete_endpoint, session)
        return

    if not args.model_data:
        raise SystemExit("--model-data is required for deployment")

    model_name = register_model(args.model_data, role, session, args.model_name)
    endpoint_name = deploy_endpoint(
        model_name, role, session,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name,
    )

    if args.smoke_test:
        smoke_test(endpoint_name, session)


if __name__ == "__main__":
    main()
