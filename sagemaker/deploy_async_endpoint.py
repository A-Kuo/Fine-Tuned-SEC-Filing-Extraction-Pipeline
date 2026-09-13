"""Deploy the fine-tuned adapter to a SageMaker Asynchronous Inference endpoint.

WARNING: running this with --yes-really-deploy creates real, billable AWS
resources (an S3 upload, a SageMaker Model, EndpointConfig, and Endpoint,
plus an Application Auto Scaling policy). Nothing in this repo runs this
script automatically. Written but not executed as of this commit -- no
live endpoint exists yet.

Why Asynchronous Inference specifically (not Serverless Inference or a
real-time endpoint):
  - Serverless Inference: no GPU support at all (CPU-only, 6GB max) --
    can't run an NF4-quantized 8B model (~7.2GB resident, per this repo's
    real measured T4 load benchmark in MODEL_CARD.md).
  - Real-time endpoint: supports GPU, but bills per-hour for the instance
    whether or not it's handling requests -- expensive for a demo project
    with sporadic traffic.
  - Async Inference: supports GPU AND can scale to zero when idle via
    Application Auto Scaling's HasNoActiveInferenceRequests metric --
    the only mode that fits both "needs GPU" and "pay only when used."

Prerequisites this script does NOT create for you:
  1. A trained LoRA adapter. Check models/llama-sec-v1/ (or wherever
     .github/workflows/kaggle_training.yml's artifact was downloaded to)
     exists and has real adapter_config.json/adapter_model.safetensors
     before running this -- packaging a nonexistent/untrained adapter
     produces a working endpoint that returns garbage, not an error.
  2. An IAM role with sagemaker:*, s3:GetObject/PutObject on the buckets
     used here, and logs:* for CloudWatch. Pass its ARN via --role-arn.
  3. HF_TOKEN available to the endpoint's environment (Llama 3.1 is
     gated) -- passed via --hf-token or the HF_TOKEN env var.

Usage (dry run, default -- prints what would happen, touches no AWS):
    python sagemaker/deploy_async_endpoint.py --adapter-dir models/llama-sec-v1 \\
        --role-arn arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole

Usage (real deployment):
    python sagemaker/deploy_async_endpoint.py --adapter-dir models/llama-sec-v1 \\
        --role-arn arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole \\
        --yes-really-deploy
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

ENDPOINT_NAME = "findoc-llama-sec-async"
MODEL_NAME = "findoc-llama-sec-model"
ENDPOINT_CONFIG_NAME = "findoc-llama-sec-async-config"
# Smallest GPU instance with enough VRAM for an NF4 8B model + KV cache
# headroom -- matches this repo's own real Tesla T4 load benchmark.
INSTANCE_TYPE = "ml.g4dn.xlarge"


def package_adapter(adapter_dir: Path) -> Path:
    """Tar the LoRA adapter directory (small -- ~85MB per this repo's real
    measured fp16 size, see MODEL_CARD.md) into model.tar.gz. Does NOT
    include base model weights -- those are pulled from HuggingFace at
    container startup by sagemaker/inference.py::model_fn()."""
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"{adapter_dir} does not exist. No trained adapter to package -- "
            "run .github/workflows/kaggle_training.yml first and download its artifact."
        )
    required = ["adapter_config.json"]
    missing = [f for f in required if not (adapter_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{adapter_dir} is missing {missing} -- doesn't look like a real "
            "trained adapter directory."
        )

    tmp_path = Path(tempfile.mkdtemp()) / "model.tar.gz"
    with tarfile.open(tmp_path, "w:gz") as tar:
        for item in adapter_dir.iterdir():
            tar.add(item, arcname=item.name)
    return tmp_path


def deploy(adapter_dir: Path, role_arn: str, hf_token: str, s3_bucket: str, dry_run: bool) -> None:
    print(f"Packaging adapter from {adapter_dir}...")
    tarball = package_adapter(adapter_dir)
    print(f"  -> {tarball} ({tarball.stat().st_size / 1e6:.1f} MB)")

    model_s3_uri = f"s3://{s3_bucket}/findoc-llama-sec/model.tar.gz"
    async_output_s3 = f"s3://{s3_bucket}/findoc-llama-sec/async-output/"

    print(f"\nWould upload to: {model_s3_uri}")
    print(f"Would create SageMaker Model: {MODEL_NAME}")
    print(f"  image: a PyTorch inference container (e.g. the SageMaker-provided")
    print(f"         HuggingFace PyTorch inference image for your region/framework version)")
    print(f"  model_data: {model_s3_uri}")
    print(f"  entry_point: sagemaker/inference.py")
    print(f"  role: {role_arn}")
    print(f"  env: HF_TOKEN=<redacted, {len(hf_token)} chars>")
    print(f"\nWould create EndpointConfig: {ENDPOINT_CONFIG_NAME}")
    print(f"  instance_type: {INSTANCE_TYPE}, initial_instance_count: 1")
    print(f"  AsyncInferenceConfig.OutputConfig.S3OutputPath: {async_output_s3}")
    print(f"\nWould create Endpoint: {ENDPOINT_NAME}")
    print(f"\nWould then register a scalable target (min=0, max=1) and a")
    print(f"scaling policy on the HasNoActiveInferenceRequests predefined")
    print(f"metric via Application Auto Scaling, so the instance scales to")
    print(f"zero when idle -- this is a SEPARATE API call from endpoint")
    print(f"creation, not an EndpointConfig field.")

    if dry_run:
        print("\n[DRY RUN] No AWS resources were created. Pass --yes-really-deploy to actually deploy.")
        return

    # Real deployment. Imports here, not at module level, so --dry-run
    # (the default) never requires boto3/sagemaker to be installed.
    import boto3

    s3 = boto3.client("s3")
    sm = boto3.client("sagemaker")
    autoscaling = boto3.client("application-autoscaling")

    print(f"\nUploading {tarball} to {model_s3_uri}...")
    bucket, key = model_s3_uri.replace("s3://", "").split("/", 1)
    s3.upload_file(str(tarball), bucket, key)

    print(f"Creating model {MODEL_NAME}...")
    sm.create_model(
        ModelName=MODEL_NAME,
        PrimaryContainer={
            # Placeholder -- pick the real HuggingFace/PyTorch DLC URI for
            # your region and framework version before actually running this.
            "Image": "REPLACE_WITH_REAL_ECR_IMAGE_URI",
            "ModelDataUrl": model_s3_uri,
            "Environment": {"HF_TOKEN": hf_token},
        },
        ExecutionRoleArn=role_arn,
    )

    print(f"Creating endpoint config {ENDPOINT_CONFIG_NAME}...")
    sm.create_endpoint_config(
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
        ProductionVariants=[{
            "VariantName": "primary",
            "ModelName": MODEL_NAME,
            "InstanceType": INSTANCE_TYPE,
            "InitialInstanceCount": 1,
        }],
        AsyncInferenceConfig={
            "OutputConfig": {"S3OutputPath": async_output_s3},
        },
    )

    print(f"Creating endpoint {ENDPOINT_NAME}...")
    sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=ENDPOINT_CONFIG_NAME)

    resource_id = f"endpoint/{ENDPOINT_NAME}/variant/primary"
    print(f"Registering scale-to-zero autoscaling on {resource_id}...")
    autoscaling.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=0,
        MaxCapacity=1,
    )
    autoscaling.put_scaling_policy(
        PolicyName=f"{ENDPOINT_NAME}-scale-to-zero",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 1.0,
            "CustomizedMetricSpecification": {
                "MetricName": "HasNoActiveInferenceRequests",
                "Namespace": "AWS/SageMaker",
                "Dimensions": [
                    {"Name": "EndpointName", "Value": ENDPOINT_NAME},
                    {"Name": "VariantName", "Value": "primary"},
                ],
                "Statistic": "Average",
            },
            "ScaleInCooldown": 300,
            "ScaleOutCooldown": 60,
        },
    )
    print(f"\nDone. Endpoint '{ENDPOINT_NAME}' is provisioning -- check status with:")
    print(f"  aws sagemaker describe-endpoint --endpoint-name {ENDPOINT_NAME}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--hf-token", default=None, help="Defaults to $HF_TOKEN env var")
    parser.add_argument("--s3-bucket", required=True, help="Bucket to upload model.tar.gz and receive async output to")
    parser.add_argument(
        "--yes-really-deploy", action="store_true",
        help="Actually create AWS resources (billable). Without this flag, prints a dry run only.",
    )
    args = parser.parse_args()

    import os
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("::error:: --hf-token or $HF_TOKEN is required (Llama 3.1 is a gated model).")
        sys.exit(1)

    deploy(args.adapter_dir, args.role_arn, hf_token, args.s3_bucket, dry_run=not args.yes_really_deploy)


if __name__ == "__main__":
    main()
