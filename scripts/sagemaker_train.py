"""AWS SageMaker Training Job Launcher for QLoRA Fine-Tuning.

Launches a managed SageMaker training job using the HuggingFace estimator,
which handles cluster provisioning, distributed training, and artifact
storage in S3.

Why SageMaker for QLoRA fine-tuning:
    - Managed ml.g5.2xlarge instances with A10G GPUs (24 GB VRAM)
    - NF4 quantization + LoRA keeps peak VRAM under 14 GB (fits on g5.2xlarge)
    - 77% memory reduction vs FP32: 32 GB → 7.2 GB frozen weights
    - Spot instance support cuts training cost by ~70%
    - S3-backed checkpointing survives spot interruptions

Pricing comparison (us-east-1, 2025):
    GPT-4 fine-tuning (OpenAI):  ~$1,200 for equivalent extraction quality
    SageMaker g5.2xlarge on-demand: ~$8 per training run  (~150× cheaper)
    SageMaker g5.2xlarge spot:      ~$2.40 per training run (~500× cheaper)

Usage:
    # Launch a training job with defaults
    python scripts/sagemaker_train.py

    # Override instance type and epochs
    python scripts/sagemaker_train.py --instance-type ml.g5.4xlarge --epochs 5

    # Use spot instances for cost savings
    python scripts/sagemaker_train.py --spot --max-wait 7200

    # Dry-run: print the estimator config without launching
    python scripts/sagemaker_train.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from loguru import logger


def build_hyperparameters(args: argparse.Namespace) -> dict:
    """Translate CLI args into SageMaker hyperparameter dict (all values are strings)."""
    return {
        "model_name_or_path": args.base_model,
        "num_train_epochs": str(args.epochs),
        "per_device_train_batch_size": str(args.batch_size),
        "gradient_accumulation_steps": str(args.grad_accum),
        "learning_rate": str(args.learning_rate),
        "lora_r": str(args.lora_r),
        "lora_alpha": str(args.lora_alpha),
        "lora_dropout": str(args.lora_dropout),
        "max_seq_length": str(args.max_seq_length),
        "output_dir": "/opt/ml/model",
        "logging_steps": "10",
        "save_strategy": "epoch",
        "load_best_model_at_end": "true",
        "bf16": "true" if args.bf16 else "false",
    }


def launch_training_job(args: argparse.Namespace) -> str:
    """
    Configure and launch a SageMaker HuggingFace training job.

    Returns the training job name.
    """
    import sagemaker
    from sagemaker.huggingface import HuggingFace

    session = sagemaker.Session(
        boto_session=__import__("boto3").Session(
            region_name=args.region,
            profile_name=args.profile or None,
        )
    )
    role = args.role or sagemaker.get_execution_role(session)

    job_name = f"findoc-qlora-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    hyperparameters = build_hyperparameters(args)

    # ---------------------------------------------------------------------------
    # Source directory: the repo root is packaged and uploaded to S3 by SageMaker
    # ---------------------------------------------------------------------------
    repo_root = str(Path(__file__).parent.parent)

    estimator = HuggingFace(
        entry_point="train.py",
        source_dir=repo_root,
        role=role,
        instance_type=args.instance_type,
        instance_count=1,
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        hyperparameters=hyperparameters,
        sagemaker_session=session,
        output_path=f"s3://{args.s3_bucket}/findoc-qlora/output/",
        checkpoint_s3_uri=f"s3://{args.s3_bucket}/findoc-qlora/checkpoints/{job_name}/",
        # Spot instance configuration
        use_spot_instances=args.spot,
        max_run=args.max_run,
        max_wait=args.max_wait if args.spot else None,
        # Metrics to surface in SageMaker console + CloudWatch
        metric_definitions=[
            {"Name": "train:loss",       "Regex": r"'loss': ([0-9\.]+)"},
            {"Name": "train:epoch",      "Regex": r"'epoch': ([0-9\.]+)"},
            {"Name": "eval:accuracy",    "Regex": r"eval_accuracy.*?([0-9\.]+)"},
            {"Name": "eval:field_f1",    "Regex": r"field_f1.*?([0-9\.]+)"},
        ],
        tags=[
            {"Key": "Project",     "Value": "FinDocAnalyzer"},
            {"Key": "Component",   "Value": "QLoRA-fine-tuning"},
            {"Key": "BaseModel",   "Value": args.base_model.replace("/", "_")},
        ],
        environment={
            "PYTHONIOENCODING": "utf-8",
            "TRANSFORMERS_CACHE": "/tmp/hf_cache",
        },
    )

    # Training data from S3 (upload local data first with scripts/upload_data.py)
    data_channels = {
        "train": f"s3://{args.s3_bucket}/findoc-qlora/data/train/",
        "validation": f"s3://{args.s3_bucket}/findoc-qlora/data/validation/",
    }

    logger.info(
        "Launching SageMaker training job: {}  instance={}  spot={}",
        job_name, args.instance_type, args.spot,
    )

    if args.dry_run:
        logger.info("DRY-RUN — estimator config:")
        logger.info(json.dumps(hyperparameters, indent=2))
        logger.info("Job name would be: {}", job_name)
        logger.info("Data channels: {}", data_channels)
        return job_name

    estimator.fit(data_channels, job_name=job_name, wait=args.wait)

    logger.info("Training job '{}' submitted successfully.", job_name)
    logger.info(
        "Monitor at: https://{}.console.aws.amazon.com/sagemaker/home?region={}#/jobs/{}",
        args.region, args.region, job_name,
    )
    return job_name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch QLoRA fine-tuning on AWS SageMaker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Cloud config
    p.add_argument("--region",         default=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    p.add_argument("--profile",        default=os.getenv("AWS_PROFILE"),
                   help="AWS CLI profile name")
    p.add_argument("--role",           default=os.getenv("SAGEMAKER_ROLE"),
                   help="SageMaker execution role ARN (auto-detected on EC2/SageMaker)")
    p.add_argument("--s3-bucket",      default=os.getenv("S3_BUCKET", "findoc-ml-artifacts"),
                   dest="s3_bucket")
    p.add_argument("--instance-type",  default="ml.g5.2xlarge", dest="instance_type",
                   help="SageMaker training instance type")

    # Spot instances
    p.add_argument("--spot",           action="store_true",
                   help="Use managed spot training (up to ~70% cost reduction)")
    p.add_argument("--max-run",        type=int, default=3600, dest="max_run",
                   help="Maximum training run time in seconds")
    p.add_argument("--max-wait",       type=int, default=7200, dest="max_wait",
                   help="Maximum wait time for spot capacity (seconds; spot only)")

    # Model & training hyperparameters
    p.add_argument("--base-model",     default="meta-llama/Llama-3.1-8B", dest="base_model")
    p.add_argument("--epochs",         type=int,   default=3)
    p.add_argument("--batch-size",     type=int,   default=4,   dest="batch_size")
    p.add_argument("--grad-accum",     type=int,   default=8,   dest="grad_accum")
    p.add_argument("--learning-rate",  type=float, default=2e-4, dest="learning_rate")
    p.add_argument("--lora-r",         type=int,   default=16,  dest="lora_r")
    p.add_argument("--lora-alpha",     type=int,   default=32,  dest="lora_alpha")
    p.add_argument("--lora-dropout",   type=float, default=0.05, dest="lora_dropout")
    p.add_argument("--max-seq-length", type=int,   default=2048, dest="max_seq_length")
    p.add_argument("--bf16",           action="store_true", default=True,
                   help="Use bfloat16 mixed precision")

    # Execution flags
    p.add_argument("--wait",     action="store_true",
                   help="Block until the training job completes")
    p.add_argument("--dry-run",  action="store_true", dest="dry_run",
                   help="Print config and exit without launching")

    return p.parse_args()


if __name__ == "__main__":
    launch_training_job(parse_args())
