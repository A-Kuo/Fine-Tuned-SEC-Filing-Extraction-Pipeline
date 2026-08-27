"""QLoRA Fine-Tuning for SEC Filing Extraction.

Fine-tunes Llama 3.1 8B using QLoRA (Quantized Low-Rank Adaptation) to extract
structured financial data from SEC filings.

Mathematical Foundation:
    Standard fine-tuning updates all parameters W ∈ R^{d×k}:
        W' = W + ΔW    (ΔW has d×k parameters)

    LoRA decomposes the update into low-rank matrices:
        W' = W + BA     where B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k)

    This reduces trainable params from d×k to r×(d+k).
    For Llama 8B with r=16: 8B frozen → ~200M trainable (2.5% of total).

    QLoRA adds 4-bit NormalFloat quantization to the frozen weights:
        W_frozen stored in NF4 (7.2GB vs 32GB at FP32)
        BA adapters trained in FP16 for gradient stability

    The loss is standard cross-entropy over the output tokens:
        L = -Σ log P(y_t | y_{<t}, x; W_frozen + BA)
    where x is the filing text and y is the target JSON extraction.

Usage:
    python training/train.py
    python training/train.py --num_epochs 5 --learning_rate 2e-4
    python training/train.py --model meta-llama/Llama-3.1-8B --dataset data/sec_filings_train.jsonl

Runs locally by default (GPU required). Dataset is pulled from the Kaggle
dataset configured in config.yaml -> kaggle.dataset_id when available,
falling back to the local JSONL path otherwise. All runs are logged to
MLFlow (tracking server hosted on DagsHub); see configure_mlflow(). To run
training on Kaggle's hosted GPU compute instead, use
scripts/submit_kaggle_job.py.
"""

import argparse
import inspect
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import torch
from datasets import load_dataset
from loguru import logger
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config, get_project_root
from src.chat_template import ensure_chat_template
from training.callbacks import MetricsCallback, EarlyStoppingOnLoss
from training.data_collator import FinancialDataCollator


def configure_mlflow(config: dict) -> None:
    """Point MLFlow tracking at the DagsHub-hosted server for this repo.

    DAGSHUB_USER_TOKEN (from .env) authenticates non-interactively via
    dagshub.init(). Without it, dagshub.init() falls back to an interactive
    browser OAuth flow that blocks for minutes and cannot succeed on a
    headless box (e.g. a Kaggle kernel) — so that path is skipped entirely
    when the token isn't set, going straight to unauthenticated tracking.
    """
    mlflow_cfg = config["mlflow"]

    if os.environ.get("DAGSHUB_USER_TOKEN"):
        try:
            import dagshub

            dagshub.init(
                repo_owner=mlflow_cfg["dagshub_repo_owner"],
                repo_name=mlflow_cfg["dagshub_repo_name"],
                mlflow=True,
            )
            mlflow.set_experiment(mlflow_cfg["experiment_name"])
            return
        except Exception as e:
            logger.warning(f"dagshub.init() failed ({e}); falling back to MLFLOW_TRACKING_URI")
    else:
        logger.warning(
            "DAGSHUB_USER_TOKEN not set; skipping DagsHub auth. "
            "Set it in .env to log runs to the shared MLFlow experiment."
        )

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])


def resolve_dataset_path(config: dict, override_path: str | None = None) -> str:
    """Resolve the training data path.

    Primary source is Kaggle (config.kaggle.dataset_id); falls back to the
    local JSONL path if Kaggle credentials/dataset are unavailable.
    """
    if override_path:
        return override_path

    kaggle_cfg = config["kaggle"]
    dataset_id = kaggle_cfg.get("dataset_id")

    if dataset_id:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()

            dest_dir = get_project_root() / "data" / "kaggle_cache"
            dest_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading training data from Kaggle dataset: {dataset_id}")
            api.dataset_download_files(dataset_id, path=str(dest_dir), unzip=True)

            jsonl_files = list(dest_dir.glob("*.jsonl"))
            if jsonl_files:
                return str(jsonl_files[0])
            logger.warning(f"No .jsonl files found in Kaggle dataset {dataset_id}")
        except (Exception, SystemExit) as e:
            # kaggle's KaggleApi.authenticate() calls exit(1) (raises SystemExit,
            # not Exception) when no credentials are configured — must catch both
            # to fall back to local data instead of crashing training.
            logger.warning(f"Kaggle dataset download failed ({e})")

    if not kaggle_cfg.get("use_local_if_unavailable", True):
        raise RuntimeError(
            f"Kaggle dataset '{dataset_id}' unavailable and local fallback disabled."
        )

    local_path = kaggle_cfg["local_fallback"]
    logger.info(f"Using local training data: {local_path}")
    return local_path


def create_bnb_config(config: dict) -> BitsAndBytesConfig:
    """Create BitsAndBytes config for 4-bit quantization.

    NF4 (NormalFloat4) quantization maps FP16 weights to 4-bit values
    using a lookup table optimized for normally-distributed weights.
    Double quantization further compresses the quantization constants.

    Memory math:
        FP32: 8B params × 4 bytes = 32GB
        FP16: 8B params × 2 bytes = 16GB
        NF4:  8B params × 0.5 bytes + overhead ≈ 7.2GB
    """
    quant_cfg = config["quantization"]
    compute_dtype = getattr(torch, quant_cfg["bnb_4bit_compute_dtype"])

    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )


def create_lora_config(config: dict) -> LoraConfig:
    """Create LoRA configuration.

    Target modules are the attention projection matrices (Q, K, V, O)
    plus the MLP gate/up/down projections. These are the layers where
    low-rank adaptation is most effective for instruction following.

    Parameter count with r=16:
        Each target module adds 2 × d_model × r parameters
        For Llama 8B (d=4096), 7 target modules:
            7 × 2 × 4096 × 16 = 917,504 per layer
            × 32 layers = ~29.4M LoRA params
        Plus embeddings/head ≈ 200M total trainable
    """
    lora_cfg = config["lora"]

    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def single_device_map() -> dict | None:
    """Device map that pins the whole model to one GPU.

    `device_map="auto"` shards an 8B model across *every* visible GPU. On
    Kaggle's 2xT4 runtime that leaves half the layers on cuda:1, and a
    bitsandbytes-quantized model split that way cannot be trained: HF Trainer
    wraps any run that sees >1 GPU in `nn.DataParallel`, which then replicates
    batches onto cuda:1 and dies with "Expected all tensors to be on the same
    device, but found at least two devices, cuda:1 and cuda:0".

    An NF4-quantized 8B model fits in a single 16 GB T4, so pinning is both
    correct and sufficient. See also the `_n_gpu` override in
    create_training_args() -- both are needed, because CUDA_VISIBLE_DEVICES
    has no effect once torch has already initialised CUDA.
    """
    return {"": 0} if torch.cuda.is_available() else None


def load_base_model(
    model_id: str,
    bnb_config: BitsAndBytesConfig,
    max_seq_length: int = 2048,
) -> tuple:
    """Load base model with 4-bit quantization + tokenizer.

    Returns:
        (model, tokenizer) tuple ready for LoRA adapter injection.
    """
    logger.info(f"Loading base model: {model_id} (4-bit quantized)")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="right",
        truncation_side="right",
        model_max_length=max_seq_length,
    )

    # Llama models don't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # The base (non-Instruct) checkpoint ships no chat template, so SFTTrainer
    # raises the moment it tries to render the `messages` column. Install the
    # shared one -- inference applies the identical template, which is what
    # keeps the trained prompt shape and the served prompt shape in sync.
    if ensure_chat_template(tokenizer):
        logger.info("Installed Llama 3.1 chat template (base checkpoint had none)")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        # Match the outer dtype to bnb_4bit_compute_dtype (fp16 by default).
        # T4/older GPUs have no native bf16 tensor cores; loading in bf16
        # while quantization computes in fp16 and the trainer runs fp16 AMP
        # creates a 3-way dtype mismatch that inflates memory and can OOM.
        torch_dtype=bnb_config.bnb_4bit_compute_dtype,
        device_map=single_device_map(),
        attn_implementation="sdpa",  # <-- force non-flash attention
        # max_seq_length is usually passed via config, not as a kwarg
    )

    # Prepare model for k-bit training:
    # - Freezes quantized layers
    # - Casts layer norms to FP32 for training stability
    # - Enables gradient checkpointing to save memory
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # Log model size info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Base model loaded: {total_params / 1e9:.1f}B total parameters")

    return model, tokenizer


def prepare_dataset(
    data_path: str,
    tokenizer,
    max_seq_length: int = 2048,
    max_samples: int | None = None,
):
    """Load and prepare dataset for SFTTrainer.

    Loads JSONL with chat-format messages, applies tokenizer chat template.
    SFTTrainer handles the rest (masking instruction tokens so loss is only
    computed on the assistant's response tokens).
    """
    logger.info(f"Loading dataset from {data_path}")

    # Check if chat-formatted version exists
    chat_path = Path(data_path).with_suffix(".chat.jsonl")
    if chat_path.exists():
        load_path = str(chat_path)
        logger.info(f"Using pre-formatted chat data: {load_path}")
    else:
        load_path = data_path
        logger.info("Using raw data (will format on-the-fly)")

    dataset = load_dataset("json", data_files=load_path, split="train")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    logger.info(f"Dataset loaded: {len(dataset)} examples")
    return dataset


def to_text_dataset(dataset, tokenizer):
    """Render conversational rows into a single `text` column.

    SFTTrainer can auto-detect a `messages` column, but *when* and *how* it
    applies the chat template changed repeatedly across TRL versions. Rendering
    it ourselves makes the exact training string explicit and identical on
    every version -- and it is rendered with the same template
    ExtractionEngine uses at inference, so the two cannot drift apart.
    """
    if "text" in dataset.column_names:
        return dataset

    if "messages" not in dataset.column_names:
        raise ValueError(
            f"Dataset has neither a 'text' nor a 'messages' column "
            f"(found: {dataset.column_names}). Run scripts/format_data.py."
        )

    def _render(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    # Drop the original columns: only `text` should reach the collator.
    return dataset.map(_render, remove_columns=dataset.column_names)


def formatting_func(example: dict) -> str:
    """Format a single example for SFTTrainer.

    If data is in chat format (has 'messages'), applies the tokenizer's
    chat template. Otherwise falls back to alpaca-style formatting.
    """
    if "messages" in example:
        # Chat format - let SFTTrainer handle via dataset_text_field
        # Return the messages as-is for the collator
        return example["messages"]

    # Fallback: alpaca format
    if "text" in example:
        return example["text"]

    # Raw format: construct prompt
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )


def create_training_args(config: dict, output_dir: str) -> SFTConfig:
    """Build the SFTConfig, tolerating TRL's renamed arguments.

    TRL renamed SFTConfig's `max_seq_length` to `max_length`; which spelling is
    valid depends on the installed version, and Kaggle's image drifts
    independently of this repo. Rather than pin a version (or monkeypatch the
    file at runtime), candidate kwargs are filtered against the real signature
    so the same code runs against either.
    """
    train_cfg = config["training"]
    max_seq_length = config["model"]["max_seq_length"]

    # MLflow tracking points at the DagsHub-hosted server, which needs a token.
    # Without one the callback still fires on every logging step and fails, so
    # on Kaggle (no token) we don't report at all.
    report_to = ["mlflow"] if os.environ.get("DAGSHUB_USER_TOKEN") else []

    candidate_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": train_cfg["num_epochs"],
        "per_device_train_batch_size": train_cfg["batch_size"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "learning_rate": float(train_cfg["learning_rate"]),
        "weight_decay": train_cfg["weight_decay"],
        "warmup_ratio": train_cfg.get("warmup_ratio", 0.0),
        "lr_scheduler_type": train_cfg["lr_scheduler_type"],
        "max_grad_norm": train_cfg["max_grad_norm"],
        "logging_steps": train_cfg["logging_steps"],
        "save_strategy": "epoch",
        "save_total_limit": 3,
        "load_best_model_at_end": False,
        "fp16": train_cfg["fp16"],
        "seed": train_cfg["seed"],
        "optim": "paged_adamw_8bit",
        "report_to": report_to,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataset_text_field": "text",
        # TRL <0.20 spells this max_seq_length, >=0.20 spells it max_length.
        # Exactly one survives the filter below.
        "max_seq_length": max_seq_length,
        "max_length": max_seq_length,
    }

    valid_params = inspect.signature(SFTConfig.__init__).parameters
    args = SFTConfig(**{k: v for k, v in candidate_kwargs.items() if k in valid_params})

    # Touch the property first so transformers populates _n_gpu, then force it
    # to 1. Trainer wraps any run that reports >1 GPU in nn.DataParallel, which
    # a 4-bit quantized model cannot survive: it scatters the batch onto every
    # device (breaking a model pinned to cuda:0) and chokes on the 0-dim
    # num_items_in_batch scalar with "chunk expects at least a 1-dimensional
    # tensor". Setting CUDA_VISIBLE_DEVICES is *not* an alternative -- it has
    # no effect once torch has already initialised CUDA, which any earlier
    # torch.cuda call in a notebook will have done.
    _ = args.n_gpu
    args._n_gpu = 1

    return args


def train(
    model_id: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    max_samples: int | None = None,
):
    """Main training function.

    Orchestrates the full QLoRA fine-tuning pipeline:
    1. Load config
    2. Create quantization config (NF4 4-bit)
    3. Load base model (quantized)
    4. Inject LoRA adapters
    5. Load + format dataset
    6. Train with SFTTrainer
    7. Save adapter weights (~200MB)
    """
    config = load_config()

    # Override config with CLI args
    if num_epochs:
        config["training"]["num_epochs"] = num_epochs
    if batch_size:
        config["training"]["batch_size"] = batch_size
    if learning_rate:
        config["training"]["learning_rate"] = learning_rate

    model_name = model_id or config["model"]["base_model"]
    data_path = resolve_dataset_path(config, dataset_path)
    out_dir = output_dir or config["training"]["output_dir"]
    max_samp = max_samples or config["data"].get("max_train_samples")

    logger.info("=" * 60)
    logger.info("QLoRA Fine-Tuning: SEC Filing Extraction")
    logger.info("=" * 60)
    logger.info(f"Base model:      {model_name}")
    logger.info(f"Dataset:         {data_path}")
    logger.info(f"Output:          {out_dir}")
    logger.info(f"Epochs:          {config['training']['num_epochs']}")
    logger.info(f"Batch size:      {config['training']['batch_size']}")
    logger.info(f"Learning rate:   {config['training']['learning_rate']}")
    logger.info(f"LoRA rank:       {config['lora']['r']}")
    logger.info("=" * 60)

    configure_mlflow(config)
    mlflow_cfg = config["mlflow"]
    run_name = f"{mlflow_cfg['run_name_prefix']}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "base_model": model_name,
            "lora_r": config["lora"]["r"],
            "lora_alpha": config["lora"]["lora_alpha"],
            "lora_dropout": config["lora"]["lora_dropout"],
            "compute_source": "kaggle" if config["kaggle"].get("dataset_id") else "local",
        })

        # ── Step 1: Quantization config ──
        bnb_config = create_bnb_config(config)

        # ── Step 2: Load base model (4-bit) ──
        model, tokenizer = load_base_model(
            model_name,
            bnb_config,
            config["model"]["max_seq_length"],
        )

        # ── Step 3: Inject LoRA adapters ──
        lora_config = create_lora_config(config)
        model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            f"LoRA injected: {trainable / 1e6:.1f}M trainable / "
            f"{total / 1e6:.0f}M total ({100 * trainable / total:.2f}%)"
        )

        # ── Step 4: Load dataset ──
        dataset = prepare_dataset(
            data_path, tokenizer, config["model"]["max_seq_length"], max_samp
        )
        dataset = to_text_dataset(dataset, tokenizer)

        # ── Step 5: Training arguments ──
        training_args = create_training_args(config, out_dir)

        # ── Step 6: SFTTrainer ──
        trainer = SFTTrainer(
            model=model,
            args=training_args,           # this is SFTConfig
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[
                MetricsCallback(),
                EarlyStoppingOnLoss(patience=5, min_delta=0.01),
            ],
        )

        # ── Step 7: Train ──
        logger.info("Starting training...")
        train_result = trainer.train()

        # ── Step 8: Save adapter ──
        logger.info(f"Saving adapter to {out_dir}")
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

        # Save training metrics
        metrics = train_result.metrics
        metrics_path = Path(out_dir) / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # ── Step 9: Log adapter to MLFlow + register in model registry ──
        mlflow.log_artifacts(out_dir, artifact_path="adapter")
        mlflow.register_model(
            f"runs:/{run.info.run_id}/adapter",
            name=mlflow_cfg["registered_model_name"],
        )

    logger.info(f"Training complete. Metrics: {metrics}")
    logger.info(f"Adapter saved to: {out_dir}")
    logger.info(f"Adapter size: {sum(f.stat().st_size for f in Path(out_dir).rglob('*') if f.is_file()) / 1e6:.1f} MB")
    logger.info(f"MLFlow run: {mlflow_cfg['tracking_uri']}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama for SEC filing extraction")
    parser.add_argument("--model", type=str, default=None, help="Base model ID")
    parser.add_argument("--dataset", type=str, default=None, help="Training data path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit training examples")
    args = parser.parse_args()

    train(
        model_id=args.model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
