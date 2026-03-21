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
"""

import argparse
import json
import sys
from pathlib import Path

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
from trl import SFTTrainer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config, get_project_root
from training.callbacks import MetricsCallback, EarlyStoppingOnLoss
from training.data_collator import FinancialDataCollator


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

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
        if torch.cuda.is_available()
        else "eager",
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


def create_training_args(config: dict, output_dir: str) -> TrainingArguments:
    """Create HuggingFace TrainingArguments from config."""
    train_cfg = config["training"]

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        max_grad_norm=train_cfg["max_grad_norm"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        fp16=train_cfg["fp16"],
        seed=train_cfg["seed"],
        optim="paged_adamw_8bit",  # Memory-efficient optimizer for QLoRA
        report_to="none",  # Disable wandb/tensorboard for prototype
        save_total_limit=3,
        load_best_model_at_end=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )


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
    data_path = dataset_path or config["data"]["train_path"]
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

    # ── Step 5: Training arguments ──
    training_args = create_training_args(config, out_dir)

    # ── Step 6: SFTTrainer ──
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        max_seq_length=config["model"]["max_seq_length"],
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

    logger.info(f"Training complete. Metrics: {metrics}")
    logger.info(f"Adapter saved to: {out_dir}")
    logger.info(f"Adapter size: {sum(f.stat().st_size for f in Path(out_dir).rglob('*') if f.is_file()) / 1e6:.1f} MB")

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
