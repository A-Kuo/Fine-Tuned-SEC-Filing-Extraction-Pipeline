"""Model loading utilities for inference.

Loads the quantized Llama base model + QLoRA adapter for production inference.
Supports two modes:
    1. Full load: base model (NF4) + merged LoRA adapter
    2. Separate load: base model + adapter loaded independently (for vLLM)

The key optimization: at inference time we can merge the LoRA weights into
the base model: W_merged = W_base + BA, eliminating the adapter overhead.
This gives us the accuracy of fine-tuning with the speed of a single model.

Memory footprint at inference:
    Base (NF4):     7.2 GB
    LoRA adapter:   0.2 GB (or 0 if merged)
    KV cache:       ~0.5 GB (depends on batch size)
    Total:          ~7.4-7.9 GB
"""

import time
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig,
)

from src.config import load_config


class FinancialLLM:
    """Wrapper for the fine-tuned financial extraction model.

    Encapsulates model loading, generation config, and inference.
    Designed for both single-document and batch inference.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        generation_config: GenerationConfig,
        device: str = "auto",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.device = device
        self._model_version = None

    @classmethod
    def from_pretrained(
        cls,
        base_model_id: str | None = None,
        adapter_path: str | None = None,
        merge_adapter: bool = True,
        device: str = "auto",
        max_tokens: int = 512,
    ) -> "FinancialLLM":
        """Load fine-tuned model from base + adapter.

        Args:
            base_model_id: HuggingFace model ID or local path.
            adapter_path: Path to LoRA adapter directory.
            merge_adapter: If True, merge LoRA into base weights (faster inference).
            device: Device mapping strategy.
            max_tokens: Maximum tokens to generate.

        Returns:
            Initialized FinancialLLM ready for inference.
        """
        config = load_config()
        base_model_id = base_model_id or config["model"]["base_model"]
        adapter_path = adapter_path or config["model"]["adapter_path"]

        logger.info(f"Loading model: {base_model_id}")
        logger.info(f"Adapter: {adapter_path}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            padding_side="left",  # Left padding for batch generation
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load base model with 4-bit quantization
        quant_config = config["quantization"]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config["load_in_4bit"],
            bnb_4bit_compute_dtype=getattr(torch, quant_config["bnb_4bit_compute_dtype"]),
            bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"],
        )

        start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=torch.float16,
        )
        load_time = time.time() - start
        logger.info(f"Base model loaded in {load_time:.1f}s")

        # Load LoRA adapter
        adapter_dir = Path(adapter_path)
        if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
            logger.info("Loading LoRA adapter...")
            model = PeftModel.from_pretrained(model, adapter_path)

            if merge_adapter:
                logger.info("Merging adapter into base model...")
                model = model.merge_and_unload()
                logger.info("Adapter merged (zero overhead at inference)")
        else:
            logger.warning(f"No adapter found at {adapter_path}. Using base model only.")

        model.eval()

        # Generation config optimized for JSON extraction
        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            do_sample=False,  # Greedy for deterministic extraction
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        instance = cls(model, tokenizer, gen_config, device)
        instance._model_version = f"llama-sec-v1-{adapter_path.split('/')[-1]}"

        mem_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        logger.info(f"Model ready. GPU memory: {mem_mb:.0f} MB")

        return instance

    @classmethod
    def from_config(cls) -> "FinancialLLM":
        """Load model using default config.yaml settings."""
        return cls.from_pretrained()

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
    ) -> tuple[str, float]:
        """Generate extraction from a single prompt.

        Args:
            prompt: Formatted extraction prompt.
            max_tokens: Override max generation length.

        Returns:
            (generated_text, latency_ms) tuple.
        """
        gen_config = self.generation_config
        if max_tokens:
            gen_config = GenerationConfig(**{
                **gen_config.to_dict(),
                "max_new_tokens": max_tokens,
            })

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length or 2048,
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
            )
        latency_ms = (time.time() - start) * 1000

        # Decode only the generated tokens (not the prompt)
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][prompt_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return text.strip(), latency_ms

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int | None = None,
    ) -> list[tuple[str, float]]:
        """Generate extractions for a batch of prompts.

        Batching amortizes the fixed costs of GPU kernel launches and
        memory transfers, improving throughput significantly.

        Args:
            prompts: List of formatted extraction prompts.
            max_tokens: Override max generation length.

        Returns:
            List of (generated_text, latency_ms) tuples.
        """
        gen_config = self.generation_config
        if max_tokens:
            gen_config = GenerationConfig(**{
                **gen_config.to_dict(),
                "max_new_tokens": max_tokens,
            })

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length or 2048,
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
            )
        total_latency_ms = (time.time() - start) * 1000
        per_doc_latency = total_latency_ms / len(prompts)

        results = []
        for i, output_ids in enumerate(outputs):
            prompt_length = inputs["input_ids"][i].ne(self.tokenizer.pad_token_id).sum()
            generated_ids = output_ids[prompt_length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append((text.strip(), per_doc_latency))

        return results

    @property
    def model_version(self) -> str:
        return self._model_version or "unknown"

    def get_memory_stats(self) -> dict:
        """Return current GPU memory usage statistics."""
        if not torch.cuda.is_available():
            return {"gpu_available": False}

        return {
            "gpu_available": True,
            "allocated_mb": torch.cuda.memory_allocated() / 1e6,
            "reserved_mb": torch.cuda.memory_reserved() / 1e6,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
        }
