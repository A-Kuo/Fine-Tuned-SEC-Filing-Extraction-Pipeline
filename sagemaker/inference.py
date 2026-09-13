"""SageMaker inference handler for the fine-tuned QLoRA extraction model.

Implements the standard SageMaker inference-toolkit contract
(model_fn/input_fn/predict_fn/output_fn) so this can be deployed as a
custom PyTorch inference container. Reuses this repo's own FinancialLLM
(src/extraction/model.py) rather than reimplementing model loading --
local serving and SageMaker serving load the model identically.

Model packaging: the LoRA adapter (small, ~85MB per README.md's real
measurement) is what ships in model.tar.gz uploaded to S3 -- NOT the base
Llama 3.1 8B weights (multi-GB). The base model is pulled fresh from
HuggingFace at container startup (model_fn), keeping the S3 artifact small
and avoiding re-uploading weights that are already versioned on the Hub.
This requires HF_TOKEN to be set in the endpoint's environment (SageMaker
Model.env) since Llama 3.1 is a gated model.

Real, honest status: this is deployment code, not a claim that a live
SageMaker endpoint exists. No AWS resources have been created by writing
this file -- see deploy_async_endpoint.py for that (explicit, separate,
costs real money) step.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# SageMaker's inference toolkit runs this file directly inside the
# container; it doesn't install this repo as a package, so make src/
# importable the same way every other script here does.
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.extraction.inference import SYSTEM_PROMPT, EXTRACTION_INSTRUCTION


def model_fn(model_dir: str):
    """Load base model (from HuggingFace Hub) + LoRA adapter (from model_dir,
    populated by SageMaker from the S3 model.tar.gz this endpoint was
    created with). Called once per container instance, not per request.

    Imports FinancialLLM here rather than at module level -- it pulls in
    peft/torch/bitsandbytes, which aren't installed in every environment
    that might import this module (e.g. running its own unit tests against
    input_fn/predict_fn/output_fn's pure logic, with no model loaded).
    """
    from src.extraction.model import FinancialLLM

    logger.info(f"Loading model for SageMaker inference, adapter_path={model_dir}")
    return FinancialLLM.from_pretrained(adapter_path=model_dir, merge_adapter=True)


def input_fn(request_body: str, content_type: str = "application/json") -> dict:
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    return json.loads(request_body)


def predict_fn(input_data: dict, model: FinancialLLM) -> dict:
    """input_data shape mirrors serving/api.py's request models:
        {"task": "extract", "text": "...", "max_tokens": 512}
        {"task": "rag_query", "prompt": "...", "max_tokens": 512}
    "task" lets one endpoint serve both /extract and /rag/query use cases
    -- both need the same loaded model, just a different prompt built by
    the caller (the RAG retrieval step itself stays out-of-endpoint, done
    by the proxy/caller before invoking, since it needs Postgres access
    this container doesn't have).
    """
    task = input_data.get("task", "extract")
    max_tokens = input_data.get("max_tokens", 512)

    if task == "extract":
        text = input_data["text"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{EXTRACTION_INSTRUCTION}\n\n{text}"},
        ]
        if getattr(model.tokenizer, "chat_template", None):
            prompt = model.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"{SYSTEM_PROMPT}\n\n{EXTRACTION_INSTRUCTION}\n\n{text}\n\nResponse:"
    elif task == "rag_query":
        prompt = input_data["prompt"]
    else:
        raise ValueError(f"Unknown task: {task}")

    raw_output, latency_ms = model.generate(prompt, max_tokens=max_tokens)
    return {
        "raw_output": raw_output,
        "latency_ms": latency_ms,
        "model_version": model.model_version,
    }


def output_fn(prediction: dict, accept: str = "application/json") -> str:
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps(prediction)
