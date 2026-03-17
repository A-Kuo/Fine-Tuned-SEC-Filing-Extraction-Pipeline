"""vLLM Inference Server for Financial LLM.

Wraps vLLM's high-performance serving engine with our extraction-specific
configuration. vLLM achieves ~2x throughput over naive HuggingFace generation
through two key optimizations:

1. **Continuous Batching**: Instead of waiting for all requests in a batch to
   finish, vLLM processes requests as they arrive and returns results as they
   complete. This eliminates idle GPU time from length mismatches.

2. **PagedAttention**: KV-cache is stored in non-contiguous memory pages
   (like OS virtual memory), eliminating fragmentation. This allows larger
   effective batch sizes within the same GPU memory.

Memory math with vLLM:
    Model weights (NF4):     7.2 GB
    KV cache (batch=32):     ~1.5 GB  (depends on sequence length)
    Overhead:                ~0.3 GB
    Total:                   ~9.0 GB  (fits on RTX 3090/4090 24GB)

Usage:
    # Start server
    python serving/inference_server.py --port 8000

    # With custom settings
    python serving/inference_server.py --port 8000 --max-batch-size 16 --gpu-memory 0.85
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config


def build_vllm_args(config: dict, cli_args: argparse.Namespace) -> dict:
    """Build vLLM engine arguments from config + CLI overrides.

    Returns kwargs dict for vllm.AsyncLLMEngine or the vllm serve CLI.
    """
    model_id = cli_args.model or config["model"]["base_model"]
    adapter_path = cli_args.adapter_path or config["model"]["adapter_path"]
    serving_cfg = config["serving"]

    engine_args = {
        "model": model_id,
        "tokenizer": model_id,
        "dtype": "float16",
        "max_model_len": config["model"]["max_seq_length"],
        "gpu_memory_utilization": cli_args.gpu_memory or 0.85,
        "max_num_batched_tokens": serving_cfg["max_batch_size"] * config["model"]["max_seq_length"],
        "max_num_seqs": serving_cfg["max_batch_size"],
        "trust_remote_code": True,
    }

    # Add quantization config
    quant_cfg = config["quantization"]
    if quant_cfg.get("load_in_4bit"):
        engine_args["quantization"] = "bitsandbytes"
        engine_args["load_format"] = "bitsandbytes"

    # Add LoRA adapter if it exists
    adapter_dir = Path(adapter_path)
    if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
        engine_args["enable_lora"] = True
        engine_args["lora_modules"] = [
            {"name": "financial-extraction", "path": str(adapter_dir)}
        ]
        logger.info(f"LoRA adapter enabled: {adapter_dir}")
    else:
        logger.warning(f"No adapter at {adapter_path}, using base model")

    return engine_args


def start_server_vllm(config: dict, cli_args: argparse.Namespace):
    """Start vLLM's built-in OpenAI-compatible API server.

    This gives us an OpenAI-compatible /v1/completions endpoint out of the box,
    which our FastAPI wrapper and batch inference scripts call into.
    """
    try:
        from vllm import LLM, SamplingParams
        from vllm.entrypoints.openai.api_server import run_server
    except ImportError:
        logger.error(
            "vLLM not installed. Install with: pip install vllm\n"
            "Note: vLLM requires Linux + CUDA GPU."
        )
        sys.exit(1)

    engine_args = build_vllm_args(config, cli_args)
    port = cli_args.port or config["serving"]["port"]
    host = cli_args.host or config["serving"]["host"]

    logger.info(f"Starting vLLM server on {host}:{port}")
    logger.info(f"Model: {engine_args['model']}")
    logger.info(f"Max batch size: {config['serving']['max_batch_size']}")
    logger.info(f"GPU memory utilization: {engine_args['gpu_memory_utilization']}")

    # vLLM's built-in server handles batching, caching, and concurrency
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass

    # Build vLLM CLI args
    vllm_cli_args = [
        "--model", engine_args["model"],
        "--dtype", engine_args["dtype"],
        "--max-model-len", str(engine_args["max_model_len"]),
        "--gpu-memory-utilization", str(engine_args["gpu_memory_utilization"]),
        "--host", host,
        "--port", str(port),
    ]

    if engine_args.get("quantization"):
        vllm_cli_args.extend(["--quantization", engine_args["quantization"]])

    if engine_args.get("enable_lora"):
        vllm_cli_args.extend(["--enable-lora"])

    logger.info(f"vLLM args: {' '.join(vllm_cli_args)}")

    # Use vLLM's entrypoint
    from vllm.entrypoints.openai import api_server
    import asyncio

    # Run the server
    sys.argv = ["vllm.entrypoints.openai.api_server"] + vllm_cli_args
    api_server.run_server(api_server.parse_args())


def start_server_fallback(config: dict, cli_args: argparse.Namespace):
    """Fallback: start a simple inference server using HuggingFace directly.

    Used when vLLM is not available (e.g., macOS, no GPU). Provides the same
    API interface but without vLLM's batching optimizations.
    """
    logger.warning("vLLM not available, starting fallback HuggingFace server")
    logger.info("For production throughput, use vLLM on a Linux GPU machine")

    # Import FastAPI app which handles fallback internally
    from serving.api import create_app

    import uvicorn

    port = cli_args.port or config["serving"]["port"]
    host = cli_args.host or config["serving"]["host"]

    app = create_app(config)
    uvicorn.run(app, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="Start Financial LLM inference server")
    parser.add_argument("--model", type=str, default=None, help="Base model ID")
    parser.add_argument("--adapter-path", type=str, default=None, help="LoRA adapter path")
    parser.add_argument("--host", type=str, default=None, help="Server host")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    parser.add_argument("--max-batch-size", type=int, default=None, help="Max batch size")
    parser.add_argument("--gpu-memory", type=float, default=None, help="GPU memory fraction (0-1)")
    parser.add_argument(
        "--fallback", action="store_true",
        help="Use HuggingFace fallback instead of vLLM",
    )
    args = parser.parse_args()

    config = load_config()

    if args.max_batch_size:
        config["serving"]["max_batch_size"] = args.max_batch_size

    if args.fallback:
        start_server_fallback(config, args)
    else:
        try:
            import vllm
            start_server_vllm(config, args)
        except ImportError:
            logger.warning("vLLM not found, falling back to HuggingFace server")
            start_server_fallback(config, args)


if __name__ == "__main__":
    main()
