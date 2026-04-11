"""Download base model from HuggingFace Hub.

Downloads the Llama 3.1 8B model (or specified alternative) for fine-tuning.
Supports resumable downloads and token authentication for gated models.

Usage:
    python scripts/download_model.py --model meta-llama/Llama-3.1-8B
    python scripts/download_model.py --model meta-llama/Llama-3.1-8B --quantize-check
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download, login
from loguru import logger
from rich.console import Console
from rich.progress import Progress

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config

console = Console()


def download_model(model_id: str, cache_dir: str | None = None, token: str | None = None) -> Path:
    """Download model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model identifier (e.g., 'meta-llama/Llama-3.1-8B').
        cache_dir: Local directory to cache model. Defaults to HF cache.
        token: HuggingFace API token for gated models.

    Returns:
        Path to downloaded model directory.
    """
    if token:
        login(token=token)
        logger.info("Authenticated with HuggingFace Hub")

    logger.info(f"Downloading model: {model_id}")
    console.print(f"[bold blue]Downloading[/bold blue] {model_id}")
    console.print("This may take a while for large models (~4.5GB for 8B params)...")

    try:
        model_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            # Only download model weights + config, skip README etc.
            ignore_patterns=["*.md", "*.txt", "original/*"],
            resume_download=True,
        )
        logger.info(f"Model downloaded to: {model_path}")
        console.print(f"[bold green][OK][/bold green] Model saved to: {model_path}")
        return Path(model_path)

    except Exception as e:
        logger.error(f"Download failed: {e}")
        console.print(f"[bold red][FAIL][/bold red] Download failed: {e}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  1. For gated models (Llama): Accept license at huggingface.co/{model_id}")
        console.print("  2. Set HF_TOKEN in .env or pass --token")
        console.print("  3. Check internet connection")
        raise


def verify_model(model_path: Path) -> bool:
    """Verify downloaded model has required files.

    Checks for config.json and at least one weight shard—the minimum
    needed to load the model for QLoRA fine-tuning.
    """
    required_files = ["config.json"]
    weight_patterns = ["*.safetensors", "*.bin", "model*.safetensors"]

    for req in required_files:
        if not (model_path / req).exists():
            logger.error(f"Missing required file: {req}")
            return False

    # Check for at least one weight file
    has_weights = any(
        list(model_path.glob(pattern))
        for pattern in weight_patterns
    )
    if not has_weights:
        logger.error("No model weight files found (.safetensors or .bin)")
        return False

    console.print("[bold green][OK][/bold green] Model verification passed")
    return True


def check_quantization_ready() -> dict:
    """Check if system is ready for 4-bit quantization.

    Returns dict with compatibility info for QLoRA fine-tuning:
    - CUDA availability
    - bitsandbytes installation
    - Available GPU memory
    """
    import torch

    info = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": None,
        "gpu_memory_gb": None,
        "bitsandbytes_available": False,
    }

    if info["cuda_available"]:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / 1e9, 1
        )

    try:
        import bitsandbytes
        info["bitsandbytes_available"] = True
    except ImportError:
        pass

    # Display results
    console.print("\n[bold]System Quantization Check:[/bold]")
    for key, val in info.items():
        status = "[green][OK][/green]" if val else "[red][NO][/red]"
        console.print(f"  {status} {key}: {val}")

    if info["gpu_memory_gb"] and info["gpu_memory_gb"] >= 8:
        console.print("\n[green]System ready for QLoRA 4-bit fine-tuning![/green]")
    elif info["cuda_available"]:
        console.print("\n[yellow]GPU detected but may have limited memory. "
                       "Reduce batch_size in config.yaml.[/yellow]")
    else:
        console.print("\n[yellow]No GPU detected. Fine-tuning will be very slow. "
                       "Consider using a cloud GPU.[/yellow]")

    return info


def main():
    parser = argparse.ArgumentParser(description="Download base model for fine-tuning")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model ID (default: from config.yaml)",
    )
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace API token")
    parser.add_argument(
        "--quantize-check",
        action="store_true",
        help="Check system readiness for quantization",
    )
    args = parser.parse_args()

    config = load_config()
    model_id = args.model or config["model"]["base_model"]

    if args.quantize_check:
        check_quantization_ready()
        return

    model_path = download_model(model_id, args.cache_dir, args.token)
    verify_model(model_path)


if __name__ == "__main__":
    main()
