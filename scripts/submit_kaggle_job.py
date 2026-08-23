"""Kaggle Notebook Training Job Launcher.

Pushes scripts/kaggle_kernel/ (a GPU-enabled Kaggle kernel that clones this
repo and runs training/train.py) to Kaggle and polls until it finishes.
This is the primary remote training path; local GPU training via
`make train` is the fallback when Kaggle is unavailable.

Requires KAGGLE_USERNAME + KAGGLE_KEY in .env (or ~/.kaggle/kaggle.json).
Requires DAGSHUB_USER_TOKEN and HF_TOKEN configured as Kaggle Notebook
secrets (Kaggle UI -> Notebook -> Add-ons -> Secrets) so the kernel can
authenticate to DagsHub/MLFlow and HuggingFace.

Usage:
    # Push and wait for the training kernel to complete
    python scripts/submit_kaggle_job.py --wait

    # Push without waiting (check status later)
    python scripts/submit_kaggle_job.py

    # Check status of a previously pushed kernel
    python scripts/submit_kaggle_job.py --status-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config

KERNEL_DIR = Path(__file__).parent / "kaggle_kernel"


def _get_kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise SystemExit(
            "kaggle package not installed. Run: pip install kaggle\n"
            "Falling back to local training is recommended: make train"
        )

    api = KaggleApi()
    try:
        api.authenticate()
    except (Exception, SystemExit) as e:
        # KaggleApi.authenticate() calls exit(1) (SystemExit, not Exception) and
        # prints its own help text when credentials are missing — catch both so
        # our more specific guidance is shown too.
        raise SystemExit(
            f"Kaggle authentication failed ({e}).\n"
            "Set KAGGLE_USERNAME + KAGGLE_KEY in .env, or place credentials "
            "at ~/.kaggle/kaggle.json. Falling back to local training is "
            "recommended: make train"
        )
    return api


def kernel_slug(config: dict) -> str:
    slug = config["kaggle"].get("kernel_slug")
    if not slug:
        raise SystemExit(
            "config.yaml -> kaggle.kernel_slug is not set. "
            "Set it to '<kaggle_username>/findoc-qlora-train' and update "
            "scripts/kaggle_kernel/kernel-metadata.json 'id' to match."
        )
    return slug


def push_kernel(api, config: dict) -> str:
    slug = kernel_slug(config)
    logger.info(f"Pushing training kernel '{slug}' to Kaggle...")
    api.kernels_push(str(KERNEL_DIR))
    logger.info("Kernel pushed. Kaggle will now provision a GPU instance and run it.")
    return slug


def poll_status(api, slug: str, interval_s: int = 30, timeout_s: int = 7200) -> str:
    logger.info(f"Polling status for '{slug}' every {interval_s}s (timeout {timeout_s}s)...")
    elapsed = 0
    while elapsed < timeout_s:
        status = api.kernels_status(slug)
        state = getattr(status, "status", None) or status.get("status", "unknown")
        logger.info(f"Status: {state}  (elapsed {elapsed}s)")

        if state in ("complete",):
            logger.info(f"Training kernel finished successfully: {slug}")
            return state
        if state in ("error", "cancelAcknowledged"):
            logger.error(f"Training kernel failed: {slug} (status={state})")
            return state

        time.sleep(interval_s)
        elapsed += interval_s

    logger.warning(f"Timed out after {timeout_s}s waiting for kernel '{slug}'")
    return "timeout"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit QLoRA training job to Kaggle Notebooks")
    p.add_argument("--wait", action="store_true", help="Block until the kernel finishes")
    p.add_argument("--status-only", action="store_true", dest="status_only",
                   help="Only check status of the existing kernel; do not push")
    p.add_argument("--poll-interval", type=int, default=30, dest="poll_interval")
    p.add_argument("--timeout", type=int, default=7200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    api = _get_kaggle_api()

    slug = kernel_slug(config)

    if not args.status_only:
        push_kernel(api, config)

    if args.wait or args.status_only:
        poll_status(api, slug, args.poll_interval, args.timeout)


if __name__ == "__main__":
    main()
