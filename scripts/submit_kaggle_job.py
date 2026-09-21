"""Kaggle Notebook Training Job Launcher.

Pushes scripts/kaggle_kernel/ (a GPU-enabled Kaggle kernel that clones this
repo and runs training/train.py) to Kaggle and polls until it finishes.
This is the primary remote training path; local GPU training via
`make train` is the fallback when Kaggle is unavailable.

Requires KAGGLE_USERNAME + KAGGLE_KEY in .env (or ~/.kaggle/kaggle.json).
Requires HF_TOKEN configured as a Kaggle Notebook secret (Kaggle UI ->
Notebook -> Add-ons -> Secrets) so the kernel can download the gated base
model. Optional secrets (HF_WRITE_TOKEN, HF_ADAPTER_REPO, MLFLOW_TRACKING_URI,
DAGSHUB_USER_TOKEN) are listed in scripts/kaggle_kernel/train_kernel.py.

Usage:
    # Push and wait for the training kernel to complete (full training run)
    python scripts/submit_kaggle_job.py --wait

    # A short end-to-end check first: 8 examples, 1 epoch, no upload
    python scripts/submit_kaggle_job.py --wait --mode smoke

    # Push without waiting (check status later)
    python scripts/submit_kaggle_job.py

    # Check status of a previously pushed kernel
    python scripts/submit_kaggle_job.py --status-only
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.config import load_config

KERNEL_DIR = Path(__file__).parent / "kaggle_kernel"
RUN_MODES = ("smoke", "full")
_RUN_MODE_LINE = re.compile(r'^RUN_MODE = "[^"]*"', re.MULTILINE)


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
            "Set KAGGLE_USERNAME + KAGGLE_KEY in .env / ~/.kaggle/kaggle.json "
            "(kaggle.com/settings -> API -> Create New Token). "
            "To sync Kaggle editor -> repo first: make pull-kaggle-kernel. "
            "For local fallback, run: make train"
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


def prepare_kernel_dir(mode: str) -> Path:
    """A throwaway copy of the kernel folder with RUN_MODE set to `mode`.

    Kaggle uploads only the metadata and the code file, so the mode has to be
    written into the code itself; a sidecar config file would never arrive.
    The checked-in train_kernel.py is left untouched.
    """
    if mode not in RUN_MODES:
        raise ValueError(f"mode must be one of {RUN_MODES}, got {mode!r}")

    meta = (KERNEL_DIR / "kernel-metadata.json").read_text(encoding="utf-8")
    code = (KERNEL_DIR / "train_kernel.py").read_text(encoding="utf-8")
    patched, count = _RUN_MODE_LINE.subn(f'RUN_MODE = "{mode}"', code)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one RUN_MODE assignment in train_kernel.py, found {count}"
        )

    tmp = Path(tempfile.mkdtemp(prefix="kaggle_kernel_"))
    (tmp / "kernel-metadata.json").write_text(meta, encoding="utf-8")
    (tmp / "train_kernel.py").write_text(patched, encoding="utf-8")
    return tmp


def push_kernel(api, config: dict, mode: str = "full") -> str:
    slug = kernel_slug(config)
    logger.info(f"Pushing training kernel '{slug}' to Kaggle (mode={mode})...")
    kernel_dir = prepare_kernel_dir(mode)
    try:
        api.kernels_push(str(kernel_dir))
    finally:
        shutil.rmtree(kernel_dir, ignore_errors=True)
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
    p.add_argument(
        "--mode", choices=RUN_MODES, default="full",
        help="smoke = 8 examples, 1 epoch, no upload; full = the whole training run (default)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    api = _get_kaggle_api()

    slug = kernel_slug(config)

    if not args.status_only:
        push_kernel(api, config, args.mode)

    if args.wait or args.status_only:
        final_state = poll_status(api, slug, args.poll_interval, args.timeout)
        if final_state != "complete":
            sys.exit(1)


if __name__ == "__main__":
    main()
