"""Fetch a completed Kaggle training kernel's output and distill it into a
lightweight metrics.json for a coding agent to read.

Run after scripts/submit_kaggle_job.py --wait succeeds. Downloads the full
kernel output (weights, logs) into a scratch directory, locates the
training_metrics.json / training_log.json that training/train.py and
training/callbacks.py write, and converts them into a small, stable summary
committed at notebooks/results/metrics.json (+ a timestamped history copy
and an optional loss-curve plot).

Usage:
    python scripts/fetch_kaggle_results.py --status complete
    python scripts/fetch_kaggle_results.py --status error --kernel-slug user/slug
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.config import load_config
from scripts.submit_kaggle_job import _get_kaggle_api, kernel_slug

REPO_ROOT = Path(__file__).parent.parent
DOWNLOAD_DIR = REPO_ROOT / ".kaggle_output"
RESULTS_DIR = REPO_ROOT / "notebooks" / "results"
HISTORY_DIR = RESULTS_DIR / "history"
LOSS_CURVE_MAX_POINTS = 50


def build_metrics_summary(
    training_metrics: dict,
    training_log: list[dict],
    *,
    status: str,
    kernel_slug: str,
    git_commit_sha: str,
    kernel_version: int | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> dict:
    """Convert raw HuggingFace training outputs into the agent-facing schema.

    Uses .get() with fallbacks throughout: HF Trainer.metrics key names
    drift across transformers versions, and a missing field should degrade
    to null rather than crash the whole results fetch.
    """
    duration_s = None
    if started_at and completed_at:
        try:
            start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            end = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
            duration_s = (end - start).total_seconds()
        except ValueError:
            duration_s = None
    if duration_s is None and training_metrics.get("train_runtime") is not None:
        duration_s = training_metrics.get("train_runtime")

    final_train_loss = training_metrics.get("train_loss")
    final_epoch = training_metrics.get("epoch")
    total_steps = None

    loss_curve: list[dict] = []
    if training_log:
        total_steps = training_log[-1].get("step")
        loss_points = [entry for entry in training_log if "loss" in entry]
        loss_curve = _downsample(loss_points, LOSS_CURVE_MAX_POINTS)
        if final_train_loss is None and loss_points:
            final_train_loss = loss_points[-1].get("loss")
        if final_epoch is None and training_log:
            final_epoch = training_log[-1].get("epoch")

    return {
        "schema_version": 1,
        "status": status,
        "kernel_slug": kernel_slug,
        "kernel_version": kernel_version,
        "git_commit_sha": git_commit_sha,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_s": duration_s,
        "final_train_loss": final_train_loss,
        "final_epoch": final_epoch,
        "total_steps": total_steps,
        "loss_curve": [
            {"step": p.get("step"), "epoch": p.get("epoch"), "loss": p.get("loss")}
            for p in loss_curve
        ],
        "raw_hf_metrics": training_metrics,
    }


def _downsample(points: list[dict], max_points: int) -> list[dict]:
    """Keep at most max_points entries, always including the first and last."""
    if len(points) <= max_points:
        return points
    if max_points <= 2:
        return [points[0], points[-1]]
    step = (len(points) - 1) / (max_points - 1)
    indices = sorted({round(i * step) for i in range(max_points)})
    return [points[i] for i in indices]


def _find_json(root: Path, filename: str) -> dict | list | None:
    matches = list(root.rglob(filename))
    if not matches:
        logger.warning(f"{filename} not found under {root}")
        return None
    with open(matches[0]) as f:
        return json.load(f)


def _current_commit_sha() -> str:
    import os

    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"][:12]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _render_loss_curve_plot(training_log: list[dict], dest: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"Skipping loss curve plot (matplotlib unavailable: {e})")
        return

    points = [p for p in training_log if "loss" in p]
    if not points:
        logger.warning("No loss entries found; skipping loss curve plot")
        return

    try:
        steps = [p.get("step") for p in points]
        losses = [p.get("loss") for p in points]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, losses)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("Training loss")
        fig.tight_layout()
        fig.savefig(dest)
        plt.close(fig)
        logger.info(f"Loss curve plot saved to {dest}")
    except Exception as e:
        logger.warning(f"Failed to render loss curve plot: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch Kaggle kernel output and write notebooks/results/metrics.json"
    )
    p.add_argument("--kernel-slug", default=None, help="Override kernel slug (default: config.yaml)")
    p.add_argument("--commit-sha", default=None, help="Override git commit SHA")
    p.add_argument("--status", default="complete", choices=["complete", "error", "timeout"])
    p.add_argument("--kernel-version", type=int, default=None)
    p.add_argument("--started-at", default=None, help="ISO8601 timestamp")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    slug = args.kernel_slug or kernel_slug(config)
    commit_sha = args.commit_sha or _current_commit_sha()
    completed_at = datetime.now(timezone.utc).isoformat()

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    training_metrics: dict = {}
    training_log: list[dict] = []

    try:
        api = _get_kaggle_api()
        logger.info(f"Downloading kernel output for '{slug}' into {DOWNLOAD_DIR}...")
        api.kernels_output(slug, path=str(DOWNLOAD_DIR), force=True)

        training_metrics = _find_json(DOWNLOAD_DIR, "training_metrics.json") or {}
        training_log = _find_json(DOWNLOAD_DIR, "training_log.json") or []
    except Exception as e:
        logger.error(f"Failed to download/parse Kaggle kernel output: {e}")

    summary = build_metrics_summary(
        training_metrics,
        training_log,
        status=args.status,
        kernel_slug=slug,
        git_commit_sha=commit_sha,
        kernel_version=args.kernel_version,
        started_at=args.started_at,
        completed_at=completed_at,
    )
    if not training_metrics and not training_log:
        summary["note"] = "training_metrics.json/training_log.json not found in kernel output"

    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote {metrics_path}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    history_path = HISTORY_DIR / f"{ts}_{commit_sha}.json"
    with open(history_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote {history_path}")

    if training_log:
        _render_loss_curve_plot(training_log, RESULTS_DIR / "loss_curve.png")


if __name__ == "__main__":
    main()
