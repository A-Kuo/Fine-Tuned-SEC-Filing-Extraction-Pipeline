"""Runs INSIDE a Kaggle Notebook (kernel), on the same clone-and-install path
as train_kernel.py. Trains each config in training/ablation_config.py's
ABLATION_CONFIGS sequentially, logging one row per config to
results/ablation_results.jsonl so the LoRA rank / target-module choice
currently just asserted in the README can be justified by measurement
instead.

This CANNOT run without a GPU -- there is no fallback/simulate path, unlike
the CPU-only report scripts elsewhere in this repo. See
training/ablation_config.py's own docstring for exactly which of the 5
requested ablation axes are wired into training/train.py today (lora_r and
target_modules; the other three are defined but not yet wired).

Secrets required (Kaggle Notebook -> Add-ons -> Secrets), same as
train_kernel.py: DAGSHUB_USER_TOKEN, HF_TOKEN.

Usage (on Kaggle, after the standard clone-and-install steps
train_kernel.py performs):
    python scripts/kaggle_kernel/run_ablation_sweep.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.ablation_config import ABLATION_CONFIGS
from training.train import train


def run_sweep() -> list[dict]:
    results = []

    for config in ABLATION_CONFIGS:
        print(f"\n{'=' * 60}\nRunning ablation config: {config.name} ({config.config_id})\n{'=' * 60}")

        start = time.time()
        try:
            metrics = train(
                output_dir=f"models/ablation-{config.config_id}",
                config_overrides=config.to_config_overrides(),
            )
            train_time_min = (time.time() - start) / 60
            row = {
                "config_name": config.name,
                "config_id": config.config_id,
                "lora_r": config.lora_r,
                "target_modules": list(config.target_modules),
                "train_time_min": round(train_time_min, 2),
                "final_train_loss": metrics.get("train_loss"),
                "status": "success",
            }
        except Exception as e:
            row = {
                "config_name": config.name,
                "config_id": config.config_id,
                "status": "failed",
                "error": str(e),
            }
            print(f"Config {config.name} failed: {e}")

        results.append(row)

    return results


def main():
    results = run_sweep()

    out_path = REPO_ROOT / "results" / "ablation_results.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    print(f"\n{'=' * 60}")
    print("Ablation sweep complete:")
    for row in results:
        status = row["status"]
        detail = f"loss={row.get('final_train_loss')}" if status == "success" else row.get("error", "")
        print(f"  {row['config_name']} ({row['config_id']}): {status} -- {detail}")
    print(f"\nResults written to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
