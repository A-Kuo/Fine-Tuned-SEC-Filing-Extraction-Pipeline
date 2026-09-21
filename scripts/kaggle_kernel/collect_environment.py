"""Record exactly what a training run executed on: library versions, GPU,
git commit, and the hashes of the data files it read. Written next to the
run report so a result can always be traced to the environment that produced
it (Kaggle's image drifts independently of this repo).

Usage:
    python scripts/kaggle_kernel/collect_environment.py --out reports/environment.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PACKAGES = [
    "torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate",
    "datasets", "mlflow", "huggingface_hub", "tokenizers", "safetensors",
]
DATA_FILES = [
    "data/sec_filings_train.jsonl",
    "data/sec_filings_train.chat.jsonl",
    "data/sec_filings_test.jsonl",
]


def package_versions(names: list[str] = PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def file_fingerprint(path: Path) -> dict | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def gpu_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"cuda_available": False, "note": "torch not installed"}
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "name": props.name,
        "total_memory_gb": round(props.total_memory / 1e9, 1),
        "torch_cuda_version": torch.version.cuda,
    }


def git_commit(repo_dir: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True, timeout=15
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def collect(repo_dir: Path) -> dict:
    return {
        "schema_version": 1,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit(repo_dir),
        "packages": package_versions(),
        "gpu": gpu_info(),
        "data_files": {name: file_fingerprint(repo_dir / name) for name in DATA_FILES},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Record the training environment")
    p.add_argument("--repo-dir", default=".")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    report = collect(Path(args.repo_dir).resolve())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Environment written to {out}")


if __name__ == "__main__":
    main()
