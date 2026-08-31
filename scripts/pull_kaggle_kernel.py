"""Pull the canonical Kaggle training kernel into scripts/kaggle_kernel/.

Use this before editing train_kernel.py locally so the repo matches what is
already on Kaggle (avoids make train-kaggle clobbering editor changes).

Requires Kaggle API auth: KAGGLE_USERNAME + KAGGLE_KEY in .env (or
~/.kaggle/kaggle.json), generated at kaggle.com/settings -> API -> Create
New Token.

Usage:
    python scripts/pull_kaggle_kernel.py
    python scripts/pull_kaggle_kernel.py --slug other-user/other-kernel
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.config import load_config

KERNEL_DIR = Path(__file__).parent / "kaggle_kernel"
DEFAULT_CODE_FILE = "train_kernel.py"


def _slug_from_config(config: dict, override: str | None) -> str:
    if override:
        return override
    slug = config.get("kaggle", {}).get("kernel_slug")
    if not slug:
        raise SystemExit(
            "config.yaml -> kaggle.kernel_slug is not set. "
            "Pass --slug <username>/findoc-qlora-train"
        )
    return slug


def _pull_with_cli(slug: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "kernels", "pull", slug, "-m", "-p", str(dest)]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _install_pulled_files(staging: Path) -> list[str]:
    """Copy pulled kernel files into KERNEL_DIR; return list of updated paths."""
    updated: list[str] = []
    metadata = staging / "kernel-metadata.json"
    if metadata.exists():
        shutil.copy2(metadata, KERNEL_DIR / "kernel-metadata.json")
        updated.append(str(KERNEL_DIR / "kernel-metadata.json"))

    # Script kernels use code_file from metadata; notebook kernels use .ipynb
    code_name = DEFAULT_CODE_FILE
    if metadata.exists():
        import json

        meta = json.loads(metadata.read_text())
        code_name = meta.get("code_file") or DEFAULT_CODE_FILE

    candidates = [staging / code_name]
    candidates.extend(staging.glob("*.ipynb"))
    candidates.extend(staging.glob("*.py"))

    copied_code = False
    for src in candidates:
        if not src.exists() or src.name == "kernel-metadata.json":
            continue
        if src.suffix == ".ipynb":
            dest = KERNEL_DIR / src.name
        else:
            dest = KERNEL_DIR / DEFAULT_CODE_FILE
        shutil.copy2(src, dest)
        updated.append(str(dest))
        copied_code = True
        break

    if not copied_code:
        raise SystemExit(
            f"No kernel code file found in {staging}. "
            "Check Kaggle auth and kernel slug."
        )
    return updated


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pull Kaggle kernel into scripts/kaggle_kernel/")
    p.add_argument("--slug", default=None, help="Override kernel slug (default: config.yaml)")
    p.add_argument("--dry-run", action="store_true", help="Print slug only; do not pull")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    slug = _slug_from_config(config, args.slug)
    print(f"Kernel slug: {slug}")

    if args.dry_run:
        return

    with tempfile.TemporaryDirectory(prefix="kaggle-kernel-pull-") as tmp:
        staging = Path(tmp)
        _pull_with_cli(slug, staging)
        updated = _install_pulled_files(staging)

    print("Updated:")
    for path in updated:
        print(f"  {path}")
    print(
        "\nReview the diff, commit if correct, then make train-kaggle will push "
        "this same content back to Kaggle."
    )


if __name__ == "__main__":
    main()
