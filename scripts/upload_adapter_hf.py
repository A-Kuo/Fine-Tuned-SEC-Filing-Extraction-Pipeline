"""Publish a trained LoRA adapter to a Hugging Face repo (private by default).

Only an allowlist of adapter files is uploaded. The training output directory
also holds trainer checkpoints (with optimizer state, hundreds of MB) and a
local MLFlow store; none of that belongs in the published repo.

Settings
--------
    repo:   --repo, or HF_ADAPTER_REPO / config.yaml huggingface.adapter_repo
    token:  HF_WRITE_TOKEN, else HF_TOKEN (must be write-scoped to upload)
    private: config.yaml huggingface.private (default true)

Missing repo or token is reported as "skipped", not an error: a run should
still finish and say plainly that nothing was published. Keep the repo private
until the Llama 3.1 Community License terms for publishing derivatives have
been reviewed.

Usage:
    python scripts/upload_adapter_hf.py --adapter models/llama-sec-v1
    python scripts/upload_adapter_hf.py --adapter models/llama-sec-v1 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

BASE_MODEL = "meta-llama/Llama-3.1-8B"
GITHUB_URL = "https://github.com/A-Kuo/Fine-Tuned-SEC-Filing-Extraction-Pipeline"
REPO_ID_PATTERN = re.compile(r"^[\w.-]+/[\w.-]+$")

# Files that make up a usable adapter, plus the two provenance files.
UPLOAD_ALLOWLIST = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "training_metrics.json",
    "training_run_info.json",
]


def resolve_settings(config: dict, environ: dict | None = None, repo_override: str | None = None) -> dict:
    env = os.environ if environ is None else environ
    hf_cfg = config.get("huggingface", {})
    repo = (repo_override or env.get("HF_ADAPTER_REPO") or hf_cfg.get("adapter_repo") or "").strip()
    token = env.get("HF_WRITE_TOKEN") or env.get("HF_TOKEN") or ""
    return {"repo": repo, "token": token, "private": bool(hf_cfg.get("private", True))}


def adapter_files(adapter_dir: str | Path) -> list[str]:
    """The allowlisted files that actually exist in the adapter directory."""
    root = Path(adapter_dir)
    return [name for name in UPLOAD_ALLOWLIST if (root / name).is_file()]


def build_readme(repo_id: str) -> str:
    """A deliberately number-free model card: it states what this is and
    where the evidence lives, so it can never drift from measured results."""
    return f"""---
base_model: {BASE_MODEL}
library_name: peft
license: llama3.1
tags:
  - lora
  - qlora
  - sec-filings
  - information-extraction
---

# {repo_id}

A QLoRA (LoRA rank 16, 4-bit NF4) adapter for [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})
that extracts structured fields (company, form type, dates, revenue, net income,
total assets, total liabilities, EPS, sector) from SEC filing text.

- **Training data:** a small set of templated synthetic filings, not real filings.
- **Evaluation:** a base-versus-adapter comparison on held-out synthetic examples
  is in the project repository. Accuracy on real SEC filings has **not** been measured.
- **Use:** subject to the Llama 3.1 Community License. Validate extracted values
  against the primary filing before relying on them.

Provenance for this exact adapter (dataset hash, sizes, hardware) is in
`training_run_info.json`. Project, code and evidence: {GITHUB_URL}
"""


def upload(
    adapter_dir: str | Path,
    settings: dict,
    dry_run: bool = False,
    api=None,
) -> dict:
    """Upload the adapter. Returns a status dict; never raises for the
    expected "nothing to do" cases."""
    files = adapter_files(adapter_dir)
    result = {"status": "skipped", "repo": settings["repo"], "private": settings["private"], "files": files}

    if not settings["repo"]:
        result["reason"] = "no repo configured (set HF_ADAPTER_REPO or huggingface.adapter_repo)"
        return result
    if not REPO_ID_PATTERN.match(settings["repo"]):
        result["reason"] = f"repo id {settings['repo']!r} is not of the form owner/name"
        return result
    if "adapter_model.safetensors" not in files or "adapter_config.json" not in files:
        result["reason"] = f"no adapter weights/config found in {adapter_dir}"
        return result
    if dry_run:
        result["reason"] = "dry run: nothing uploaded"
        return result
    if not settings["token"]:
        result["reason"] = "no HF_WRITE_TOKEN or HF_TOKEN available"
        return result

    try:
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi(token=settings["token"])

        api.create_repo(repo_id=settings["repo"], private=settings["private"], exist_ok=True)

        readme = Path(adapter_dir) / "README.md"
        readme.write_text(build_readme(settings["repo"]), encoding="utf-8")
        api.upload_folder(
            folder_path=str(adapter_dir),
            repo_id=settings["repo"],
            allow_patterns=[*UPLOAD_ALLOWLIST, "README.md"],
            commit_message="Upload QLoRA adapter",
        )
    except Exception as e:  # noqa: BLE001 - report every failure mode, never crash the run
        logger.error(f"Hugging Face upload failed: {e}")
        result.update(status="failed", reason=str(e))
        return result

    result.update(status="uploaded", reason=None, files=[*files, "README.md"])
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publish a trained adapter to Hugging Face")
    p.add_argument("--adapter", default="models/llama-sec-v1")
    p.add_argument("--repo", default=None, help="owner/name (default: HF_ADAPTER_REPO or config)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=None, help="Write the result JSON here")
    return p.parse_args()


def main() -> None:
    from src.core.config import load_config

    args = parse_args()
    settings = resolve_settings(load_config(), repo_override=args.repo)
    result = upload(args.adapter, settings, dry_run=args.dry_run)

    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
