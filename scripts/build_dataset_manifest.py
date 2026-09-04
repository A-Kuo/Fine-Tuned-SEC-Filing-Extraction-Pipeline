"""Wrap an existing training/test JSONL file (produced by
scripts/format_data.py / scripts/download_dataset.py, already exercised in
CI) into a versioned, checksummed DatasetManifest -- with malformed records
quarantined rather than silently dropped.

This needs no GPU/Docker: it only reads files CI already generates.

Usage:
    python scripts/build_dataset_manifest.py data/sec_filings_train.jsonl --split train
    python scripts/build_dataset_manifest.py data/sec_filings_test.jsonl --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.core.dataset_manifest import (
    DatasetManifest,
    compute_dataset_checksum,
    compute_record_checksum,
    make_dataset_version,
)
from src.core.dataset_schemas import ExtractedTargetRecord
from src.core.dataset_validator import (
    compute_field_null_rates,
    validate_and_quarantine,
    write_manifest_summary,
    write_quarantine,
)

TARGET_FIELDS = [
    "filing_id", "company_name", "ticker", "filing_type", "date",
    "fiscal_year_end", "revenue", "net_income", "total_assets",
    "total_liabilities", "eps", "sector",
]


def load_training_jsonl(path: Path) -> list[dict]:
    """Map the repo's {id, instruction, input, output} training-example shape
    onto ExtractedTargetRecord's {record_id, template_family, target} shape."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        try:
            target = json.loads(raw["output"]) if isinstance(raw.get("output"), str) else raw.get("output")
        except (json.JSONDecodeError, KeyError):
            target = None
        records.append({
            "record_id": raw.get("id", ""),
            "template_family": "synthetic_v1",
            "target": target if isinstance(target, dict) else {},
            "schema_version": 1,
        })
    return records


def main():
    parser = argparse.ArgumentParser(description="Build a versioned dataset manifest")
    parser.add_argument("input", type=str, help="Path to a training/test JSONL file")
    parser.add_argument(
        "--split", required=True,
        choices=["train", "val", "test", "benchmark_synthetic", "benchmark_real",
                 "benchmark_adversarial", "benchmark_ood"],
    )
    parser.add_argument("--template-family", type=str, default="synthetic_v1")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    records = load_training_jsonl(input_path)
    valid, quarantined = validate_and_quarantine(records, ExtractedTargetRecord, "pending")

    checksums = [compute_record_checksum(r) for r in valid]
    dataset_checksum = compute_dataset_checksum(checksums)
    generated_at = datetime.now(timezone.utc)
    dataset_version = make_dataset_version(
        schema_version=1, source_split=args.split, template_family=args.template_family,
        generated_at=generated_at, checksum=dataset_checksum,
    )

    manifest = DatasetManifest(
        dataset_version=dataset_version,
        source_split=args.split,
        template_family=args.template_family,
        checksum=dataset_checksum,
        record_count=len(valid),
        quarantined_count=len(quarantined),
        created_at=generated_at.isoformat(),
        lineage={"source_file": str(input_path)},
    )

    manifest_dir = REPO_ROOT / "data" / "manifests"
    quarantine_dir = REPO_ROOT / "data" / "quarantine"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / f"{dataset_version}.manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    quarantine_path = write_quarantine(quarantined, dataset_version, quarantine_dir)

    null_rates = compute_field_null_rates([r["target"] for r in valid], TARGET_FIELDS)
    summary_path = write_manifest_summary(
        dataset_version, valid, quarantined, null_rates, manifest_dir,
    )

    print(f"Dataset version: {dataset_version}")
    print(f"  Valid records:       {len(valid)}")
    print(f"  Quarantined records: {len(quarantined)}")
    print(f"  Manifest:  {manifest_path.relative_to(REPO_ROOT)}")
    print(f"  Summary:   {summary_path.relative_to(REPO_ROOT)}")
    if quarantine_path:
        print(f"  Quarantine: {quarantine_path.relative_to(REPO_ROOT)}")
    print("  Field null rates:")
    for field, rate in sorted(null_rates.items(), key=lambda kv: -kv[1]):
        if rate > 0:
            print(f"    {field}: {rate:.1%}")


if __name__ == "__main__":
    main()
