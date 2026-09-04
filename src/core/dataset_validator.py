"""Validate a batch of raw records against a Pydantic schema, quarantining
malformed ones instead of silently dropping them.

A dataset build that silently drops bad records produces a manifest whose
record_count looks fine but whose provenance is now a lie -- nobody can tell
later whether 1000 records were processed or 1000 records *survived*.
Quarantining preserves both counts and the original content for inspection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Type

from pydantic import BaseModel, ValidationError


def validate_and_quarantine(
    records: Iterable[dict],
    schema: Type[BaseModel],
    dataset_version: str,
) -> tuple[list[dict], list[dict]]:
    """Validate each record against `schema`.

    Returns (valid, quarantined). `valid` entries are the original dicts
    (not re-serialized from the model, so extra caller-side fields survive).
    `quarantined` entries are `{"record": <original>, "errors": [...],
    "reason_code": "schema_validation_failed"}` -- never silently dropped.
    """
    valid: list[dict] = []
    quarantined: list[dict] = []

    for record in records:
        try:
            schema.model_validate(record)
        except ValidationError as e:
            quarantined.append({
                "record": record,
                "errors": [
                    {"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]}
                    for err in e.errors()
                ],
                "reason_code": "schema_validation_failed",
            })
            continue
        valid.append(record)

    return valid, quarantined


def write_quarantine(
    quarantined: list[dict],
    dataset_version: str,
    quarantine_dir: Path,
) -> Path | None:
    """Write quarantined records to a JSONL file. Returns None (writes
    nothing) if there is nothing to quarantine -- an empty file would read as
    "quarantine ran and found issues" when it didn't run at all."""
    if not quarantined:
        return None

    quarantine_dir.mkdir(parents=True, exist_ok=True)
    out_path = quarantine_dir / f"{dataset_version}.quarantine.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in quarantined:
            f.write(json.dumps(entry, default=str) + "\n")
    return out_path


def compute_field_null_rates(records: list[dict], fields: list[str]) -> dict[str, float]:
    """Per-field fraction of records where the field is missing or None --
    a cheap, honest completeness signal for the manifest summary."""
    if not records:
        return {field: 0.0 for field in fields}

    return {
        field: sum(1 for r in records if r.get(field) is None) / len(records)
        for field in fields
    }


def write_manifest_summary(
    dataset_version: str,
    valid: list[dict],
    quarantined: list[dict],
    field_null_rates: dict[str, float],
    manifest_dir: Path,
) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out_path = manifest_dir / f"{dataset_version}.summary.json"
    summary = {
        "dataset_version": dataset_version,
        "valid_count": len(valid),
        "quarantined_count": len(quarantined),
        "field_null_rates": field_null_rates,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path
