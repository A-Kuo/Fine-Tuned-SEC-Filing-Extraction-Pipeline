"""Deterministic dataset versioning: checksums, version strings, and the
manifest that ties a dataset version to the lineage (model/adapter/prompt/
parser version) it was built or evaluated with.

A dataset_version string is meant to be citable evidence -- if two people
generate a manifest from the same input records, they get the same
dataset_version, because the checksum is order-independent and content-derived
rather than a timestamp or a counter.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def compute_record_checksum(record: dict) -> str:
    """sha256 over the record's canonical JSON form (sorted keys) so field
    order in the source dict never changes the checksum."""
    canonical = json.dumps(record, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_dataset_checksum(record_checksums: list[str]) -> str:
    """sha256 over the sorted, concatenated per-record checksums -- so the
    dataset checksum is independent of record order, only of content."""
    joined = "".join(sorted(record_checksums))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def make_dataset_version(
    schema_version: int,
    source_split: str,
    template_family: str,
    generated_at: datetime,
    checksum: str,
) -> str:
    timestamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    return f"v{schema_version}-{source_split}-{template_family}-{timestamp}-{checksum[:8]}"


class DatasetManifest(BaseModel):
    """Everything needed to know what a dataset version *is* and what
    produced/consumed it. Lineage fields are optional and filled in as they
    become known -- a freshly-built training manifest won't have
    model_version yet; an evaluation-run manifest will."""

    dataset_version: str
    schema_version: int = 1
    source_split: Literal[
        "train", "val", "test",
        "benchmark_synthetic", "benchmark_real",
        "benchmark_adversarial", "benchmark_ood",
    ]
    template_family: str
    checksum: str
    record_count: int
    quarantined_count: int = 0
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    lineage: dict = Field(default_factory=dict)
