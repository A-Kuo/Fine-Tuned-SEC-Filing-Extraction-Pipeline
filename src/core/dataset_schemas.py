"""Contracts for data that moves through the training/evaluation pipeline:
raw filing input, extracted target labels, model predictions, evaluation
records, and monitoring records.

These are separate from src/core/schemas.py's FilingRecord family, which
describes the *extraction pipeline's own output shape* (sections, metrics,
risk factors). This module describes the surrounding data-engineering
lifecycle: what corpus a record came from, what produced a prediction, and
how a metric traces back to a model/prompt/parser/dataset version. See
docs/TRUTH_AUDIT.md for why lineage tracking didn't exist before this.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RawFilingInputRecord(BaseModel):
    """One filing's raw text as it enters the pipeline, before extraction."""

    record_id: str
    source_type: Literal["synthetic", "real_edgar"]
    source_path: str  # relative to repo root -- absolute paths aren't portable
    checksum_sha256: str
    fetched_at: str | None = None
    cik: str | None = None
    accession_no: str | None = None


class ExtractedTargetRecord(BaseModel):
    """Ground-truth label for a filing, used for training or evaluation."""

    record_id: str
    template_family: str | None = None  # e.g. "synthetic_v1", "real_edgar"
    target: dict
    schema_version: int = 1


class PredictionRecord(BaseModel):
    """One model prediction, carrying full lineage back to what produced it."""

    record_id: str
    model_version: str
    adapter_version: str | None = None
    prompt_version: str
    parser_version: str
    parse_stage: str | None = None  # winning stage from ParseTelemetry, if used
    prediction: dict
    latency_ms: float | None = None


class EvaluationRecord(BaseModel):
    """One prediction's scoring result against ground truth."""

    record_id: str
    dataset_version: str
    prediction_record_id: str
    field_results: dict[str, dict]
    error_tags: list[str] = Field(default_factory=list)
    schema_conformant: bool


class MonitoringRecord(BaseModel):
    """One measured monitoring data point, feeding drift detection."""

    metric_name: str
    dataset_version: str | None = None
    value: float
    sample_size: int
    measured_at: str
