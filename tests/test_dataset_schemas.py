"""Tests for src/core/dataset_schemas.py."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.dataset_schemas import (
    EvaluationRecord,
    ExtractedTargetRecord,
    MonitoringRecord,
    PredictionRecord,
    RawFilingInputRecord,
)


class TestRawFilingInputRecord:
    def test_valid_synthetic_record(self):
        r = RawFilingInputRecord(
            record_id="r-1", source_type="synthetic",
            source_path="data/sample_10k.txt", checksum_sha256="abc123",
        )
        assert r.source_type == "synthetic"
        assert r.cik is None

    def test_invalid_source_type_rejected(self):
        with pytest.raises(ValidationError):
            RawFilingInputRecord(
                record_id="r-1", source_type="scraped",
                source_path="x", checksum_sha256="abc",
            )


class TestExtractedTargetRecord:
    def test_defaults(self):
        t = ExtractedTargetRecord(record_id="r-1", target={"revenue": "1"})
        assert t.schema_version == 1
        assert t.template_family is None


class TestPredictionRecord:
    def test_carries_full_lineage(self):
        p = PredictionRecord(
            record_id="r-1", model_version="llama-sec-v1",
            adapter_version="adapter-2026-09-02", prompt_version="v1",
            parser_version="0.1.0+abc123", parse_stage="direct",
            prediction={"revenue": "1"}, latency_ms=120.5,
        )
        assert p.parse_stage == "direct"
        assert p.adapter_version == "adapter-2026-09-02"

    def test_adapter_version_optional(self):
        p = PredictionRecord(
            record_id="r-1", model_version="base", prompt_version="v1",
            parser_version="v1", prediction={},
        )
        assert p.adapter_version is None


class TestEvaluationRecord:
    def test_error_tags_default_empty(self):
        e = EvaluationRecord(
            record_id="r-1", dataset_version="v1-test-x-20260101T000000Z-aaaaaaaa",
            prediction_record_id="p-1", field_results={}, schema_conformant=True,
        )
        assert e.error_tags == []

    def test_error_tags_independent_between_instances(self):
        """Regression guard: a mutable default (list()) shared across
        instances would let one record's tags leak into another's."""
        e1 = EvaluationRecord(
            record_id="r-1", dataset_version="v1", prediction_record_id="p-1",
            field_results={}, schema_conformant=True,
        )
        e2 = EvaluationRecord(
            record_id="r-2", dataset_version="v1", prediction_record_id="p-2",
            field_results={}, schema_conformant=False,
        )
        e1.error_tags.append("wrong_field")
        assert e2.error_tags == []


class TestMonitoringRecord:
    def test_dataset_version_optional(self):
        m = MonitoringRecord(
            metric_name="cache_hit_rate", value=0.42, sample_size=100,
            measured_at="2026-09-02T00:00:00Z",
        )
        assert m.dataset_version is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
