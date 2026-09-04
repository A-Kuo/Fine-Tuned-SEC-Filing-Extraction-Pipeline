"""Tests for src/core/dataset_validator.py.

The core guarantee under test: a malformed record is never silently
dropped -- it always shows up somewhere (quarantined, with its original
content and a reason), so counts stay honest.
"""

import json
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.dataset_validator import (
    compute_field_null_rates,
    validate_and_quarantine,
    write_manifest_summary,
    write_quarantine,
)


class Widget(BaseModel):
    widget_id: str
    count: int


class TestValidateAndQuarantine:
    def test_all_valid_records_pass_through(self):
        records = [{"widget_id": "a", "count": 1}, {"widget_id": "b", "count": 2}]
        valid, quarantined = validate_and_quarantine(records, Widget, "v1")
        assert valid == records
        assert quarantined == []

    def test_malformed_record_is_quarantined_not_dropped(self):
        records = [{"widget_id": "a", "count": 1}, {"widget_id": "b"}]  # missing count
        valid, quarantined = validate_and_quarantine(records, Widget, "v1")
        assert len(valid) == 1
        assert len(quarantined) == 1
        assert quarantined[0]["record"] == {"widget_id": "b"}
        assert quarantined[0]["reason_code"] == "schema_validation_failed"

    def test_quarantine_preserves_original_record_with_extra_fields(self):
        """valid[] keeps the original dict (not a re-serialized model), so
        caller-side fields the schema doesn't know about survive."""
        records = [{"widget_id": "a", "count": 1, "extra_field": "kept"}]
        valid, _ = validate_and_quarantine(records, Widget, "v1")
        assert valid[0]["extra_field"] == "kept"

    def test_quarantine_records_error_details(self):
        records = [{"widget_id": "a", "count": "not-a-number"}]
        _, quarantined = validate_and_quarantine(records, Widget, "v1")
        assert len(quarantined) == 1
        assert quarantined[0]["errors"][0]["loc"] == ["count"]

    def test_empty_input(self):
        valid, quarantined = validate_and_quarantine([], Widget, "v1")
        assert valid == []
        assert quarantined == []


class TestWriteQuarantine:
    def test_writes_nothing_when_empty(self, tmp_path):
        result = write_quarantine([], "v1", tmp_path)
        assert result is None
        assert list(tmp_path.glob("*")) == []

    def test_writes_jsonl_when_nonempty(self, tmp_path):
        quarantined = [{"record": {"a": 1}, "errors": [], "reason_code": "x"}]
        out_path = write_quarantine(quarantined, "v1", tmp_path)
        assert out_path is not None
        assert out_path.name == "v1.quarantine.jsonl"
        lines = out_path.read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["reason_code"] == "x"


class TestComputeFieldNullRates:
    def test_all_present(self):
        records = [{"revenue": "1"}, {"revenue": "2"}]
        assert compute_field_null_rates(records, ["revenue"]) == {"revenue": 0.0}

    def test_half_missing(self):
        records = [{"revenue": "1"}, {"revenue": None}]
        assert compute_field_null_rates(records, ["revenue"]) == {"revenue": 0.5}

    def test_missing_key_counts_as_null(self):
        records = [{"other": 1}]
        assert compute_field_null_rates(records, ["revenue"]) == {"revenue": 1.0}

    def test_empty_records_returns_zero_not_divide_by_zero(self):
        assert compute_field_null_rates([], ["revenue"]) == {"revenue": 0.0}


class TestWriteManifestSummary:
    def test_writes_expected_fields(self, tmp_path):
        out_path = write_manifest_summary(
            "v1", valid=[{"a": 1}], quarantined=[{"a": 2}],
            field_null_rates={"a": 0.0}, manifest_dir=tmp_path,
        )
        summary = json.loads(out_path.read_text())
        assert summary["valid_count"] == 1
        assert summary["quarantined_count"] == 1
        assert summary["field_null_rates"] == {"a": 0.0}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
