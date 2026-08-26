"""Tests for the normalized Pydantic schemas (src/schemas.py)."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas import (
    FilingMetadata,
    FilingRecord,
    MdnaSummaryRecord,
    MetricRecord,
    RiskFactorRecord,
    SectionRecord,
)


class TestConfidenceBounds:
    def test_section_record_confidence_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            SectionRecord(
                section_type="mdna", title="t", text="x", start=0, end=1, confidence=1.5
            )

    def test_metric_record_confidence_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            MetricRecord(name="revenue", method="llm", confidence=-0.1)

    def test_metric_record_confidence_in_range_accepted(self):
        m = MetricRecord(name="revenue", method="llm", confidence=0.5)
        assert m.confidence == 0.5


class TestMethodLiteral:
    def test_invalid_method_rejected(self):
        with pytest.raises(ValidationError):
            MetricRecord(name="revenue", method="guess", confidence=0.5)

    @pytest.mark.parametrize("method", ["xbrl", "heuristic", "llm"])
    def test_valid_methods_accepted(self, method):
        m = MetricRecord(name="revenue", method=method, confidence=0.5)
        assert m.method == method

    def test_mdna_summary_invalid_method_rejected(self):
        with pytest.raises(ValidationError):
            MdnaSummaryRecord(summary="text", method="xbrl")

    @pytest.mark.parametrize("method", ["heuristic", "llm"])
    def test_mdna_summary_valid_methods_accepted(self, method):
        m = MdnaSummaryRecord(summary="text", method=method)
        assert m.method == method

    def test_mdna_summary_default_method(self):
        m = MdnaSummaryRecord(summary="text")
        assert m.method == "heuristic"


class TestFilingRecordRoundTrip:
    def test_round_trip_serialization(self):
        record = FilingRecord(
            metadata=FilingMetadata(
                filing_id="f-1",
                cik="0000320193",
                accession_no="0000320193-23-000106",
                ticker="AAPL",
                company_name="Apple Inc.",
                filing_type="10-K",
                filing_date="2023-11-03",
                raw_text_hash="abc123",
            ),
            sections=[
                SectionRecord(
                    section_type="mdna", title="MD&A", text="text", start=0, end=4,
                    confidence=0.9,
                )
            ],
            metrics=[
                MetricRecord(name="revenue", value=1000.0, method="llm", confidence=0.8)
            ],
            risk_factors=[RiskFactorRecord(text="some risk", confidence=0.7)],
            mdna=MdnaSummaryRecord(summary="summary text", method="llm", model_version="v1"),
        )

        dumped = record.model_dump()
        restored = FilingRecord.model_validate(dumped)

        assert restored == record
        assert restored.mdna.method == "llm"
        assert restored.metadata.accession_no == "0000320193-23-000106"

    def test_mdna_defaults_to_none(self):
        record = FilingRecord(
            metadata=FilingMetadata(filing_id="f-2", filing_type="10-Q"),
            sections=[],
            metrics=[],
            risk_factors=[],
        )
        assert record.mdna is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
