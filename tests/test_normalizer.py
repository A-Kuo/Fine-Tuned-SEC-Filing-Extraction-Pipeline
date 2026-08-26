"""Tests for src/normalizer.py: numeric parsing, metric normalization,
XBRL precedence, and FilingRecord -> DB row shaping.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.normalizer import (
    filing_record_to_rows,
    metric_to_row,
    mdna_to_row,
    normalize_metric,
    parse_numeric_value,
    resolve_metric_precedence,
    risk_factor_to_row,
    section_to_row,
)
from src.schemas import (
    FilingMetadata,
    FilingRecord,
    MdnaSummaryRecord,
    MetricRecord,
    RiskFactorRecord,
    SectionRecord,
)


class TestParseNumericValue:
    def test_none_input(self):
        assert parse_numeric_value(None) is None

    def test_passthrough_numeric(self):
        assert parse_numeric_value(42) == 42
        assert parse_numeric_value(3.14) == 3.14

    def test_plain_dollar_amount(self):
        assert parse_numeric_value("$5.23") == 5.23

    def test_million_multiplier(self):
        assert parse_numeric_value("$12.1 million") == 12_100_000

    def test_billion_multiplier(self):
        assert parse_numeric_value("$383.3 billion") == 383_300_000_000

    def test_thousand_multiplier(self):
        assert parse_numeric_value("$450 thousand") == 450_000

    def test_commas_stripped(self):
        assert parse_numeric_value("$1,234,567") == 1234567

    def test_no_digits_returns_none(self):
        assert parse_numeric_value("not a number") is None

    def test_integer_result_not_float(self):
        result = parse_numeric_value("$5 million")
        assert result == 5_000_000
        assert isinstance(result, int)


class TestNormalizeMetric:
    def test_field_mapping(self):
        m = normalize_metric(
            "revenue",
            raw_value="$1.2 million",
            unit="usd",
            period="FY2024",
            segment="Americas",
            method="xbrl",
            confidence=0.99,
            source_section="financial_statements",
            evidence_text="evidence",
            model_version="llama-sec-v1",
        )
        assert m.name == "revenue"
        assert m.value == 1_200_000
        assert m.unit == "usd"
        assert m.period == "FY2024"
        assert m.segment == "Americas"
        assert m.method == "xbrl"
        assert m.confidence == 0.99
        assert m.source_section == "financial_statements"
        assert m.evidence_text == "evidence"
        assert m.model_version == "llama-sec-v1"

    def test_defaults(self):
        m = normalize_metric("revenue", raw_value="$1 million")
        assert m.method == "llm"
        assert m.unit == "usd"
        assert m.model_version is None


class TestResolveMetricPrecedence:
    def _metric(self, method: str) -> MetricRecord:
        return MetricRecord(name="revenue", value=1.0, method=method, confidence=0.5)

    def test_none_existing_returns_incoming(self):
        incoming = self._metric("llm")
        assert resolve_metric_precedence(None, incoming) is incoming

    @pytest.mark.parametrize(
        "existing_method,incoming_method,expect_existing_wins",
        [
            ("xbrl", "xbrl", False),  # last write wins between two xbrl facts
            ("xbrl", "heuristic", True),
            ("xbrl", "llm", True),
            ("heuristic", "xbrl", False),  # xbrl always wins when incoming
            ("heuristic", "heuristic", False),
            ("heuristic", "llm", False),
            ("llm", "xbrl", False),
            ("llm", "heuristic", False),  # last-write-wins between heuristic/llm
            ("llm", "llm", False),
        ],
    )
    def test_all_method_pair_combinations(
        self, existing_method, incoming_method, expect_existing_wins
    ):
        existing = self._metric(existing_method)
        incoming = self._metric(incoming_method)
        result = resolve_metric_precedence(existing, incoming)
        if expect_existing_wins:
            assert result is existing
        else:
            assert result is incoming


class TestRowShaping:
    def test_section_to_row(self):
        s = SectionRecord(
            section_type="mdna", title="MD&A", text="body", start=10, end=20, confidence=0.9
        )
        row = section_to_row("f-1", s)
        assert row == {
            "filing_id": "f-1",
            "section_type": "mdna",
            "title": "MD&A",
            "char_start": 10,
            "char_end": 20,
            "confidence": 0.9,
        }

    def test_metric_to_row_empty_period_segment_default_to_empty_string(self):
        m = MetricRecord(name="revenue", value=1.0, method="llm", confidence=0.5)
        row = metric_to_row("f-1", m)
        assert row["period"] == ""
        assert row["segment"] == ""
        assert row["filing_id"] == "f-1"
        assert row["metric_name"] == "revenue"

    def test_risk_factor_to_row_hash_deterministic(self):
        r = RiskFactorRecord(text="same risk text", confidence=0.5)
        row1 = risk_factor_to_row("f-1", r)
        row2 = risk_factor_to_row("f-1", r)
        assert row1["risk_hash"] == row2["risk_hash"]
        assert len(row1["risk_hash"]) == 64

    def test_mdna_to_row(self):
        m = MdnaSummaryRecord(summary="summary", method="llm", model_version="v1")
        row = mdna_to_row("f-1", m)
        assert row == {
            "filing_id": "f-1",
            "summary": "summary",
            "method": "llm",
            "model_version": "v1",
        }

    def test_filing_record_to_rows_shape(self):
        record = FilingRecord(
            metadata=FilingMetadata(filing_id="f-1", filing_type="10-K"),
            sections=[
                SectionRecord(
                    section_type="mdna", title="t", text="x", start=0, end=1, confidence=0.9
                )
            ],
            metrics=[MetricRecord(name="revenue", value=1.0, method="llm", confidence=0.5)],
            risk_factors=[RiskFactorRecord(text="risk", confidence=0.5)],
            mdna=MdnaSummaryRecord(summary="s"),
        )
        rows = filing_record_to_rows(record)
        assert set(rows.keys()) == {
            "filings", "filing_sections", "financial_metrics", "risk_factors",
            "mdna_summaries",
        }
        assert len(rows["filings"]) == 1
        assert len(rows["filing_sections"]) == 1
        assert len(rows["financial_metrics"]) == 1
        assert len(rows["risk_factors"]) == 1
        assert len(rows["mdna_summaries"]) == 1

    def test_filing_record_to_rows_no_mdna(self):
        record = FilingRecord(
            metadata=FilingMetadata(filing_id="f-1", filing_type="10-K"),
            sections=[], metrics=[], risk_factors=[], mdna=None,
        )
        rows = filing_record_to_rows(record)
        assert rows["mdna_summaries"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
