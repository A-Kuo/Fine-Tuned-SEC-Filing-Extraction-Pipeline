"""Tests for src/pipeline.py: build_filing_record() and the LLM-track adapter
that maps architecture A's flat ExtractionResult onto normalized MetricRecords.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.inference import ExtractionRequest, ExtractionResponse
from src.extraction.pipeline import build_filing_record, extract_llm_metrics, extraction_result_to_metrics
from src.extraction.postprocessing import ExtractionResult

SAMPLE_FILING = """
Item 1A. Risk Factors

Our business faces significant competitive pressure from larger rivals
with greater financial resources, and any failure to keep pace with
technological change could materially harm our market position and results.

Item 7. Management's Discussion and Analysis

Total revenue for the period grew as demand increased across all segments.

Item 8. Financial Statements

Consolidated Statements of Operations follow below.
"""


class StubExtractionEngine:
    """Duck-typed stand-in for ExtractionEngine -- no model load required."""

    def __init__(self, result: ExtractionResult, model_version: str = "stub-v1",
                 confidence: float = 0.8):
        self._result = result
        self._model_version = model_version
        self._confidence = confidence
        self.calls: list[ExtractionRequest] = []

    def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        self.calls.append(request)
        return ExtractionResponse(
            result=self._result,
            raw_output="{}",
            latency_ms=1.0,
            model_version=self._model_version,
            status="success",
            confidence_score=self._confidence,
        )


class TestBuildFilingRecordHeuristicOnly:
    def test_revenue_heuristic_labeled_correctly(self):
        """Regression test: the naive 'revenue' keyword match must be labeled
        method='heuristic', not the normalize_metric default of 'llm'."""
        record = build_filing_record(
            SAMPLE_FILING, filing_id="f-1", filing_type="10-K"
        )
        revenue_metrics = [m for m in record.metrics if m.name == "revenue"]
        assert len(revenue_metrics) == 1
        assert revenue_metrics[0].method == "heuristic"

    def test_sections_detected(self):
        record = build_filing_record(SAMPLE_FILING, filing_id="f-1", filing_type="10-K")
        types = {s.section_type for s in record.sections}
        assert "risk_factors" in types
        assert "mdna" in types

    def test_risk_factors_extracted(self):
        record = build_filing_record(SAMPLE_FILING, filing_id="f-1", filing_type="10-K")
        assert len(record.risk_factors) >= 1

    def test_mdna_populated(self):
        record = build_filing_record(SAMPLE_FILING, filing_id="f-1", filing_type="10-K")
        assert record.mdna is not None
        assert record.mdna.method == "heuristic"

    def test_no_engine_means_no_llm_calls(self):
        # engine=None (default) -- nothing to assert beyond "it doesn't crash"
        # and no LLM-derived metrics with model_version appear.
        record = build_filing_record(SAMPLE_FILING, filing_id="f-1", filing_type="10-K")
        assert all(m.model_version is None for m in record.metrics)


class TestExtractionResultToMetrics:
    def test_maps_flat_fields_to_metric_records(self):
        result = ExtractionResult(
            revenue="$1.2 million",
            net_income="$300,000",
            total_assets=None,  # should be skipped
        )
        metrics = extraction_result_to_metrics(
            result, model_version="v1", confidence=0.9, source_section="mdna"
        )
        names = {m.name for m in metrics}
        assert names == {"revenue", "net_income"}
        for m in metrics:
            assert m.method == "llm"
            assert m.model_version == "v1"
            assert m.confidence == 0.9
            assert m.source_section == "mdna"

    def test_no_fields_returns_empty(self):
        result = ExtractionResult()
        assert extraction_result_to_metrics(result, model_version="v1", confidence=0.9) == []


class TestExtractLlmMetrics:
    def test_wraps_engine_and_adapts_response(self):
        result = ExtractionResult(revenue="$5 million")
        engine = StubExtractionEngine(result, model_version="llama-sec-v1", confidence=0.75)

        metrics = extract_llm_metrics(
            "some section text", engine=engine, filing_id="f-1", source_section="mdna"
        )

        assert len(engine.calls) == 1
        assert engine.calls[0].text == "some section text"
        assert engine.calls[0].filing_id == "f-1"

        assert len(metrics) == 1
        assert metrics[0].name == "revenue"
        assert metrics[0].method == "llm"
        assert metrics[0].model_version == "llama-sec-v1"
        assert metrics[0].confidence == 0.75

    def test_none_result_returns_empty(self):
        engine = StubExtractionEngine(None)
        assert extract_llm_metrics("text", engine=engine) == []


class TestBuildFilingRecordWithEngine:
    def test_llm_metrics_merged_in(self):
        result = ExtractionResult(revenue="$999 million", net_income="$100 million")
        engine = StubExtractionEngine(result, model_version="llama-sec-v1", confidence=0.85)

        record = build_filing_record(
            SAMPLE_FILING, filing_id="f-1", filing_type="10-K", engine=engine
        )

        net_income_metrics = [m for m in record.metrics if m.name == "net_income"]
        assert len(net_income_metrics) == 1
        assert net_income_metrics[0].method == "llm"
        assert net_income_metrics[0].model_version == "llama-sec-v1"

    def test_llm_overwrites_heuristic_for_same_key_last_write_wins(self):
        """Both the heuristic 'revenue' guess and the LLM 'revenue' guess share
        the natural key (name, period=None, segment=None); per the confirmed
        precedence rule, llm (called after heuristics) wins."""
        result = ExtractionResult(revenue="$999 million")
        engine = StubExtractionEngine(result, model_version="llama-sec-v1", confidence=0.85)

        record = build_filing_record(
            SAMPLE_FILING, filing_id="f-1", filing_type="10-K", engine=engine
        )

        revenue_metrics = [m for m in record.metrics if m.name == "revenue"]
        assert len(revenue_metrics) == 1
        assert revenue_metrics[0].method == "llm"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
