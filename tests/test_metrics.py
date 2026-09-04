"""Tests for evaluation/metrics.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import (
    error_taxonomy_breakdown,
    exact_json_match_rate,
    null_handling_correctness,
    per_field_accuracy,
    report_confidence_calibration,
)


class TestPerFieldAccuracy:
    def test_all_correct(self):
        preds = [{"company_name": "Apple", "revenue": "$100 million"}]
        truths = [{"company_name": "Apple", "revenue": "$100 million"}]
        result = per_field_accuracy(preds, truths)
        assert result["company_name"]["accuracy"] == 1.0
        assert result["revenue"]["accuracy"] == 1.0

    def test_wrong_text_field(self):
        preds = [{"company_name": "Apple Corp"}]
        truths = [{"company_name": "Apple Inc"}]
        result = per_field_accuracy(preds, truths)
        assert result["company_name"]["accuracy"] == 0.0

    def test_fuzzy_numeric_within_tolerance_counts_correct(self):
        preds = [{"revenue": "$100 million"}]
        truths = [{"revenue": "$103 million"}]
        result = per_field_accuracy(preds, truths)
        assert result["revenue"]["accuracy"] == 1.0

    def test_both_none_not_counted_in_total(self):
        preds = [{"eps": None}]
        truths = [{"eps": None}]
        result = per_field_accuracy(preds, truths)
        assert result["eps"]["total"] == 0
        assert result["eps"]["accuracy"] is None

    def test_aggregates_across_multiple_examples(self):
        preds = [{"company_name": "Apple"}, {"company_name": "Wrong"}]
        truths = [{"company_name": "Apple"}, {"company_name": "Google"}]
        result = per_field_accuracy(preds, truths)
        assert result["company_name"]["correct"] == 1
        assert result["company_name"]["total"] == 2
        assert result["company_name"]["accuracy"] == 0.5


class TestExactJsonMatchRate:
    def test_all_fields_match(self):
        preds = [{"company_name": "Apple", "revenue": "$100 million"}]
        truths = [{"company_name": "Apple", "revenue": "$100 million"}]
        assert exact_json_match_rate(preds, truths) == 1.0

    def test_one_field_wrong_fails_whole_record(self):
        preds = [{"company_name": "Apple", "revenue": "$50 million"}]
        truths = [{"company_name": "Apple", "revenue": "$100 million"}]
        assert exact_json_match_rate(preds, truths) == 0.0

    def test_empty_predictions_returns_zero_not_crash(self):
        assert exact_json_match_rate([], []) == 0.0

    def test_partial_match_across_records(self):
        preds = [{"company_name": "Apple"}, {"company_name": "Wrong"}]
        truths = [{"company_name": "Apple"}, {"company_name": "Google"}]
        assert exact_json_match_rate(preds, truths) == 0.5


class TestNullHandlingCorrectness:
    """null_handling_correctness() checks every field in ALL_FIELDS per
    record (missing keys read as None via dict.get()), so these tests build
    a single-field dict deliberately and assert against total_field_checks
    to isolate the one field under test from the other 11."""

    def test_correct_abstain(self):
        preds = [{"eps": None}]
        truths = [{"eps": None}]
        result = null_handling_correctness(preds, truths)
        assert result["correct_abstain"] == result["total_field_checks"]

    def test_incorrect_abstain_when_truth_present(self):
        preds = [{"eps": None}]
        truths = [{"eps": "$1.23"}]
        result = null_handling_correctness(preds, truths)
        assert result["incorrect_abstain_rate"] == pytest.approx(1 / result["total_field_checks"])

    def test_hallucination_when_truth_absent(self):
        preds = [{"eps": "$1.23"}]
        truths = [{"eps": None}]
        result = null_handling_correctness(preds, truths)
        assert result["hallucination_rate"] == pytest.approx(1 / result["total_field_checks"])


class TestErrorTaxonomyBreakdown:
    def test_aggregates_tags_across_records(self):
        preds = [{"company_name": None}, {"company_name": "Wrong Corp"}]
        truths = [{"company_name": "Apple"}, {"company_name": "Google"}]
        breakdown = error_taxonomy_breakdown(preds, truths)
        assert breakdown["missing_required_field"] == 1
        assert breakdown["wrong_field"] == 1

    def test_parser_recovery_failure_tagged_when_stage_provided(self):
        preds = [{"company_name": "Wrong"}]
        truths = [{"company_name": "Apple"}]
        breakdown = error_taxonomy_breakdown(preds, truths, parse_stages=["field_fallback"])
        assert breakdown.get("parser_recovery_failure") == 1


class TestConfidenceCalibration:
    def test_reports_not_applicable(self):
        result = report_confidence_calibration([])
        assert result["applicable"] is False
        assert "heuristic constant" in result["reason"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
