"""Tests for monitoring and evaluation modules."""

import json
import math
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.monitor import (
    proportion_z_test,
    check_accuracy_drift,
    check_latency_sla,
    generate_full_report,
    DriftReport,
)
from evaluation.evaluate import (
    exact_match,
    fuzzy_financial_match,
    _parse_to_number,
    evaluate_single,
    generate_sample_metrics,
)


# ─── Drift Detection Tests ──────────────────────────────────────────────────

class TestDriftDetection:
    def test_no_drift_same_accuracy(self):
        """Same accuracy as baseline → no drift."""
        drift = check_accuracy_drift(
            current_accuracy=0.94,
            baseline_accuracy=0.94,
            threshold=0.92,
        )
        assert not drift.is_drifted

    def test_drift_below_threshold(self):
        """Accuracy below threshold with statistical significance → drift."""
        drift = check_accuracy_drift(
            current_accuracy=0.88,
            baseline_accuracy=0.94,
            threshold=0.92,
            n_current=100,
            n_baseline=500,
        )
        assert drift.is_drifted
        assert drift.z_score < 0  # Negative = current worse than baseline

    def test_no_drift_above_threshold(self):
        """Accuracy dropped slightly but still above threshold → no drift."""
        drift = check_accuracy_drift(
            current_accuracy=0.93,
            baseline_accuracy=0.94,
            threshold=0.92,
        )
        assert not drift.is_drifted

    def test_drift_report_fields(self):
        drift = check_accuracy_drift(0.90, 0.94, 0.92, n_current=50)
        d = drift.to_dict()
        assert "metric_name" in d
        assert "z_score" in d
        assert "p_value" in d
        assert d["sample_size"] == 50


class TestProportionZTest:
    def test_equal_proportions(self):
        z, p = proportion_z_test(0.94, 0.94, 100, 100)
        assert abs(z) < 0.01
        assert p > 0.4  # Not significant

    def test_significant_drop(self):
        z, p = proportion_z_test(0.80, 0.94, 200, 500)
        assert z < -2.0  # Strong negative
        assert p < 0.05  # Significant

    def test_zero_samples(self):
        z, p = proportion_z_test(0.5, 0.5, 0, 100)
        assert z == 0.0
        assert p == 1.0


# ─── Latency SLA Tests ──────────────────────────────────────────────────────

class TestLatencySLA:
    def test_within_sla(self):
        latencies = [300, 350, 400, 320, 380, 410, 290, 500, 450, 350]
        report = check_latency_sla(latencies, sla_p99_ms=1200)
        assert report.is_within_sla
        assert report.p99_ms < 1200

    def test_exceeds_sla(self):
        latencies = [300, 350, 1500, 1400, 1300, 1600, 400, 1200, 1100, 1700]
        report = check_latency_sla(latencies, sla_p99_ms=1200)
        assert not report.is_within_sla

    def test_empty_latencies(self):
        report = check_latency_sla([], sla_p99_ms=1200)
        assert report.is_within_sla  # No data = no violation
        assert report.sample_size == 0


# ─── Full Report Tests ───────────────────────────────────────────────────────

class TestFullReport:
    def test_healthy_report(self):
        report = generate_full_report(
            current_accuracy=0.94,
            baseline_accuracy=0.94,
            latencies_ms=[300, 350, 400, 320],
        )
        assert report.status == "healthy"
        assert len(report.alerts) == 0

    def test_critical_accuracy_report(self):
        report = generate_full_report(
            current_accuracy=0.85,
            baseline_accuracy=0.94,
            latencies_ms=[300, 350, 400],
        )
        assert report.status == "critical"
        assert any("accuracy" in a.lower() or "Accuracy" in a for a in report.alerts)

    def test_critical_latency_report(self):
        report = generate_full_report(
            current_accuracy=0.94,
            baseline_accuracy=0.94,
            latencies_ms=[1500, 1600, 1700, 1800, 1400, 1300, 1200, 1900, 1100, 2000],
        )
        assert report.status == "critical"
        assert any("latency" in a.lower() for a in report.alerts)

    def test_report_serialization(self):
        report = generate_full_report(0.94, 0.94, [300, 400])
        d = report.to_dict()
        assert "status" in d
        assert "drift" in d
        assert "latency" in d
        # Should be JSON-serializable
        json.dumps(d)


# ─── Evaluation: Field Matching ──────────────────────────────────────────────

class TestFieldMatching:
    def test_exact_match_identical(self):
        assert exact_match("Apple Inc.", "Apple Inc.")

    def test_exact_match_case_insensitive(self):
        assert exact_match("apple inc.", "Apple Inc.")

    def test_exact_match_whitespace(self):
        assert exact_match("  Apple Inc.  ", "Apple Inc.")

    def test_exact_match_none(self):
        assert exact_match(None, None)
        assert not exact_match("Apple", None)
        assert not exact_match(None, "Apple")


class TestFuzzyFinancialMatch:
    def test_identical_strings(self):
        assert fuzzy_financial_match("$383.3 billion", "$383.3 billion")

    def test_different_format_same_value(self):
        assert fuzzy_financial_match("$383.3 billion", "$383.3B")

    def test_numeric_vs_text(self):
        assert fuzzy_financial_match("$12.1 million", "$12100000")

    def test_within_tolerance(self):
        # 5% tolerance
        assert fuzzy_financial_match("$100 billion", "$103 billion")

    def test_outside_tolerance(self):
        assert not fuzzy_financial_match("$100 billion", "$120 billion")

    def test_none_values(self):
        assert fuzzy_financial_match(None, None)
        assert not fuzzy_financial_match("$100", None)


class TestParseToNumber:
    def test_billions(self):
        assert _parse_to_number("$383.3 billion") == pytest.approx(383.3e9)

    def test_millions(self):
        assert _parse_to_number("$12.1 million") == pytest.approx(12.1e6)

    def test_abbreviation_B(self):
        assert _parse_to_number("$383.3B") == pytest.approx(383.3e9)

    def test_plain_number(self):
        assert _parse_to_number("$5.23") == pytest.approx(5.23)

    def test_no_number(self):
        assert _parse_to_number("not a number") is None


class TestEvaluateSingle:
    def test_perfect_match(self):
        pred = {
            "company_name": "Apple Inc.",
            "filing_type": "10-K",
            "date": "2023-11-03",
            "revenue": "$383.3 billion",
        }
        truth = pred.copy()
        results = evaluate_single(pred, truth)
        assert all(r["correct"] for r in results.values() if r["ground_truth"] is not None)

    def test_partial_mismatch(self):
        pred = {"company_name": "Apple Inc.", "filing_type": "10-Q"}
        truth = {"company_name": "Apple Inc.", "filing_type": "10-K"}
        results = evaluate_single(pred, truth)
        assert results["company_name"]["correct"]
        assert not results["filing_type"]["correct"]


class TestSampleMetrics:
    def test_sample_metrics_valid(self):
        metrics = generate_sample_metrics()
        assert metrics["overall_accuracy"] == 0.94
        assert "per_field" in metrics
        assert "company_name" in metrics["per_field"]
        assert metrics["per_field"]["company_name"]["accuracy"] > 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
