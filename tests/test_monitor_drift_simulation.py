"""Controlled drift simulations: inject a KNOWN synthetic drop/rise and
confirm the drift detector actually catches it, plus a negative control
(noise only, no real drift) confirming it does NOT false-positive.

monitoring/monitor.py's z-test math (proportion_z_test) is real and was
already unit-tested for its formula -- what was missing was validation
against realistic before/after scenarios, and an explicit lock on the sign
convention (current < baseline -> negative z -> small p-value -> flagged),
which a past commit (git log: "z score p value fix") apparently had to
correct once already.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.monitor import (
    check_accuracy_drift,
    check_cache_performance_drift,
    check_field_accuracy_drift,
    check_metric_drift,
    check_parser_fallback_drift,
    check_schema_conformance_drift,
    proportion_z_test,
)


class TestSignConvention:
    """Explicit guard against the exact class of bug a future refactor could
    silently reintroduce: current worse than baseline must produce a
    NEGATIVE z-score and a SMALL p-value (significant)."""

    def test_current_worse_than_baseline_gives_negative_z(self):
        z, p = proportion_z_test(p_current=0.80, p_baseline=0.94, n_current=100, n_baseline=500)
        assert z < 0
        assert p < 0.05

    def test_current_better_than_baseline_gives_positive_z(self):
        z, p = proportion_z_test(p_current=0.98, p_baseline=0.94, n_current=100, n_baseline=500)
        assert z > 0

    def test_current_equals_baseline_gives_zero_z(self):
        z, p = proportion_z_test(p_current=0.94, p_baseline=0.94, n_current=100, n_baseline=500)
        assert z == pytest.approx(0.0)
        assert p == pytest.approx(0.5)


class TestInjectedDriftDetection:
    """A known, deliberately large drop must be caught."""

    def test_large_injected_accuracy_drop_is_flagged(self):
        report = check_accuracy_drift(
            current_accuracy=0.80, baseline_accuracy=0.94, threshold=0.90,
            n_current=50, n_baseline=500,
        )
        assert report.is_drifted is True

    def test_small_noise_within_threshold_not_flagged(self):
        """Negative control: baseline and current both hover near 0.94 with
        no real drift -- must NOT false-positive."""
        report = check_accuracy_drift(
            current_accuracy=0.935, baseline_accuracy=0.94, threshold=0.90,
            n_current=50, n_baseline=500,
        )
        assert report.is_drifted is False

    def test_drop_below_threshold_but_not_significant_is_not_flagged(self):
        """Both conditions (below threshold AND significant) must hold --
        a small sample size can cross the threshold by chance without being
        statistically significant."""
        report = check_accuracy_drift(
            current_accuracy=0.85, baseline_accuracy=0.90, threshold=0.90,
            n_current=3, n_baseline=500,  # tiny n_current -> high variance, likely not significant
        )
        # With n_current=3, this specific drop should not clear p < 0.05.
        assert report.is_drifted is False


class TestGenericMetricDriftReusableAcrossMetrics:
    def test_schema_conformance_drift_uses_same_math_as_accuracy(self):
        acc = check_accuracy_drift(0.80, 0.94, 0.90, n_current=50, n_baseline=500)
        conf = check_schema_conformance_drift(0.80, 0.94, 0.90, n_current=50, n_baseline=500)
        assert acc.is_drifted == conf.is_drifted
        assert acc.z_score == pytest.approx(conf.z_score)
        assert conf.metric_name == "schema_conformance_rate"

    def test_field_accuracy_drift_labels_the_field(self):
        report = check_field_accuracy_drift("revenue", 0.70, 0.90, 0.85, n_current=50, n_baseline=500)
        assert report.metric_name == "field_accuracy:revenue"
        assert report.is_drifted is True

    def test_cache_performance_drift(self):
        report = check_cache_performance_drift(0.20, 0.60, 0.50, n_current=100, n_baseline=1000)
        assert report.metric_name == "cache_hit_rate"
        assert report.is_drifted is True


class TestParserFallbackDriftDirection:
    """Fallback rate RISING is what's bad here -- the opposite direction
    from every other check in this module, which all flag on a metric
    dropping. Verified directly rather than trusted by construction, since
    getting this backwards would silently suppress every real alert."""

    def test_fallback_rate_rising_past_threshold_is_flagged(self):
        report = check_parser_fallback_drift(
            current_fallback_rate=0.30, baseline_fallback_rate=0.05, threshold=0.15,
            n_current=100, n_baseline=500,
        )
        assert report.is_drifted is True
        assert report.metric_name == "parser_fallback_rate"
        assert report.current_value == 0.30
        assert report.baseline_value == 0.05

    def test_fallback_rate_flat_is_not_flagged(self):
        report = check_parser_fallback_drift(
            current_fallback_rate=0.05, baseline_fallback_rate=0.05, threshold=0.15,
            n_current=100, n_baseline=500,
        )
        assert report.is_drifted is False

    def test_fallback_rate_falling_is_not_flagged(self):
        """A LOWER fallback rate is an improvement, not drift."""
        report = check_parser_fallback_drift(
            current_fallback_rate=0.02, baseline_fallback_rate=0.10, threshold=0.15,
            n_current=100, n_baseline=500,
        )
        assert report.is_drifted is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
