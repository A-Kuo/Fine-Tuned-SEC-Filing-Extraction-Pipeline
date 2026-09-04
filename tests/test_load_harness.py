"""Tests for evaluation/load_harness.py's pure percentiles() function.

The rest of this module needs a live server (Docker) to exercise for real --
run_concurrent_load()/measure_cold_start()/measure_degraded_path() are not
unit-tested here for that reason; see docs/NEXT_EXPERIMENTS.md for the
runbook to execute them against a real server.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.load_harness import percentiles


class TestPercentiles:
    def test_empty_list_returns_none_values(self):
        result = percentiles([])
        assert result["p50_ms"] is None

    def test_single_value(self):
        result = percentiles([100.0])
        assert result["p50_ms"] == 100.0
        assert result["p95_ms"] == 100.0
        assert result["min_ms"] == 100.0
        assert result["max_ms"] == 100.0

    def test_computes_correct_median(self):
        result = percentiles([100.0, 200.0, 300.0])
        assert result["p50_ms"] == 200.0

    def test_min_and_max(self):
        result = percentiles([50.0, 10.0, 90.0, 30.0])
        assert result["min_ms"] == 10.0
        assert result["max_ms"] == 90.0

    def test_p99_does_not_index_out_of_range(self):
        """A small n could otherwise compute int(n*0.99) == n, an IndexError."""
        result = percentiles([1.0, 2.0, 3.0])
        assert result["p99_ms"] == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
