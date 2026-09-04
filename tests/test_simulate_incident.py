"""Tests for scripts/simulate_incident.py's pure _measure_fallback_rate()
helper."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.simulate_incident import _measure_fallback_rate


class TestMeasureFallbackRate:
    def test_all_clean_json_gives_zero_fallback_rate(self):
        corpus = ['{"a": 1}', '{"b": 2}']
        rate, n = _measure_fallback_rate(corpus)
        assert rate == 0.0
        assert n == 2

    def test_malformed_input_counts_as_fallback(self):
        corpus = ['```json\n{"a": 1}\n```']  # needs fence-strip, not direct parse
        rate, n = _measure_fallback_rate(corpus)
        assert rate == 1.0

    def test_total_failure_also_counts_as_fallback(self):
        corpus = ["completely unparseable garbage with no fields"]
        rate, n = _measure_fallback_rate(corpus)
        assert rate == 1.0

    def test_mixed_corpus(self):
        corpus = ['{"a": 1}', '```json\n{"a": 1}\n```']
        rate, n = _measure_fallback_rate(corpus)
        assert rate == 0.5

    def test_empty_corpus_returns_zero_not_divide_by_zero(self):
        rate, n = _measure_fallback_rate([])
        assert rate == 0.0
        assert n == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
