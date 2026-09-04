"""Tests for evaluation/benchmark_splits.py's adversarial text transforms
(pure functions, no fixtures needed) and the split-selection logic."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.benchmark_splits import (
    _corrupt_currency_symbols,
    _corrupt_extra_whitespace,
    _corrupt_html_entity_leftovers,
    CURATED_AUTHENTIC_TICKERS,
    OUT_OF_DOMAIN_TICKERS,
)


class TestCorruptExtraWhitespace:
    def test_produces_different_text(self):
        text = "line one\nline two\nline three\nline four\nline five"
        assert _corrupt_extra_whitespace(text) != text

    def test_preserves_all_words(self):
        text = "Revenue increased significantly this year"
        corrupted = _corrupt_extra_whitespace(text)
        for word in text.split():
            assert word in corrupted


class TestCorruptCurrencySymbols:
    def test_dollar_sign_becomes_usd(self):
        assert _corrupt_currency_symbols("$100 million") == "USD 100 million"

    def test_negative_dollar_uses_unicode_minus(self):
        result = _corrupt_currency_symbols("-$5 million")
        assert "−" in result

    def test_text_without_dollar_sign_unchanged(self):
        text = "No currency here"
        assert _corrupt_currency_symbols(text) == text


class TestCorruptHtmlEntityLeftovers:
    def test_introduces_some_entities(self):
        text = "R&D and 'quotes' and \"double quotes\""
        corrupted = _corrupt_html_entity_leftovers(text)
        assert corrupted != text

    def test_does_not_crash_on_text_without_special_chars(self):
        text = "Plain text with no special characters"
        assert _corrupt_html_entity_leftovers(text) == text


class TestSectorClassification:
    def test_no_overlap_between_curated_and_ood_tickers(self):
        assert CURATED_AUTHENTIC_TICKERS.isdisjoint(OUT_OF_DOMAIN_TICKERS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
