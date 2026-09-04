"""Tests for src/extraction/numeric_normalize.py's fuzzy_numeric_match().

parse_numeric_value() itself is already covered by tests/test_normalizer.py
(which imports it via the normalizer.py re-export) -- this file covers the
new fuzzy_numeric_match() function and confirms both call sites (the
extraction pipeline and the storage layer) now share one implementation.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.numeric_normalize import fuzzy_numeric_match, parse_numeric_value


class TestFuzzyNumericMatch:
    def test_exact_match(self):
        assert fuzzy_numeric_match("$100 million", "$100 million") is True

    def test_within_tolerance(self):
        assert fuzzy_numeric_match("$100 million", "$103 million", tolerance=0.05) is True

    def test_outside_tolerance(self):
        assert fuzzy_numeric_match("$100 million", "$120 million", tolerance=0.05) is False

    def test_different_units_same_magnitude(self):
        assert fuzzy_numeric_match("1000 million", "$1 billion") is True

    def test_both_none_is_a_match(self):
        """Both correctly abstaining is not an error."""
        assert fuzzy_numeric_match(None, None) is True

    def test_one_none_is_not_a_match(self):
        assert fuzzy_numeric_match(None, "$1 million") is False
        assert fuzzy_numeric_match("$1 million", None) is False

    def test_truth_zero_requires_exact_zero(self):
        assert fuzzy_numeric_match("$0", "$0") is True
        assert fuzzy_numeric_match("$5", "$0") is False

    def test_unparseable_values_do_not_match(self):
        assert fuzzy_numeric_match("not a number", "$5 million") is False


class TestConsolidatedImplementationSharedAcrossCallSites:
    def test_storage_layer_delegates_to_shared_function(self):
        from src.storage.database import PostgresStorage
        assert PostgresStorage._parse_financial("$12.1 million") == parse_numeric_value("$12.1 million")

    def test_normalizer_reexports_shared_function(self):
        from src.extraction.normalizer import parse_numeric_value as normalizer_parse
        assert normalizer_parse is parse_numeric_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
