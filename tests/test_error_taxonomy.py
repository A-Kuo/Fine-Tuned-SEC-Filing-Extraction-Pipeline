"""Tests for evaluation/error_taxonomy.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.error_taxonomy import classify_error


class TestClassifyError:
    def test_correct_match_yields_no_tags(self):
        assert classify_error("company_name", "Apple Inc.", "Apple Inc.") == []

    def test_correct_numeric_match_within_tolerance_yields_no_tags(self):
        assert classify_error("revenue", "$100 million", "$103 million") == []

    def test_both_absent_yields_no_tags(self):
        assert classify_error("eps", None, None) == []

    def test_missing_required_field(self):
        assert classify_error("company_name", None, "Apple Inc.") == ["missing_required_field"]

    def test_hallucinated_field(self):
        assert classify_error("eps", "$1.23", None) == ["hallucinated_field"]

    def test_wrong_text_field(self):
        assert classify_error("company_name", "Apple Corp", "Apple Inc.") == ["wrong_field"]

    def test_unparseable_numeric_is_formatting_failure(self):
        tags = classify_error("revenue", "a lot of money", "$100 million")
        assert "formatting_normalization_failure" in tags

    def test_scaling_error_1000x(self):
        tags = classify_error("revenue", "$100 thousand", "$100 million")
        assert "numeric_scaling_failure" in tags

    def test_scaling_error_inverse_1000x(self):
        tags = classify_error("revenue", "$100 billion", "$100 million")
        assert "numeric_scaling_failure" in tags

    def test_plain_wrong_number_not_scaling(self):
        """A prediction that's just wrong (not off by a clean power of 1000)
        should not be misclassified as a scaling error."""
        tags = classify_error("revenue", "$150 million", "$100 million")
        assert "numeric_scaling_failure" not in tags
        assert "wrong_field" in tags

    def test_parser_recovery_failure_tag_added_for_fallback_stages(self):
        tags = classify_error("company_name", "Apple Corp", "Apple Inc.", parse_stage="field_fallback")
        assert "wrong_field" in tags
        assert "parser_recovery_failure" in tags

    def test_no_parser_recovery_failure_tag_for_direct_stage(self):
        tags = classify_error("company_name", "Apple Corp", "Apple Inc.", parse_stage="direct")
        assert "parser_recovery_failure" not in tags

    def test_no_parser_recovery_failure_tag_when_not_actually_an_error(self):
        tags = classify_error("company_name", "Apple Inc.", "Apple Inc.", parse_stage="field_fallback")
        assert tags == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
