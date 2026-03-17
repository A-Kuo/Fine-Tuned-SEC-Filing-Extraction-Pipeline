"""Tests for postprocessing module.

Tests JSON parsing robustness, schema validation, and edge cases.
These run without GPU (pure Python logic) and should pass before any
model-dependent testing.
"""

import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.postprocessing import (
    ExtractionResult,
    parse_extraction,
    validate_extraction,
    _strip_code_fences,
    _fix_truncated_json,
    _dict_to_result,
)


# ─── ExtractionResult Tests ─────────────────────────────────────────────────

class TestExtractionResult:
    def test_to_dict_excludes_none(self):
        result = ExtractionResult(company_name="Apple Inc.", filing_type="10-K")
        d = result.to_dict()
        assert "company_name" in d
        assert "filing_type" in d
        assert "revenue" not in d  # None fields excluded

    def test_completeness(self):
        empty = ExtractionResult()
        assert empty.completeness == 0.0

        partial = ExtractionResult(company_name="Apple", ticker="AAPL", filing_type="10-K")
        assert 0.0 < partial.completeness < 1.0

    def test_filled_fields(self):
        result = ExtractionResult(company_name="Apple", revenue="$100B")
        assert result.filled_fields == 2

    def test_to_json(self):
        result = ExtractionResult(company_name="Apple Inc.")
        parsed = json.loads(result.to_json())
        assert parsed["company_name"] == "Apple Inc."


# ─── JSON Parsing Tests ─────────────────────────────────────────────────────

class TestParseExtraction:
    def test_clean_json(self):
        """Model returns perfect JSON."""
        raw = json.dumps({
            "company_name": "Apple Inc.",
            "filing_type": "10-K",
            "date": "2023-11-03",
            "revenue": "$383.3 billion",
        })
        result = parse_extraction(raw)
        assert result.company_name == "Apple Inc."
        assert result.filing_type == "10-K"
        assert result.revenue == "$383.3 billion"

    def test_json_with_code_fences(self):
        """Model wraps output in markdown code fences."""
        raw = '```json\n{"company_name": "Microsoft", "filing_type": "10-Q"}\n```'
        result = parse_extraction(raw)
        assert result.company_name == "Microsoft"
        assert result.filing_type == "10-Q"

    def test_json_with_trailing_text(self):
        """Model adds explanation after JSON."""
        raw = '{"company_name": "Tesla", "ticker": "TSLA"}\n\nThe above extraction...'
        result = parse_extraction(raw)
        assert result.company_name == "Tesla"
        assert result.ticker == "TSLA"

    def test_json_with_leading_text(self):
        """Model adds preamble before JSON."""
        raw = 'Based on the filing, here is the extraction:\n{"company_name": "NVIDIA", "sector": "Technology"}'
        result = parse_extraction(raw)
        assert result.company_name == "NVIDIA"

    def test_truncated_json(self):
        """Model hit max_tokens and JSON is incomplete."""
        raw = '{"company_name": "Apple", "ticker": "AAPL", "revenue": "$383B"'
        result = parse_extraction(raw)
        assert result.company_name == "Apple"
        assert result.ticker == "AAPL"

    def test_all_fields_parsed(self):
        """Every field type is correctly parsed."""
        raw = json.dumps({
            "filing_id": "000123-23-456",
            "company_name": "Test Corp",
            "ticker": "TEST",
            "filing_type": "10-K",
            "date": "2024-01-15",
            "fiscal_year_end": "2023-12-31",
            "revenue": "$50.2 billion",
            "net_income": "$12.1 billion",
            "total_assets": "$100 billion",
            "total_liabilities": "$45 billion",
            "eps": "$5.23",
            "sector": "Technology",
        })
        result = parse_extraction(raw)
        assert result.filing_id == "000123-23-456"
        assert result.eps == "$5.23"
        assert result.sector == "Technology"
        assert result.completeness == 1.0

    def test_camelcase_keys(self):
        """Model uses camelCase instead of snake_case."""
        raw = json.dumps({
            "companyName": "Google",
            "filingType": "10-K",
            "totalRevenue": "$300B",
        })
        result = parse_extraction(raw)
        assert result.company_name == "Google"
        assert result.filing_type == "10-K"

    def test_empty_json_raises(self):
        """Empty/garbage input raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            parse_extraction("This is not JSON at all")

    def test_empty_object(self):
        """Model returns {}."""
        result = parse_extraction("{}")
        assert result.filled_fields == 0


# ─── Validation Tests ────────────────────────────────────────────────────────

class TestValidation:
    def test_valid_complete_extraction(self):
        result = ExtractionResult(
            filing_id="000123-23-456",
            company_name="Apple Inc.",
            filing_type="10-K",
            date="2023-11-03",
        )
        is_valid, errors = validate_extraction(result)
        assert is_valid
        assert len(errors) == 0

    def test_missing_required_field(self):
        result = ExtractionResult(
            company_name="Apple Inc.",
            # Missing: filing_id, filing_type, date
        )
        is_valid, errors = validate_extraction(result)
        assert not is_valid
        assert any("filing_id" in e for e in errors)

    def test_invalid_filing_type(self):
        result = ExtractionResult(
            filing_id="000123-23-456",
            company_name="Apple",
            filing_type="INVALID",
            date="2023-01-01",
        )
        is_valid, errors = validate_extraction(result)
        assert not is_valid
        assert any("filing type" in e.lower() for e in errors)

    def test_invalid_date_format(self):
        result = ExtractionResult(
            filing_id="000123-23-456",
            company_name="Apple",
            filing_type="10-K",
            date="not-a-date",
        )
        is_valid, errors = validate_extraction(result)
        assert not is_valid
        assert any("date" in e.lower() for e in errors)

    def test_valid_date_formats(self):
        """Multiple date formats should be accepted."""
        for date_str in ["2023-11-03", "11/03/2023", "November 03, 2023", "2023"]:
            result = ExtractionResult(
                filing_id="test",
                company_name="Test",
                filing_type="10-K",
                date=date_str,
            )
            is_valid, errors = validate_extraction(result)
            date_errors = [e for e in errors if "date" in e.lower() and "format" in e.lower()]
            assert len(date_errors) == 0, f"Date '{date_str}' should be valid"

    def test_financial_field_no_digits(self):
        result = ExtractionResult(
            filing_id="test",
            company_name="Test",
            filing_type="10-K",
            date="2023-01-01",
            revenue="not a number",
        )
        is_valid, errors = validate_extraction(result)
        assert any("numeric" in e.lower() for e in errors)


# ─── Helper Function Tests ───────────────────────────────────────────────────

class TestHelpers:
    def test_strip_code_fences_json(self):
        assert _strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_strip_code_fences_plain(self):
        assert _strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_strip_code_fences_no_fences(self):
        assert _strip_code_fences('{"a": 1}') == '{"a": 1}'

    def test_fix_truncated_single_brace(self):
        result = _fix_truncated_json('{"a": 1, "b": 2')
        assert result is not None
        parsed = json.loads(result)
        assert parsed["a"] == 1

    def test_fix_truncated_returns_none_for_no_json(self):
        assert _fix_truncated_json("just text") is None

    def test_dict_to_result_standard_keys(self):
        result = _dict_to_result({"company_name": "Apple", "ticker": "AAPL"})
        assert result.company_name == "Apple"
        assert result.ticker == "AAPL"

    def test_dict_to_result_alternate_keys(self):
        result = _dict_to_result({"registrant": "Apple", "symbol": "AAPL", "form_type": "10-K"})
        assert result.company_name == "Apple"
        assert result.ticker == "AAPL"
        assert result.filing_type == "10-K"


# ─── Integration-style test with sample data ─────────────────────────────────

class TestWithSampleData:
    def test_sample_expected_output(self):
        """Parse the sample_10k.expected.json and validate it."""
        expected_path = Path(__file__).parent.parent / "data" / "sample_10k.expected.json"
        if not expected_path.exists():
            pytest.skip("Sample data not generated yet")

        with open(expected_path) as f:
            expected = json.load(f)

        result = parse_extraction(json.dumps(expected))
        assert result.company_name is not None
        assert result.filing_type is not None

        is_valid, errors = validate_extraction(result)
        assert is_valid, f"Sample extraction should be valid: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
