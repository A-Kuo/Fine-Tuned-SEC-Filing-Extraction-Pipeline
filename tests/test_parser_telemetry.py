"""Tests for src/extraction/parser_telemetry.py and its threading through
parse_extraction(). The core guarantee: passing telemetry=None (the default)
must be byte-for-byte the same behavior as before telemetry existed."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.parser_telemetry import STAGES, ParseAttempt, ParseTelemetry
from src.extraction.postprocessing import parse_extraction


class TestParseTelemetry:
    def test_record_appends_attempt(self):
        t = ParseTelemetry()
        t.record("direct", True)
        assert len(t.attempts) == 1
        assert t.attempts[0] == ParseAttempt(stage="direct", succeeded=True)

    def test_winning_stage_is_first_success(self):
        t = ParseTelemetry()
        t.record("direct", False, reason_code="not_valid_json")
        t.record("fence_strip", True)
        t.record("regex_extract", True)  # should not overwrite winning_stage
        assert t.winning_stage == "fence_strip"

    def test_winning_stage_none_when_all_fail(self):
        t = ParseTelemetry()
        t.record("direct", False)
        t.record("field_fallback", False)
        assert t.winning_stage is None

    def test_to_dict_shape(self):
        t = ParseTelemetry()
        t.raw_output_chars = 42
        t.record("direct", True, fields_recovered=3)
        d = t.to_dict()
        assert d["winning_stage"] == "direct"
        assert d["raw_output_chars"] == 42
        assert d["attempts"][0]["fields_recovered"] == 3

    def test_stages_constant_matches_the_five_documented_stages(self):
        assert STAGES == ("direct", "fence_strip", "regex_extract", "truncation_repair", "field_fallback")


class TestParseExtractionTelemetryIsAdditive:
    """telemetry=None (the default) must behave identically to calling
    parse_extraction() with no telemetry parameter at all -- the whole point
    of making it optional."""

    def test_no_telemetry_arg_still_works(self):
        result = parse_extraction('{"company_name": "Acme"}')
        assert result.company_name == "Acme"

    def test_telemetry_none_explicit_same_as_omitted(self):
        result = parse_extraction('{"company_name": "Acme"}', telemetry=None)
        assert result.company_name == "Acme"

    def test_telemetry_records_winning_stage_for_direct_parse(self):
        t = ParseTelemetry()
        parse_extraction('{"company_name": "Acme"}', telemetry=t)
        assert t.winning_stage == "direct"

    def test_telemetry_records_winning_stage_for_fence_strip(self):
        t = ParseTelemetry()
        parse_extraction('```json\n{"company_name": "Acme"}\n```', telemetry=t)
        assert t.winning_stage == "fence_strip"

    def test_telemetry_records_all_prior_failed_attempts(self):
        t = ParseTelemetry()
        parse_extraction('```json\n{"company_name": "Acme"}\n```', telemetry=t)
        stages_attempted = [a.stage for a in t.attempts]
        assert stages_attempted == ["direct", "fence_strip"]
        assert t.attempts[0].succeeded is False

    def test_telemetry_records_field_fallback_win(self):
        t = ParseTelemetry()
        parse_extraction("Registrant: Acme Corp\nRevenue: $5 million", telemetry=t)
        assert t.winning_stage == "field_fallback"
        winning_attempt = next(a for a in t.attempts if a.stage == "field_fallback")
        assert winning_attempt.fields_recovered and winning_attempt.fields_recovered > 0

    def test_telemetry_records_total_failure(self):
        t = ParseTelemetry()
        with pytest.raises(Exception):
            parse_extraction("completely unparseable nonsense", telemetry=t)
        assert t.winning_stage is None
        assert len(t.attempts) == len(STAGES)

    def test_raw_output_chars_recorded(self):
        t = ParseTelemetry()
        parse_extraction('{"company_name": "Acme"}', telemetry=t)
        assert t.raw_output_chars == len('{"company_name": "Acme"}')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
