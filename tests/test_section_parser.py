"""Tests for src/section_parser.py's extract_sections()."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.section_parser import extract_sections


SAMPLE_FILING = """
SOME PREAMBLE TEXT ABOUT THE COMPANY.

Item 1A. Risk Factors

Our business is subject to numerous risks, including competition,
regulatory changes, and supply chain disruption. These risks could
materially affect our results of operations.

Item 7. Management's Discussion and Analysis

Revenue increased 12% year over year driven by strong demand for
our products. Operating margin expanded due to cost discipline.

Item 8. Financial Statements

Consolidated Statements of Operations show total revenue of $1.2 billion.

SIGNATURES

Pursuant to the requirements of the Securities Exchange Act...
"""


class TestExtractSections:
    def test_detects_all_three_sections(self):
        spans = extract_sections(SAMPLE_FILING)
        types = {s.section_type for s in spans}
        assert types == {"risk_factors", "mdna", "financial_statements"}

    def test_spans_sorted_by_start(self):
        spans = extract_sections(SAMPLE_FILING)
        starts = [s.start for s in spans]
        assert starts == sorted(starts)

    def test_item_number_match_has_high_confidence(self):
        spans = extract_sections(SAMPLE_FILING)
        risk = next(s for s in spans if s.section_type == "risk_factors")
        assert risk.confidence == 0.9

    def test_generic_heading_has_lower_confidence(self):
        text = "Risk Factors\n\nSomething risky happens here.\n\nItem 99 stop marker\n"
        spans = extract_sections(text)
        risk = next(s for s in spans if s.section_type == "risk_factors")
        assert risk.confidence == 0.7

    def test_section_stops_at_next_item_heading(self):
        spans = extract_sections(SAMPLE_FILING)
        risk = next(s for s in spans if s.section_type == "risk_factors")
        assert "Management's Discussion" not in risk.text

    def test_section_stops_at_signatures(self):
        spans = extract_sections(SAMPLE_FILING)
        fin = next(s for s in spans if s.section_type == "financial_statements")
        assert "SIGNATURES" not in fin.text

    def test_no_match_returns_empty_list(self):
        spans = extract_sections("Nothing relevant in this text at all.")
        assert spans == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
