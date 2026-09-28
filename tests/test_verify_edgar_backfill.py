"""Tests for scripts/verify_edgar_backfill.py's report building/printing --
a fake cursor stands in for Postgres, no live database needed."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.verify_edgar_backfill import gather_report, print_report


class FakeCursor:
    """Returns canned results in call order; records every query."""

    def __init__(self, results: list):
        self._results = list(results)
        self.queries: list[str] = []

    def execute(self, sql, params=None):
        self.queries.append(sql)
        self._current = self._results.pop(0)

    def fetchone(self):
        return self._current

    def fetchall(self):
        return self._current


RESULTS = [
    (42,),                                              # entity_count
    (137,),                                             # filing_count
    [("10-K", 42), ("10-Q", 84), ("8-K", 11)],          # by_form
    [("accepted", 130), ("amended", 5), ("late_notice", 2)],  # by_status
    [("D", 20), ("H", 10), (None, 2)],                  # by_sic_division
    ("2026-01-02", "2026-09-15"),                       # date_range
    [("0000320193-25-000079", "0000320193", "Apple Inc.", "10-K", "2025-10-31",
      "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl.htm")],  # sample_rows
    [],                                                  # duplicate_accessions
]


class TestGatherReport:
    def test_collects_every_expected_section(self):
        cur = FakeCursor([list(r) if isinstance(r, list) else r for r in RESULTS])
        report = gather_report(cur, sample=10)

        assert report["entity_count"] == 42
        assert report["filing_count"] == 137
        assert report["by_form"] == [("10-K", 42), ("10-Q", 84), ("8-K", 11)]
        assert report["by_status"][0] == ("accepted", 130)
        assert report["by_sic_division"] == [("D", 20), ("H", 10), (None, 2)]
        assert report["date_range"] == ("2026-01-02", "2026-09-15")
        assert len(report["sample_rows"]) == 1
        assert report["duplicate_accessions"] == []

    def test_sample_size_is_passed_through_as_a_query_parameter(self):
        cur = FakeCursor([list(r) if isinstance(r, list) else r for r in RESULTS])
        gather_report(cur, sample=25)
        assert any("LIMIT" in q for q in cur.queries)

    def test_flags_duplicate_accession_numbers_when_present(self):
        results = list(RESULTS)
        results[-1] = [("0000320193-25-000079", 2)]
        cur = FakeCursor([list(r) if isinstance(r, list) else r for r in results])
        report = gather_report(cur, sample=10)
        assert report["duplicate_accessions"] == [("0000320193-25-000079", 2)]


class TestPrintReport:
    def _report(self, **overrides):
        base = {
            "entity_count": 42, "filing_count": 137,
            "by_form": [("10-K", 42)], "by_status": [("accepted", 130)],
            "by_sic_division": [("D", 20)], "date_range": ("2026-01-02", "2026-09-15"),
            "sample_rows": [("0000320193-25-000079", "0000320193", "Apple Inc.", "10-K", "2025-10-31", "https://sec.gov/x")],
            "duplicate_accessions": [],
        }
        base.update(overrides)
        return base

    def test_prints_counts_and_the_sample_url_for_spot_checking(self, capsys):
        print_report(self._report())
        out = capsys.readouterr().out
        assert "Entities: 42" in out
        assert "Filings: 137" in out
        assert "https://sec.gov/x" in out
        assert "Apple Inc." in out

    def test_warns_loudly_about_duplicate_accessions(self, capsys):
        print_report(self._report(duplicate_accessions=[("0000320193-25-000079", 2)]))
        out = capsys.readouterr().out
        assert "duplicate accession numbers found" in out

    def test_no_duplicates_is_reassuring_not_alarming(self, capsys):
        print_report(self._report(duplicate_accessions=[]))
        out = capsys.readouterr().out
        assert "No duplicate accession numbers" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
