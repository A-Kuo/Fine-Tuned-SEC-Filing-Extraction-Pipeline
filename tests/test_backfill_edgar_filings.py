"""Tests for scripts/backfill_edgar_filings.py.

Pure field-mapping functions (derive_status, build_entity_row,
pick_entity_filings, build_filing_row) are exercised against a canned real-
shaped submissions fixture -- no network. The upsert helpers patch
psycopg2.extras.execute_values directly (the established pattern in
tests/test_normalized_storage.py -- execute_values needs a real cursor
connection encoding a MagicMock cursor can't provide). run() is exercised
with a fake fetch_submissions and a fake DB connection/cursor -- no live
network or Postgres.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.backfill_edgar_filings as backfill_mod
from scripts.backfill_edgar_filings import (
    ENTITY_UPSERT_SQL,
    FILING_UPSERT_SQL,
    build_entity_row,
    build_filing_row,
    derive_status,
    pick_entity_filings,
    run,
    upsert_entity,
    upsert_filings,
)


def _submissions(**overrides) -> dict:
    """Shaped like a real data.sec.gov/submissions/CIK##########.json
    response (documented fields: name, tickers, exchanges, sic,
    sicDescription, category, fiscalYearEnd, filings.recent.*)."""
    base = {
        "cik": "320193",
        "name": "Apple Inc.",
        "tickers": ["AAPL"],
        "exchanges": ["Nasdaq"],
        "sic": "3571",
        "sicDescription": "ELECTRONIC COMPUTERS",
        "category": "Large accelerated filer",
        "fiscalYearEnd": "0928",
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q", "8-K", "10-K/A", "NT 10-Q", "4"],
                "accessionNumber": [
                    "0000320193-25-000079", "0000320193-25-000060",
                    "0000320193-25-000050", "0000320193-24-000010",
                    "0000320193-24-000005", "0000320193-24-000001",
                ],
                "filingDate": [
                    "2025-10-31", "2025-08-01", "2025-05-01",
                    "2024-11-01", "2024-08-15", "2024-01-05",
                ],
                "acceptanceDateTime": [
                    "2025-10-31T18:00:00.000Z", "2025-08-01T18:00:00.000Z",
                    "2025-05-01T18:00:00.000Z", "2024-11-01T18:00:00.000Z",
                    "2024-08-15T18:00:00.000Z", "2024-01-05T18:00:00.000Z",
                ],
                "reportDate": ["2025-09-27", "2025-06-28", "", "2024-09-28", "", ""],
                "items": ["", "", "2.02,9.01", "", "", ""],
                "primaryDocument": [
                    "aapl-20250927.htm", "aapl-20250628.htm", "aapl-8k.htm",
                    "aapl-10ka.htm", "aapl-nt10q.htm", "aapl-4.xml",
                ],
                "isXBRL": [1, 1, 1, 1, 0, 0],
            }
        },
    }
    base.update(overrides)
    return base


class TestDeriveStatus:
    @pytest.mark.parametrize("form,expected", [
        ("10-K", "accepted"),
        ("10-Q", "accepted"),
        ("8-K", "accepted"),
        ("10-K/A", "amended"),
        ("10-Q/A", "amended"),
        ("NT 10-K", "late_notice"),
        ("NT 10-Q", "late_notice"),
    ])
    def test_status_is_a_fact_about_the_form_string(self, form, expected):
        assert derive_status(form) == expected


class TestBuildEntityRow:
    def test_every_field_comes_from_the_real_response(self):
        row = build_entity_row(320193, _submissions())

        assert row == {
            "cik": "0000320193",
            "name": "Apple Inc.",
            "tickers": ["AAPL"],
            "exchange": "Nasdaq",
            "sic": "3571",
            "sic_description": "ELECTRONIC COMPUTERS",
            "filer_category": "Large accelerated filer",
            "fiscal_year_end": "0928",
        }

    def test_cik_is_zero_padded_to_ten_digits(self):
        assert build_entity_row(1, _submissions())["cik"] == "0000000001"

    def test_missing_exchanges_gives_null_not_a_crash(self):
        row = build_entity_row(320193, _submissions(exchanges=[]))
        assert row["exchange"] is None

    def test_no_field_is_fabricated_when_the_response_lacks_it(self):
        row = build_entity_row(1, {"name": "Bare Corp"})
        assert row["tickers"] == []
        assert row["sic"] is None
        assert row["exchange"] is None


class TestPickEntityFilings:
    def test_only_the_requested_forms_are_picked_and_nothing_is_capped(self):
        picked = pick_entity_filings(_submissions(), ["10-K", "10-Q", "8-K", "10-K/A", "NT 10-Q"])
        assert [p["form"] for p in picked] == ["10-K", "10-Q", "8-K", "10-K/A", "NT 10-Q"]

    def test_form_4_is_excluded_when_not_requested(self):
        picked = pick_entity_filings(_submissions(), ["10-K"])
        assert len(picked) == 1
        assert picked[0]["form"] == "10-K"

    def test_carries_the_real_extra_columns_pick_filings_does_not(self):
        picked = pick_entity_filings(_submissions(), ["10-K"])
        row = picked[0]
        assert row["acceptanceDateTime"] == "2025-10-31T18:00:00.000Z"
        assert row["reportDate"] == "2025-09-27"
        assert row["isXBRL"] is True

    def test_empty_string_report_date_and_items_become_none(self):
        picked = pick_entity_filings(_submissions(), ["8-K"])
        row = picked[0]
        assert row["reportDate"] is None
        assert row["items"] == "2.02,9.01"

    def test_no_recent_filings_block_returns_empty(self):
        assert pick_entity_filings({"filings": {"recent": {}}}, ["10-K"]) == []
        assert pick_entity_filings({}, ["10-K"]) == []


class TestBuildFilingRow:
    def test_status_and_source_url_are_derived_correctly(self):
        entry = pick_entity_filings(_submissions(), ["10-K/A"])[0]
        row = build_filing_row(320193, entry)

        assert row["status"] == "amended"
        assert row["accession_no"] == "0000320193-24-000010"
        assert row["source_url"] == (
            "https://www.sec.gov/Archives/edgar/data/320193/000032019324000010/aapl-10ka.htm"
        )
        assert row["cik"] == "0000320193"

    def test_late_notice_status(self):
        entry = pick_entity_filings(_submissions(), ["NT 10-Q"])[0]
        assert build_filing_row(320193, entry)["status"] == "late_notice"


class TestUpsertHelpers:
    def test_upsert_entity_calls_execute_values_once_with_the_row_tuple(self):
        row = build_entity_row(320193, _submissions())
        with patch("psycopg2.extras.execute_values") as mock_ev:
            upsert_entity(MagicMock(), row)

        mock_ev.assert_called_once()
        args = mock_ev.call_args[0]
        assert args[1] == ENTITY_UPSERT_SQL
        values = args[2][0]
        assert values[0] == "0000320193"
        assert values[1] == "Apple Inc."

    def test_upsert_filings_batches_all_rows_in_one_call(self):
        entries = pick_entity_filings(_submissions(), ["10-K", "10-Q"])
        rows = [build_filing_row(320193, e) for e in entries]
        with patch("psycopg2.extras.execute_values") as mock_ev:
            upsert_filings(MagicMock(), rows)

        mock_ev.assert_called_once()
        args = mock_ev.call_args[0]
        assert args[1] == FILING_UPSERT_SQL
        assert len(args[2]) == 2

    def test_upsert_filings_does_nothing_for_an_empty_list(self):
        with patch("psycopg2.extras.execute_values") as mock_ev:
            upsert_filings(MagicMock(), [])
        mock_ev.assert_not_called()

    def test_upsert_sql_never_overwrites_on_conflict_with_missing_no_op_columns(self):
        """Sanity check on the SQL text itself: both statements key their
        ON CONFLICT off the real primary keys (cik / accession_no)."""
        assert "ON CONFLICT (cik) DO UPDATE" in ENTITY_UPSERT_SQL
        assert "ON CONFLICT (accession_no) DO UPDATE" in FILING_UPSERT_SQL


class FakeCursor:
    def __init__(self, recorder):
        self.recorder = recorder

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.recorder.append(("execute", sql, params))


class FakeConn:
    def __init__(self):
        self.executed = []
        self.commits = 0

    def cursor(self):
        return FakeCursor(self.executed)

    def commit(self):
        self.commits += 1


class TestRun:
    TICKER_TO_CIK = {"AAPL": 320193, "MSFT": 789019}

    def _patch_fetch(self, monkeypatch, by_ticker: dict[str, dict]):
        cik_to_ticker = {cik: t for t, cik in self.TICKER_TO_CIK.items()}

        def fake_fetch_submissions(client, cik, limiter, config):
            ticker = cik_to_ticker[cik]
            if ticker not in by_ticker:
                raise RuntimeError("404")
            return by_ticker[ticker]

        monkeypatch.setattr(backfill_mod, "fetch_submissions", fake_fetch_submissions)

    def test_dry_run_touches_no_database(self, monkeypatch):
        self._patch_fetch(monkeypatch, {"AAPL": _submissions()})

        summary = run(
            ["AAPL"], self.TICKER_TO_CIK, ["10-K"], MagicMock(), {},
            conn=None, dry_run=True, client=MagicMock(),
        )

        assert summary["companies"] == 1
        assert summary["filings_upserted"] == 1

    def test_writes_entity_and_filings_and_records_an_ingest_run(self, monkeypatch):
        self._patch_fetch(monkeypatch, {"AAPL": _submissions()})
        conn = FakeConn()

        with patch("psycopg2.extras.execute_values") as mock_ev:
            summary = run(
                ["AAPL"], self.TICKER_TO_CIK, ["10-K", "10-Q"], MagicMock(), {},
                conn=conn, dry_run=False, client=MagicMock(),
            )

        assert summary["companies"] == 1
        assert summary["filings_upserted"] == 2
        assert mock_ev.call_count == 2  # one entity upsert, one filings upsert
        assert any("INSERT INTO edgar.ingest_runs" in e[1] for e in conn.executed)
        assert conn.commits >= 2  # per-company commit + the ingest_runs commit

    def test_an_unknown_ticker_is_skipped_not_fatal(self, monkeypatch):
        self._patch_fetch(monkeypatch, {"AAPL": _submissions()})

        summary = run(
            ["AAPL", "ZZZZ"], self.TICKER_TO_CIK, ["10-K"], MagicMock(), {},
            conn=None, dry_run=True, client=MagicMock(),
        )

        assert summary["companies"] == 1
        assert summary["skipped"] == ["ZZZZ"]

    def test_a_failed_fetch_for_one_ticker_does_not_stop_the_others(self, monkeypatch):
        self._patch_fetch(monkeypatch, {"AAPL": _submissions()})  # MSFT fetch will raise

        summary = run(
            ["AAPL", "MSFT"], self.TICKER_TO_CIK, ["10-K"], MagicMock(), {},
            conn=None, dry_run=True, client=MagicMock(),
        )

        assert summary["companies"] == 1
        assert summary["skipped"] == ["MSFT"]

    def test_resume_skips_ciks_already_in_the_checkpoint(self, monkeypatch, tmp_path):
        self._patch_fetch(monkeypatch, {"AAPL": _submissions()})
        (tmp_path / ".fetch_checkpoint.json").write_text('{"fetched": ["0000320193"], "total": 1}')

        summary = run(
            ["AAPL"], self.TICKER_TO_CIK, ["10-K"], MagicMock(), {},
            conn=None, dry_run=True, checkpoint_dir=tmp_path, resume=True, client=MagicMock(),
        )

        assert summary["companies"] == 0

    def test_checkpoint_is_saved_after_each_company_for_dry_runs_too(self, monkeypatch, tmp_path):
        self._patch_fetch(monkeypatch, {"AAPL": _submissions()})

        run(
            ["AAPL"], self.TICKER_TO_CIK, ["10-K"], MagicMock(), {},
            conn=None, dry_run=True, checkpoint_dir=tmp_path, client=MagicMock(),
        )

        import json

        saved = json.loads((tmp_path / ".fetch_checkpoint.json").read_text())
        assert saved["fetched"] == ["0000320193"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
