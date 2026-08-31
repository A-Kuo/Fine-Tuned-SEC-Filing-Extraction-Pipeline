"""Tests for src/normalized_storage.py (intel.* schema persistence).

Uses mocks so tests run without Docker/PostgreSQL, matching the
convention established in tests/test_database.py.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.normalized_storage import NormalizedStorage
from src.core.schemas import (
    FilingMetadata,
    FilingRecord,
    MdnaSummaryRecord,
    MetricRecord,
    RiskFactorRecord,
    SectionRecord,
)


def _make_storage() -> NormalizedStorage:
    storage = NormalizedStorage("localhost", 5432, "user", "pass", "db")
    storage._available = True
    storage._connection = MagicMock()
    return storage


class TestUnavailableGuards:
    def test_upsert_filing_returns_false_when_unavailable(self):
        storage = NormalizedStorage("localhost", 5432, "user", "pass", "db")
        storage._available = False
        assert storage.upsert_filing({}) is False

    def test_upsert_metric_returns_false_when_unavailable(self):
        storage = NormalizedStorage("localhost", 5432, "user", "pass", "db")
        storage._available = False
        m = MetricRecord(name="revenue", method="llm", confidence=0.5)
        assert storage.upsert_metric("f-1", m) is False


class TestUpsertFiling:
    def test_executes_insert(self):
        storage = _make_storage()
        mock_cursor = MagicMock()
        storage._connection.cursor = MagicMock(return_value=mock_cursor)

        row = {
            "filing_id": "f-1", "cik": "123", "accession_no": "acc-1",
            "ticker": "AAPL", "company_name": "Apple", "filing_type": "10-K",
            "filing_date": "2024-01-01", "raw_text_hash": "hash",
        }
        assert storage.upsert_filing(row) is True
        mock_cursor.execute.assert_called_once()


class TestUpsertMetricPrecedence:
    def test_no_existing_row_inserts_incoming_as_is(self):
        storage = _make_storage()
        mock_cursor = MagicMock()
        mock_cursor.fetchone = MagicMock(return_value=None)
        storage._connection.cursor = MagicMock(return_value=mock_cursor)

        incoming = MetricRecord(name="revenue", value=1.0, method="llm", confidence=0.5)
        assert storage.upsert_metric("f-1", incoming) is True

        # Second execute call is the INSERT ... ON CONFLICT; check the method written.
        insert_call = mock_cursor.execute.call_args_list[-1]
        params = insert_call[0][1]
        assert "llm" in params

    def test_existing_xbrl_row_blocks_llm_overwrite(self):
        storage = _make_storage()
        mock_cursor = MagicMock()
        # _get_existing_metric SELECT returns an existing xbrl row.
        mock_cursor.fetchone = MagicMock(return_value=(
            "revenue", 500.0, "usd", "", "", "xbrl", 0.99, "xbrl_source", "ev", None,
        ))
        storage._connection.cursor = MagicMock(return_value=mock_cursor)

        incoming = MetricRecord(name="revenue", value=1.0, method="llm", confidence=0.5)
        assert storage.upsert_metric("f-1", incoming) is True

        insert_call = mock_cursor.execute.call_args_list[-1]
        params = insert_call[0][1]
        # The xbrl value/method should have been written back, not the llm incoming one.
        assert "xbrl" in params
        assert 500.0 in params


class TestInsertRiskFactor:
    def test_executes_insert(self):
        storage = _make_storage()
        mock_cursor = MagicMock()
        storage._connection.cursor = MagicMock(return_value=mock_cursor)

        row = {
            "filing_id": "f-1", "text": "risk text", "source_section": "risk_factors",
            "confidence": 0.8, "risk_hash": "h" * 64,
        }
        assert storage.insert_risk_factor(row) is True
        mock_cursor.execute.assert_called_once()


class TestSaveFilingRecord:
    def test_writes_all_tables(self):
        storage = _make_storage()
        mock_cursor = MagicMock()
        mock_cursor.fetchone = MagicMock(return_value=None)
        storage._connection.cursor = MagicMock(return_value=mock_cursor)

        record = FilingRecord(
            metadata=FilingMetadata(filing_id="f-1", filing_type="10-K"),
            sections=[
                SectionRecord(
                    section_type="mdna", title="t", text="x", start=0, end=1, confidence=0.9
                )
            ],
            metrics=[MetricRecord(name="revenue", value=1.0, method="llm", confidence=0.5)],
            risk_factors=[RiskFactorRecord(text="risk", confidence=0.5)],
            mdna=MdnaSummaryRecord(summary="s"),
        )

        assert storage.save_filing_record(record) is True
        # filing + section + metric(select+insert) + risk + mdna
        assert mock_cursor.execute.call_count >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
