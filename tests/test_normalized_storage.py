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
    """upsert_metric() enforces xbrl precedence atomically in the SQL
    statement's WHERE clause (see the docstring on upsert_metric for why the
    previous SELECT-then-resolve-then-write approach was a race). There is no
    longer a SELECT call at all -- Postgres decides at write time whether the
    incoming row actually replaces what's there, so these tests assert (a)
    exactly one statement is executed, (b) the incoming values are always
    what's bound (never a Python-resolved "winner"), and (c) the statement
    text itself contains the precedence guard.
    """

    def test_executes_exactly_one_statement_no_prior_select(self):
        storage = _make_storage()
        mock_cursor = MagicMock()
        storage._connection.cursor = MagicMock(return_value=mock_cursor)

        incoming = MetricRecord(name="revenue", value=1.0, method="llm", confidence=0.5)
        assert storage.upsert_metric("f-1", incoming) is True
        mock_cursor.execute.assert_called_once()

    def test_incoming_values_are_always_bound_as_is(self):
        """Precedence is enforced server-side now -- the incoming llm values
        are passed unconditionally regardless of what (if anything) exists;
        Postgres's WHERE clause decides whether the write actually lands."""
        storage = _make_storage()
        mock_cursor = MagicMock()
        storage._connection.cursor = MagicMock(return_value=mock_cursor)

        incoming = MetricRecord(name="revenue", value=1.0, method="llm", confidence=0.5)
        storage.upsert_metric("f-1", incoming)

        sql, params = mock_cursor.execute.call_args[0]
        assert "llm" in params
        assert 1.0 in params

    def test_statement_contains_xbrl_precedence_guard(self):
        storage = _make_storage()
        mock_cursor = MagicMock()
        storage._connection.cursor = MagicMock(return_value=mock_cursor)

        incoming = MetricRecord(name="revenue", value=1.0, method="llm", confidence=0.5)
        storage.upsert_metric("f-1", incoming)

        sql, _ = mock_cursor.execute.call_args[0]
        assert "ON CONFLICT" in sql
        assert "method = 'xbrl'" in sql
        assert "EXCLUDED.method <> 'xbrl'" in sql


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
        # filing + section + metric + risk + mdna -- one statement each,
        # since upsert_metric no longer issues a prior SELECT.
        assert mock_cursor.execute.call_count == 5


class TestInsertSectionContent:
    """content was previously discarded before insert_section() ever saw it
    -- these confirm both the row-builder and the INSERT itself now carry
    the real section prose (needed for the RAG demo's embeddings)."""

    def test_section_to_row_includes_content(self):
        from src.extraction.normalizer import section_to_row

        section = SectionRecord(
            section_type="mdna", title="MD&A", text="real section prose",
            start=0, end=19, confidence=0.9,
        )
        row = section_to_row("f-1", section)
        assert row["content"] == "real section prose"

    def test_insert_section_writes_content(self):
        storage = _make_storage()
        mock_cursor = MagicMock()
        storage._connection.cursor = MagicMock(return_value=mock_cursor)

        row = {
            "filing_id": "f-1", "section_type": "mdna", "title": "t",
            "char_start": 0, "char_end": 10, "confidence": 0.9,
            "content": "real section prose",
        }
        assert storage.insert_section(row) is True
        params = mock_cursor.execute.call_args[0][1]
        assert "real section prose" in params

    def test_insert_section_upserts_content_on_conflict(self):
        """A re-run against an already-populated row must refresh content,
        not silently skip it (DO UPDATE, not DO NOTHING)."""
        storage = _make_storage()
        mock_cursor = MagicMock()
        storage._connection.cursor = MagicMock(return_value=mock_cursor)

        row = {
            "filing_id": "f-1", "section_type": "mdna", "title": "t",
            "char_start": 0, "char_end": 10, "confidence": 0.9,
            "content": "refreshed prose",
        }
        storage.insert_section(row)
        sql = mock_cursor.execute.call_args[0][0]
        assert "DO UPDATE" in sql
        assert "DO NOTHING" not in sql


class TestEmbeddingMethods:
    def test_update_embeddings_batches_via_execute_values(self):
        """Matches db/sync/transfer_metrics.py's established batching
        pattern -- one execute_values call, not one UPDATE per row.
        execute_values is patched directly (not run against a mocked
        cursor) since it needs cursor.connection.encoding to be a real
        string internally, which a MagicMock can't provide."""
        from unittest.mock import patch

        storage = _make_storage()
        mock_cursor = MagicMock()
        storage._connection.cursor = MagicMock(return_value=mock_cursor)

        rows = [
            {"section_id": 1, "embedding": [0.1, 0.2]},
            {"section_id": 2, "embedding": [0.3, 0.4]},
        ]
        with patch("psycopg2.extras.execute_values") as mock_execute_values:
            updated = storage.update_embeddings(rows)

        assert updated == 2
        mock_execute_values.assert_called_once()
        values_arg = mock_execute_values.call_args[0][2]
        assert values_arg == [(1, [0.1, 0.2]), (2, [0.3, 0.4])]

    def test_update_embeddings_returns_zero_for_empty_rows(self):
        storage = _make_storage()
        assert storage.update_embeddings([]) == 0

    def test_update_embeddings_returns_zero_when_unavailable(self):
        storage = NormalizedStorage("localhost", 5432, "user", "pass", "db")
        storage._available = False
        assert storage.update_embeddings([{"section_id": 1, "embedding": [0.1]}]) == 0

    def test_get_sections_needing_embeddings_filters_correctly(self):
        storage = _make_storage()
        mock_cursor = MagicMock()
        mock_cursor.fetchall = MagicMock(return_value=[
            (1, "f-1", "mdna", "some text"),
        ])
        storage._connection.cursor = MagicMock(return_value=mock_cursor)

        sections = storage.get_sections_needing_embeddings()
        assert len(sections) == 1
        assert sections[0]["content"] == "some text"
        sql = mock_cursor.execute.call_args[0][0]
        assert "content IS NOT NULL" in sql
        assert "embedding IS NULL" in sql


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
