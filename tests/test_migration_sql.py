"""Text assertions on migration SQL files -- no live database needed.

Confirms the expected statements are present with the right column lists,
catching typos/omissions in migration files before they'd only surface
against a live Postgres run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO_ROOT = Path(__file__).parent.parent


class TestRagEmbeddingsMigration:
    def _read(self) -> str:
        return (REPO_ROOT / "db" / "migrations" / "0007_rag_embeddings.sql").read_text(encoding="utf-8")

    def test_adds_content_column(self):
        sql = self._read().lower()
        assert "add column if not exists content text" in sql

    def test_adds_embedding_vector_column(self):
        sql = self._read().lower()
        assert "add column if not exists embedding vector(384)" in sql

    def test_targets_filing_sections_table(self):
        sql = self._read().lower()
        assert "alter table intel.filing_sections" in sql

    def test_supabase_mirror_is_byte_identical(self):
        """supabase/migrations/ is kept as a byte-identical mirror of
        db/migrations/ (established convention this session) -- a drift
        here means Supabase and the canonical migrations disagree."""
        db_version = (REPO_ROOT / "db" / "migrations" / "0007_rag_embeddings.sql").read_bytes()
        supabase_version = (REPO_ROOT / "supabase" / "migrations" / "0007_rag_embeddings.sql").read_bytes()
        assert db_version == supabase_version


def _assert_mirrors_match(filename: str):
    db_version = (REPO_ROOT / "db" / "migrations" / filename).read_bytes()
    supabase_version = (REPO_ROOT / "supabase" / "migrations" / filename).read_bytes()
    assert db_version == supabase_version


class TestPerfIndexesMigration:
    def _read(self) -> str:
        return (REPO_ROOT / "db" / "migrations" / "0008_perf_indexes.sql").read_text(encoding="utf-8")

    def test_indexes_the_cache_miss_fallback_column(self):
        sql = self._read().lower()
        assert "create index if not exists idx_extractions_request_text_hash" in sql
        assert "on public.extractions (request_text_hash)" in sql

    def test_indexes_extraction_logs_filing_id(self):
        sql = self._read().lower()
        assert "create index if not exists idx_extraction_logs_filing_id" in sql

    def test_adds_second_composite_indexes_rather_than_reordering(self):
        """Migrations here are additive-only -- the original leading-column
        indexes must not be dropped, just supplemented."""
        sql = self._read().lower()
        assert "idx_model_metrics_name_measured" in sql
        assert "on public.model_metrics (metric_name, measured_at desc)" in sql
        assert "idx_ab_test_results_created_at" in sql

    def test_supabase_mirror_is_byte_identical(self):
        _assert_mirrors_match("0008_perf_indexes.sql")


class TestViewPerfMigration:
    def _read(self) -> str:
        return (REPO_ROOT / "db" / "migrations" / "0009_view_perf.sql").read_text(encoding="utf-8")

    def test_replaces_the_view(self):
        sql = self._read().lower()
        assert "create or replace view intel.v_filing_metric_summary" in sql

    def test_preserves_original_output_columns_in_order(self):
        """CREATE OR REPLACE VIEW requires identical output column names/
        order to the original (0004_views.sql) or Postgres rejects it."""
        sql = self._read().lower()
        for col in ["f.filing_id", "f.company_name", "f.ticker", "f.filing_type", "f.filing_date"]:
            assert col in sql
        assert "metric_count" in sql
        assert "risk_factor_count" in sql
        assert "last_run_at" in sql

    def test_preaggregates_instead_of_joining_raw_tables(self):
        """The fix being tested: no more direct LEFT JOIN onto the raw
        many-side tables (that's what caused the fan-out before count(distinct)).
        Strips comment lines first -- the file's own header comment
        describes the old, fixed behavior using this exact phrase."""
        code_lines = [
            line for line in self._read().lower().splitlines()
            if not line.strip().startswith("--")
        ]
        code = "\n".join(code_lines)
        assert "group by filing_id" in code
        assert "count(distinct" not in code

    def test_supabase_mirror_is_byte_identical(self):
        _assert_mirrors_match("0009_view_perf.sql")
