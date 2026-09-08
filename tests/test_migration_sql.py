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
