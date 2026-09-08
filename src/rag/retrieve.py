"""Cosine-similarity retrieval over intel.filing_sections.embedding.

Real data scale here is 5-6 filings and a few dozen sections -- a
brute-force pgvector `<=>` (cosine distance) scan with no ANN index is
entirely adequate; see db/migrations/0007_rag_embeddings.sql for why no
ivfflat/hnsw index was added.

Takes a NormalizedStorage instance rather than opening its own connection,
so tests can inject `storage._connection = MagicMock()` exactly like
tests/test_normalized_storage.py's existing pattern -- no real Postgres
needed to unit-test the query shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.storage.normalized_storage import NormalizedStorage


def retrieve(
    question_embedding: list[float],
    top_k: int,
    storage: "NormalizedStorage",
) -> list[dict]:
    """Return the top_k filing_sections rows most similar to the question.

    Each result: {section_id, filing_id, section_type, content, distance}.
    Lower distance = more similar (pgvector's <=> is cosine distance, not
    similarity).
    """
    if not storage._available:
        return []

    cur = storage._connection.cursor()
    cur.execute(
        """
        SELECT section_id, filing_id, section_type, content,
               embedding <=> %s::vector AS distance
        FROM intel.filing_sections
        WHERE embedding IS NOT NULL
        ORDER BY distance
        LIMIT %s
        """,
        (question_embedding, top_k),
    )
    cols = ["section_id", "filing_id", "section_type", "content", "distance"]
    return [dict(zip(cols, row)) for row in cur.fetchall()]
