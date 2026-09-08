"""Build embeddings for the lightweight RAG demo.

Reads every intel.filing_sections row that has real content persisted but
no embedding yet, embeds it locally via sentence-transformers (no API key,
no cost), and writes the embeddings back in one batch.

This is a demo-only, workflow_dispatch-triggered step (see
.github/workflows/rag_index.yml) -- sentence-transformers pulls in torch
even for CPU-only use, which requirements-ci.txt deliberately excludes to
keep the main push-triggered CI fast. Install it separately:
    pip install sentence-transformers pgvector

Usage:
    python scripts/build_rag_index.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.config import load_config
from src.rag.embed import Embedder
from src.storage.normalized_storage import NormalizedStorage


def _connect_storage() -> NormalizedStorage:
    storage = NormalizedStorage(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        database=os.environ.get("POSTGRES_DB", "postgres"),
    )
    if not storage.connect():
        raise RuntimeError(
            "Could not connect to Postgres. Set POSTGRES_HOST/PORT/USER/PASSWORD/DB."
        )
    return storage


def build_index(storage: NormalizedStorage, embedder: Embedder) -> int:
    sections = storage.get_sections_needing_embeddings()
    if not sections:
        logger.info("No sections need embedding -- index is already up to date.")
        return 0

    logger.info(f"Embedding {len(sections)} section(s)...")
    texts = [s["content"] for s in sections]
    vectors = embedder.embed(texts)

    rows = [
        {"section_id": s["section_id"], "embedding": v}
        for s, v in zip(sections, vectors)
    ]
    updated = storage.update_embeddings(rows)
    logger.info(f"Wrote {updated} embedding(s).")
    return updated


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()

    config = load_config()
    rag_cfg = config.get("rag", {})
    embedder = Embedder(model_name=rag_cfg.get("embedding_model"))

    storage = _connect_storage()
    updated = build_index(storage, embedder)
    print(f"Embedded: {updated}")


if __name__ == "__main__":
    main()
