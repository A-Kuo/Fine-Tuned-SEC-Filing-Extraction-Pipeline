"""Query the lightweight RAG demo end-to-end: embed -> retrieve -> generate.

Loads the real fine-tuned model for generation, so this is meant for
local/GPU use, not CI (see scripts/build_rag_index.py for the CI-safe
embedding-only step). Prints the answer plus the filing_ids it was allowed
to cite.

Usage:
    python scripts/rag_query.py --question "What risks did Apple disclose?"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.config import load_config
from src.rag.embed import Embedder
from src.rag.generate import generate_answer
from src.rag.retrieve import retrieve
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, help="Question to ask over the indexed filings")
    parser.add_argument("--top-k", type=int, default=None, help="Override config.yaml's rag.top_k")
    args = parser.parse_args()

    config = load_config()
    rag_cfg = config.get("rag", {})
    top_k = args.top_k or rag_cfg.get("top_k", 5)
    max_tokens = rag_cfg.get("max_tokens", 512)

    storage = _connect_storage()
    embedder = Embedder(model_name=rag_cfg.get("embedding_model"))

    from src.extraction.model import FinancialLLM

    print("Loading model (this can take a while on first run)...")
    model = FinancialLLM.from_config()

    question_embedding = embedder.embed([args.question])[0]
    retrieved = retrieve(question_embedding, top_k=top_k, storage=storage)

    answer, cited = generate_answer(
        args.question,
        retrieved,
        generate_fn=lambda prompt: model.generate(prompt, max_tokens=max_tokens),
    )

    print(f"\nAnswer: {answer}")
    print(f"\nSources: {', '.join(cited) if cited else '(none retrieved)'}")


if __name__ == "__main__":
    main()
