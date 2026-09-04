"""Give src/storage/normalized_storage.py its first real, non-mocked caller.

Per docs/TRUTH_AUDIT.md: the intel.* schema (db/migrations/0003_intel_schema.sql)
is real and well-designed, but NormalizedStorage -- the only code that writes
to it -- has zero callers anywhere in serving/pipeline/monitoring code. It is
covered only by mocked tests (tests/test_normalized_storage.py).

This is deliberately an OFFLINE BATCH JOB, not a change to the live serving
path. serving/api.py's hot path only ever builds a flat ExtractionResult
(architecture A); making it build a normalized FilingRecord (architecture B)
on every request would add untested extraction-pipeline latency with no
Docker/load-testing available in this environment to validate the change is
safe. Running the real pipeline over a known corpus and writing the result
here is a lower-risk way to prove intel.* actually gets written to by real
code, without touching production request latency.

Reuses evaluation/schema_conformance.py's corpus loaders so this and the
schema-conformance report are always looking at the same filings.

Usage:
    python db/sync/sync_normalized_from_pipeline.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from loguru import logger

from evaluation.schema_conformance import _real_edgar_corpus, _synthetic_corpus
from src.extraction.pipeline import build_filing_record
from src.storage.normalized_storage import NormalizedStorage


def _connect_storage() -> NormalizedStorage:
    """Same env vars/localhost-default convention as db/sync/transfer_metrics.py."""
    storage = NormalizedStorage(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        database=os.environ.get("POSTGRES_DB", "postgres"),
    )
    storage.connect()
    return storage


def main():
    corpus = _synthetic_corpus() + _real_edgar_corpus()
    if len(corpus) == 1:
        logger.warning(
            "No real EDGAR filings found (data/raw_edgar/manifest.jsonl "
            "missing) -- syncing the synthetic filing only. Run "
            "scripts/fetch_edgar.py first for a fuller sync."
        )

    storage = _connect_storage()
    if not storage._available:
        logger.error(
            "Could not connect to Postgres. This script needs a live "
            "database -- see docker/docker-compose.smoke.yml (docker compose "
            "up postgres)."
        )
        sys.exit(1)

    written, failed = 0, 0
    try:
        for case in corpus:
            record = build_filing_record(
                case["text"],
                filing_id=case["filing_id"],
                filing_type=case["filing_type"],
                company_name=case.get("company_name"),
                ticker=case.get("ticker"),
                filing_date=case.get("filing_date"),
            )
            ok = storage.save_filing_record(record)
            if ok:
                written += 1
                logger.info(f"Synced {case['filing_id']} ({case['corpus']}) to intel.*")
            else:
                failed += 1
                logger.error(f"Failed to sync {case['filing_id']}")
    finally:
        storage.close()

    print(f"Synced: {written}  Failed: {failed}  Total: {len(corpus)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
