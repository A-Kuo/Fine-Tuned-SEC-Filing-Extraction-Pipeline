"""Backfill the public.* live-serving schema with real, honestly-derived data.

Nobody has ever run the live /extract API against Supabase, so public.*
(extractions, extraction_logs, ab_test_results, pipeline_stages) is empty
next to the populated intel.* batch schema. This script populates it from
the same real EDGAR filings scripts/fetch_edgar.py already fetches --
identity fields (company_name, ticker, filing_type, date) come straight
from .meta.json, and numeric fields (revenue, net_income, total_assets,
total_liabilities, eps) come from real XBRL facts (scripts/parse_xbrl.py),
reusing the exact extraction the fetch step already performed.

Deliberately NOT populated (see plan for the reasoning):
  - No regex/heuristic fallback on raw filing prose for missing XBRL
    fields -- _regex_extract()'s label:value patterns almost never match
    real 10-K prose, so running it would dress up a near-certain miss as
    a labeled attempt. Missing fields stay None.
  - model_metrics: no real production writer exists anywhere in this repo.
  - webhook_failures: the only real write path requires a registered
    webhook to genuinely fail 3 delivery attempts; manufacturing a
    failure just to populate a table isn't a real demonstration.
  - ab_test_results rows here are single-arm (is_challenger=False) --
    there is no second real model variant to compare against, so this is
    NOT a real A/B test, just this backfill's own attempt logged honestly.

Usage:
    python scripts/backfill_live_schema.py

Requires data/raw_edgar/manifest.jsonl to already exist (run
scripts/fetch_edgar.py first) and POSTGRES_HOST/PORT/USER/PASSWORD/DB
environment variables (matching db/sync/sync_normalized_from_pipeline.py's
convention -- not config.yaml's database.postgres block).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.config import get_project_root
from src.extraction.postprocessing import ExtractionResult
from src.storage.database import PostgresStorage

sys.path.insert(0, str(Path(__file__).parent))
from parse_xbrl import map_to_training_fields

MODEL_VERSION = "backfill-xbrl-v1"


def _connect_storage() -> PostgresStorage:
    storage = PostgresStorage(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        database=os.environ.get("POSTGRES_DB", "postgres"),
    )
    if not storage.connect():
        raise RuntimeError(
            "Could not connect to Postgres. Set POSTGRES_HOST/PORT/USER/PASSWORD/DB "
            "-- see docker/docker-compose.smoke.yml (docker compose up postgres) for local use."
        )
    return storage


def _load_manifest() -> list[dict]:
    manifest_path = get_project_root() / "data" / "raw_edgar" / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found -- run scripts/fetch_edgar.py first to fetch real filings."
        )
    entries = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def _build_result(entry: dict) -> tuple[ExtractionResult, bool]:
    """Build an ExtractionResult from real .meta.json + .xbrl.json data.

    Returns (result, has_real_facts). Numeric fields with no matching XBRL
    fact stay None -- see module docstring for why no fallback is used.
    """
    result = ExtractionResult(
        filing_id=f"{entry['ticker']}-{entry['accessionNumber']}",
        company_name=entry.get("company"),
        ticker=entry.get("ticker"),
        filing_type=entry.get("form"),
        date=entry.get("filingDate"),
    )

    has_real_facts = False
    xbrl_path = entry.get("xbrl_path")
    if xbrl_path:
        full_path = get_project_root() / xbrl_path
        if full_path.exists():
            facts = json.loads(full_path.read_text(encoding="utf-8"))
            mapped = map_to_training_fields(facts)
            result.revenue = mapped.get("revenue")
            result.net_income = mapped.get("net_income")
            result.total_assets = mapped.get("total_assets")
            result.total_liabilities = mapped.get("total_liabilities")
            result.eps = mapped.get("eps")
            has_real_facts = any(v is not None for v in mapped.values())

    return result, has_real_facts


def backfill(storage: PostgresStorage) -> tuple[int, int]:
    entries = _load_manifest()
    synced, failed = 0, 0

    for entry in entries:
        filing_id = f"{entry['ticker']}-{entry['accessionNumber']}"
        start = time.time()
        try:
            result, has_real_facts = _build_result(entry)
            confidence = 1.0 if has_real_facts else 0.0
            latency_ms = (time.time() - start) * 1000

            ok = storage.store_extraction(
                filing_id, result, confidence=confidence,
                latency_ms=latency_ms, model_version=MODEL_VERSION,
                method="xbrl", ticker=entry.get("ticker"), sector=None,
                fiscal_year_end=None,
            )
            status = "success" if ok else "failed"
            storage.log_extraction(filing_id, status, latency_ms, MODEL_VERSION)
            # Single-arm only -- no second real model variant exists to
            # compare against. This is NOT a real A/B test.
            storage.record_ab_result(
                filing_id, MODEL_VERSION, is_challenger=False,
                confidence_score=confidence, status=status,
                latency_ms=int(latency_ms),
            )
            storage.upsert_pipeline_stage(filing_id, "extracted", ticker=entry.get("ticker"))

            if ok:
                synced += 1
                logger.info(f"Backfilled {filing_id} (has_real_facts={has_real_facts})")
            else:
                failed += 1
                logger.warning(f"Failed to backfill {filing_id}")
        except Exception as e:
            failed += 1
            logger.error(f"Backfill error for {filing_id}: {e}")

    return synced, failed


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()

    storage = _connect_storage()
    synced, failed = backfill(storage)
    print(f"Synced: {synced}  Failed: {failed}  Total: {synced + failed}")
    if synced == 0 and failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
