"""Real cold-vs-warm benchmark of DatabaseManager's Redis read-through cache.

The README's "90% reduction in redundant computation / data-fetch latency"
claim named two mechanisms: scripts/fetch_edgar.py's checkpoint dedup
(measured for real in evaluation/benchmark_fetch_dedup.py, 91.1%) and
RedisCache's extraction-result cache (src/storage/database.py) -- this
script measures the second one, which had zero timing instrumentation
anywhere in the repo before this.

Requires a live, reachable Postgres + Redis (see
.github/workflows/live_ingestion_benchmark.yml, which runs this against
docker/docker-compose.smoke.yml's services). Connection comes from
config.yaml's database.* block, overridable via the POSTGRES_*/REDIS_* env
vars src/core/config.py already supports.

Method: store one real extraction (write-through: Postgres + Redis both
warmed), then time repeated warm reads (Redis hit) and repeated cold reads
(cache explicitly deleted before each read, forcing a Postgres fallback
that re-warms Redis). Averaged over multiple iterations, since
sub-millisecond Redis latency is too noisy to trust from a single pair --
a deliberate style difference from benchmark_fetch_dedup.py's single
cold/warm pair, which measures multi-second network calls where averaging
isn't necessary.

Usage:
    python evaluation/benchmark_redis_cache.py
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.extraction.postprocessing import ExtractionResult
from src.storage.database import DatabaseManager

REPO_ROOT = Path(__file__).parent.parent
ITERATIONS = 20


def _time_ms(fn) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000


def run_benchmark(db: DatabaseManager, filing_id: str) -> dict:
    result = ExtractionResult(
        filing_id=filing_id,
        company_name="Benchmark Corp",
        ticker="BNCH",
        filing_type="10-K",
        date="2026-01-01",
        revenue="1000000000",
    )
    db.store_extraction(
        filing_id=filing_id, result=result, confidence=1.0,
        latency_ms=0.0, model_version="benchmark-redis-cache",
    )

    warm_times_ms = []
    for _ in range(ITERATIONS):
        warm_times_ms.append(_time_ms(lambda: db.get_extraction(filing_id)))

    cold_times_ms = []
    for _ in range(ITERATIONS):
        db.cache.delete(filing_id)
        cold_times_ms.append(_time_ms(lambda: db.get_extraction(filing_id)))

    return {
        "warm_times_ms": warm_times_ms,
        "cold_times_ms": cold_times_ms,
    }


def main():
    db = DatabaseManager.from_config()
    if not db.storage._available:
        raise RuntimeError("Could not connect to Postgres. Set POSTGRES_HOST/PORT/USER/PASSWORD/DB.")
    if not db.cache._available:
        raise RuntimeError("Could not connect to Redis. Set REDIS_HOST/PORT.")

    filing_id = f"bench-redis-cache-{uuid.uuid4()}"
    try:
        raw = run_benchmark(db, filing_id)
    finally:
        # No delete-from-Postgres method exists on DatabaseManager/PostgresStorage --
        # clean up the one benchmark row directly.
        cur = db.storage._connection.cursor()
        cur.execute("DELETE FROM extractions WHERE filing_id = %s", (filing_id,))
        db.cache.delete(filing_id)

    mean_warm_ms = sum(raw["warm_times_ms"]) / len(raw["warm_times_ms"])
    mean_cold_ms = sum(raw["cold_times_ms"]) / len(raw["cold_times_ms"])
    reduction_pct = (1 - mean_warm_ms / mean_cold_ms) * 100 if mean_cold_ms > 0 else 0.0

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "DatabaseManager's RedisCache extraction-result cache only "
                             "(src/storage/database.py). Does NOT cover scripts/fetch_edgar.py's "
                             "checkpoint dedup -- that's measured separately in "
                             "evaluation/benchmark_fetch_dedup.py (91.1% measured).",
        "data_provenance": "One real ExtractionResult stored via DatabaseManager.store_extraction() "
                            "against a live Postgres + Redis, then read back repeatedly.",
        "method": f"Real read-through get_extraction() calls, {ITERATIONS} iterations each for warm "
                   f"(Redis hit) and cold (cache explicitly deleted before each read, falling through "
                   f"to Postgres and re-warming Redis) -- not a synthetic microbenchmark. Averaged "
                   f"because sub-millisecond Redis latency is noisy from a single pair.",
        "results": {
            "iterations": ITERATIONS,
            "mean_warm_get_ms": round(mean_warm_ms, 3),
            "mean_cold_get_ms": round(mean_cold_ms, 3),
            "reduction_pct": round(reduction_pct, 1),
        },
    }

    out_path = REPO_ROOT / "evaluation" / "results" / "redis_cache_benchmark.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Warm (Redis hit):  {mean_warm_ms:.3f}ms mean")
    print(f"Cold (Postgres):   {mean_cold_ms:.3f}ms mean")
    print(f"Reduction: {reduction_pct:.1f}%")
    print(f"Report written to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
