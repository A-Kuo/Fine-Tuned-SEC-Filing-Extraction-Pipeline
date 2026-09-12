"""Tests for evaluation/benchmark_redis_cache.py's report-shaping logic.

Mocks DatabaseManager entirely -- no real Postgres/Redis needed. Only
run_benchmark()'s pure timing/shape logic is exercised directly; main()'s
connection setup and file I/O are glue code, consistent with this repo's
existing convention of testing pure logic separately from CLI glue
(tests/test_kaggle_results.py's docstring states this explicitly).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.benchmark_redis_cache import run_benchmark, ITERATIONS


class TestRunBenchmark:
    def test_stores_once_then_reads_warm_and_cold(self):
        db = MagicMock()
        db.get_extraction = MagicMock(return_value={"filing_id": "f-1"})

        raw = run_benchmark(db, "bench-redis-cache-test")

        db.store_extraction.assert_called_once()
        assert len(raw["warm_times_ms"]) == ITERATIONS
        assert len(raw["cold_times_ms"]) == ITERATIONS
        assert all(t >= 0 for t in raw["warm_times_ms"])
        assert all(t >= 0 for t in raw["cold_times_ms"])

    def test_cache_deleted_before_each_cold_read(self):
        """Cold reads must force a real cache miss each time, not just once."""
        db = MagicMock()
        db.get_extraction = MagicMock(return_value={"filing_id": "f-1"})

        run_benchmark(db, "bench-redis-cache-test")

        assert db.cache.delete.call_count == ITERATIONS

    def test_store_extraction_uses_the_given_filing_id(self):
        db = MagicMock()
        db.get_extraction = MagicMock(return_value={"filing_id": "f-1"})

        run_benchmark(db, "bench-redis-cache-abc123")

        _, kwargs = db.store_extraction.call_args
        assert kwargs["filing_id"] == "bench-redis-cache-abc123"
