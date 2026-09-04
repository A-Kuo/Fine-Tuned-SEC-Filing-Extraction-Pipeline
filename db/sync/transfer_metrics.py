"""Bulk-load financial_metrics rows into Postgres (intel.financial_metrics)
and report real ingestion throughput.

This is the "postgres native" replacement for a hypothetical
row-at-a-time /API-mediated insert path: earlier code in this repo
(src/storage/normalized_storage.py) only ever exposes one-row-at-a-time
upsert_metric() calls, which cannot approach the >10,000 records/sec target --
a single-row psycopg2 round trip (network + parse + plan + commit) runs
roughly 1-3k/sec at best even on localhost. This script batches rows with
psycopg2.extras.execute_values inside one transaction, which is what actually
gets there.

XBRL precedence (xbrl facts are never overwritten by heuristic/llm; among
non-xbrl methods, last write wins) is enforced in the ON CONFLICT clause
itself -- see resolve_metric_precedence() in src/extraction/normalizer.py for
the Python-level equivalent this mirrors.

Usage:
    python db/sync/transfer_metrics.py --benchmark --records 200000
    python db/sync/transfer_metrics.py --benchmark --records 200000 --batch-size 5000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import psycopg2.extras

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

UPSERT_SQL = """
    INSERT INTO intel.financial_metrics
        (filing_id, metric_name, period, segment, value, unit, method,
         confidence, source_section, evidence_text, model_version)
    VALUES %s
    ON CONFLICT (filing_id, metric_name, period, segment) DO UPDATE SET
        value = EXCLUDED.value,
        unit = EXCLUDED.unit,
        method = EXCLUDED.method,
        confidence = EXCLUDED.confidence,
        source_section = EXCLUDED.source_section,
        evidence_text = EXCLUDED.evidence_text,
        model_version = EXCLUDED.model_version,
        updated_at = now()
    WHERE NOT (
        intel.financial_metrics.method = 'xbrl'
        AND EXCLUDED.method <> 'xbrl'
    )
"""


def _connect():
    """Connect using the same env vars docker/.env.docker.example declares,
    with localhost defaults for running this script outside a container
    against `docker compose up postgres`."""
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        dbname=os.environ.get("POSTGRES_DB", "postgres"),
    )


def ensure_benchmark_filings(conn, n_filings: int) -> list[str]:
    """financial_metrics.filing_id has an FK to intel.filings -- seed the
    small number of parent filings the benchmark's metric rows reference."""
    filing_ids = [f"bench-filing-{i:04d}" for i in range(n_filings)]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO intel.filings (filing_id, cik, ticker, company_name, filing_type)
               VALUES %s ON CONFLICT (filing_id) DO NOTHING""",
            [(fid, "0000000000", "BNCH", "Benchmark Corp", "10-K") for fid in filing_ids],
        )
    conn.commit()
    return filing_ids


def generate_rows(n_records: int, filing_ids: list[str]):
    """Synthetic but realistically-shaped metric rows. Distinct metric_name
    per row (not the same handful reused) so this measures real INSERT
    throughput, not update-path throughput."""
    methods = ["xbrl", "heuristic", "llm"]
    for i in range(n_records):
        yield (
            filing_ids[i % len(filing_ids)],
            f"metric_{i:08d}",
            "",
            "",
            round(random.uniform(0, 1e9), 4),
            "usd",
            random.choice(methods),
            round(random.uniform(0.5, 1.0), 4),
            "financial_statements",
            "benchmark evidence text",
            None,
        )


def run_benchmark(n_records: int, batch_size: int, n_filings: int) -> dict:
    conn = _connect()
    try:
        filing_ids = ensure_benchmark_filings(conn, n_filings)

        rows = list(generate_rows(n_records, filing_ids))

        start = time.perf_counter()
        with conn.cursor() as cur:
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                psycopg2.extras.execute_values(cur, UPSERT_SQL, batch, page_size=len(batch))
        conn.commit()
        elapsed_s = time.perf_counter() - start

        records_per_sec = n_records / elapsed_s if elapsed_s > 0 else float("inf")

        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM intel.financial_metrics WHERE filing_id LIKE 'bench-filing-%'"
            )
            row_count_after = cur.fetchone()[0]

        pg_version = None
        with conn.cursor() as cur:
            cur.execute("SHOW server_version")
            pg_version = cur.fetchone()[0]

        return {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "target": "> 10,000 records/sec",
            "n_records": n_records,
            "batch_size": batch_size,
            "n_filings": n_filings,
            "elapsed_seconds": round(elapsed_s, 4),
            "records_per_sec": round(records_per_sec, 1),
            "meets_target": records_per_sec > 10_000,
            "table": "intel.financial_metrics",
            "insert_method": "psycopg2.extras.execute_values, batched, single transaction",
            "postgres_version": pg_version,
            "row_count_in_table_after_run": row_count_after,
            "note": (
                "Distinct metric_name per row (metric_00000000, metric_00000001, ...) "
                "so this measures INSERT throughput, not the ON CONFLICT UPDATE path -- "
                "a second run of this script against the same table WOULD measure the "
                "update path instead, since all rows would already exist."
            ),
        }
    finally:
        conn.close()


def cleanup_benchmark_data(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM intel.filings WHERE filing_id LIKE 'bench-filing-%'")
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Bulk-load / benchmark intel.financial_metrics ingestion")
    parser.add_argument("--benchmark", action="store_true", help="Run the throughput benchmark")
    parser.add_argument("--records", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--n-filings", type=int, default=50)
    parser.add_argument("--cleanup", action="store_true", help="Delete benchmark rows and exit")
    args = parser.parse_args()

    if args.cleanup:
        conn = _connect()
        try:
            cleanup_benchmark_data(conn)
            print("Benchmark rows deleted.")
        finally:
            conn.close()
        return

    if not args.benchmark:
        parser.error("Nothing to do -- pass --benchmark or --cleanup")

    results = run_benchmark(args.records, args.batch_size, args.n_filings)

    out_dir = REPO_ROOT / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ingestion_benchmark.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Inserted {results['n_records']:,} rows in {results['elapsed_seconds']}s")
    print(f"Throughput: {results['records_per_sec']:,.0f} records/sec")
    print(f"Target (>10,000/sec): {'MET' if results['meets_target'] else 'NOT MET'}")
    print(f"Postgres: {results['postgres_version']}")
    print(f"\nReport written to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
