"""Real cold-vs-warm benchmark of scripts/fetch_edgar.py's checkpoint dedup.

The README's "90% reduction in redundant computation / data-fetch latency"
claim had no evidence behind it -- this produces real, measured evidence
for one of the two mechanisms that could substantiate it (the other,
RedisCache's extraction-result cache, needs a live Postgres/Redis and is
covered separately by scripts/smoke_test.py once Docker is available).

Measures actual wall-clock time for the same 5 real tickers:
  cold: checkpoint cleared, every filing re-fetched over the network
  warm: checkpoint present, every filing skipped before any network call
        (see scripts/fetch_edgar.py's `if acc in already_fetched: continue`)

This is a real subprocess-level timing of the actual script real users run,
not a synthetic microbenchmark -- network variance means the exact
percentage will vary run to run, which is reported honestly rather than
rounded to a fixed target.

Usage:
    python evaluation/benchmark_fetch_dedup.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TICKERS = ["AAPL", "MSFT", "KO", "MRNA", "O"]


def _run_fetch(extra_args: list[str]) -> float:
    start = time.time()
    result = subprocess.run(
        [sys.executable, "scripts/fetch_edgar.py", "--tickers", *TICKERS, "--per-ticker", "1", *extra_args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(f"fetch_edgar.py failed: {result.stderr[-2000:]}")
    return elapsed


def main():
    print("Cold run (clearing checkpoint, re-fetching all 5 filings over the network)...")
    cold_seconds = _run_fetch(["--clear-checkpoint"])
    print(f"  cold: {cold_seconds:.2f}s")

    print("Warm run (checkpoint now populated from the cold run above)...")
    warm_seconds = _run_fetch(["--resume"])
    print(f"  warm: {warm_seconds:.2f}s")

    reduction_pct = (1 - warm_seconds / cold_seconds) * 100 if cold_seconds > 0 else 0.0

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "scripts/fetch_edgar.py's checkpoint-based dedup only (skips already-fetched "
                             "accessions before any network call). Does NOT cover RedisCache's "
                             "extraction-result cache (src/storage/database.py) -- that needs a live "
                             "Postgres/Redis and is a separate, not-yet-run benchmark.",
        "data_provenance": f"Real network fetches of {len(TICKERS)} real SEC EDGAR 10-K filings "
                            f"({', '.join(TICKERS)}), same tickers used throughout this repo's evidence artifacts.",
        "method": "Two real subprocess invocations of the actual script a user runs, timed end-to-end "
                  "wall-clock -- not a synthetic microbenchmark. Network variance means the exact "
                  "percentage will differ run to run; this file records one real measured run, not a "
                  "fixed target.",
        "results": {
            "cold_fetch_seconds": round(cold_seconds, 2),
            "warm_fetch_seconds": round(warm_seconds, 2),
            "reduction_pct": round(reduction_pct, 1),
        },
    }

    out_path = REPO_ROOT / "evaluation" / "results" / "fetch_dedup_benchmark.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReduction: {reduction_pct:.1f}%")
    print(f"Report written to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
