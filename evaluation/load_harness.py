"""Concurrency/load-testing harness for the serving API.

evaluation/benchmark.py's --server (live) mode is strictly sequential
(`for i in range(0, n_docs, batch_size): await client.post(...)`) -- it never
actually issues concurrent requests, so it cannot measure throughput under
load, cold-start cost, or degraded-path behavior. This harness does, using
asyncio.gather (no new heavy dependency like locust/k6 -- stays consistent
with benchmark.py's own httpx-based approach).

Needs a running server (docker/docker-compose.smoke.yml or `python -m
serving.api` locally) -- cannot execute meaningfully without Docker/a live
process in this environment; the harness itself is real, tested code.

Usage:
    python evaluation/load_harness.py --server http://localhost:8000 --concurrency 10 --requests 50
    python evaluation/load_harness.py --server http://localhost:8000 --mode batch --concurrency 5 --requests 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def percentiles(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None}
    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)
    return {
        "p50_ms": round(sorted_lat[n // 2], 1),
        "p95_ms": round(sorted_lat[min(int(n * 0.95), n - 1)], 1),
        "p99_ms": round(sorted_lat[min(int(n * 0.99), n - 1)], 1),
        "min_ms": round(sorted_lat[0], 1),
        "max_ms": round(sorted_lat[-1], 1),
    }


async def _timed_request(client, url: str, payload: dict) -> dict:
    start = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, timeout=60)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"ok": resp.status_code == 200, "status_code": resp.status_code, "latency_ms": elapsed_ms}
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"ok": False, "error": str(e), "latency_ms": elapsed_ms}


async def measure_cold_start(server_url: str, sample_text: str, timeout_s: float = 120.0) -> dict | None:
    """Time-to-first-successful-response after the process is already up but
    before any /extract request has been served -- distinct from process
    startup time, which this harness cannot observe from outside."""
    import httpx

    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{server_url}/extract", json={"text": sample_text}, timeout=timeout_s,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {"ok": resp.status_code == 200, "cold_start_ms": round(elapsed_ms, 1)}
        except Exception as e:
            return {"ok": False, "error": str(e)}


async def run_concurrent_load(
    server_url: str,
    sample_text: str,
    concurrency: int,
    n_requests: int,
    mode: str = "single",
) -> dict:
    """Fires n_requests total, `concurrency` in flight at a time, and
    measures real throughput/latency/error-rate -- not the sequential
    approximation benchmark.py's live mode gives."""
    import httpx

    endpoint = "/extract" if mode == "single" else "/extract/batch"
    payload = (
        {"text": sample_text}
        if mode == "single"
        else {"documents": [{"text": sample_text}] * 4}
    )

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(client):
        async with semaphore:
            return await _timed_request(client, f"{server_url}{endpoint}", payload)

    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(bounded_request(client) for _ in range(n_requests)))
    total_s = time.perf_counter() - start

    latencies = [r["latency_ms"] for r in results]
    n_ok = sum(1 for r in results if r["ok"])

    return {
        "mode": mode,
        "concurrency": concurrency,
        "n_requests": n_requests,
        "n_successful": n_ok,
        "n_errors": n_requests - n_ok,
        "error_rate": round((n_requests - n_ok) / n_requests, 4) if n_requests else None,
        "total_time_s": round(total_s, 2),
        "throughput_req_per_sec": round(n_requests / total_s, 2) if total_s > 0 else None,
        "latency": percentiles(latencies),
    }


async def measure_degraded_path(server_url: str, sample_text: str) -> dict:
    """Fires one request and reports whether the response is well-formed
    regardless of backend health -- this harness cannot itself kill Redis
    (needs docker-compose control from outside), so it documents what a
    degraded-path test run should check rather than fabricating a
    before/after comparison it can't actually produce here."""
    import httpx

    async with httpx.AsyncClient() as client:
        result = await _timed_request(client, f"{server_url}/health", {})
    return {
        "note": (
            "This checks /health responds; a real degraded-path test additionally "
            "requires stopping Redis via `docker compose stop redis` mid-run and "
            "re-measuring latency/error-rate with run_concurrent_load() -- do that "
            "manually alongside this harness, see docs/NEXT_EXPERIMENTS.md."
        ),
        "health_check": result,
    }


async def run_full_harness(server_url: str, concurrency: int, n_requests: int, mode: str) -> dict:
    sample_path = REPO_ROOT / "data" / "sample_10k.txt"
    sample_text = sample_path.read_text(encoding="utf-8") if sample_path.exists() else "SEC FILING TEXT"

    cold_start = await measure_cold_start(server_url, sample_text)
    load_result = await run_concurrent_load(server_url, sample_text, concurrency, n_requests, mode)
    degraded = await measure_degraded_path(server_url, sample_text)

    return {
        "server_url": server_url,
        "cold_start": cold_start,
        "load": load_result,
        "degraded_path": degraded,
    }


def main():
    parser = argparse.ArgumentParser(description="Concurrency/load test the serving API")
    parser.add_argument("--server", required=True, help="Base URL, e.g. http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--mode", choices=["single", "batch"], default="single")
    parser.add_argument("--output", default="evaluation/results/load_harness_report.json")
    args = parser.parse_args()

    results = asyncio.run(run_full_harness(args.server, args.concurrency, args.requests, args.mode))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Cold start: {results['cold_start']}")
    print(f"Load ({args.mode}, concurrency={args.concurrency}, n={args.requests}):")
    for k, v in results["load"].items():
        print(f"  {k}: {v}")
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
