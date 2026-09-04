"""Docker Compose smoke test: API + Redis + PostgreSQL + a real extraction request.

Brings up docker/docker-compose.smoke.yml, waits for all three services to
report healthy, then fires a genuine HTTP request against a running
container -- not a mocked client, not an in-process TestClient.

Scope, stated plainly: with no GPU and no HF_TOKEN in this stack, the /extract
call cannot actually run Llama 3.1 inference. This test verifies the API
comes up wired to real Postgres and Redis containers and that /extract
returns a well-formed HTTP response (a graceful error/degraded status for the
LLM step is an ACCEPTABLE outcome here; a 5xx, a hang, or a crashed container
is not). Real LLM-inference evidence lives in evaluation/results/ and
notebooks/ (produced on Kaggle's GPU, where a GPU is actually available).

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --keep-up   # don't tear down after (debugging)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.smoke.yml"
API_BASE = "http://localhost:8000"

SAMPLE_FILING_TEXT = (Path(__file__).parent.parent / "data" / "sample_10k.txt")


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO_ROOT, **kwargs)


def compose(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return run(["docker", "compose", "-f", str(COMPOSE_FILE), *args], **kwargs)


def wait_for_healthy(timeout_s: int = 240) -> dict:
    """Poll `docker compose ps` for all services reporting healthy."""
    deadline = time.time() + timeout_s
    last_status = {}
    while time.time() < deadline:
        result = compose("ps", "--format", "json", capture_output=True, text=True)
        services = {}
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            services[entry["Service"]] = entry.get("Health", entry.get("State", "unknown"))
        last_status = services
        if services and all(v in ("healthy", "running") for v in services.values()):
            # api has no healthcheck of its own beyond the container's
            # HEALTHCHECK -- require it specifically to say healthy.
            if services.get("api") in ("healthy",) and services.get("postgres") == "healthy" \
                    and services.get("redis") == "healthy":
                return services
        time.sleep(3)
    raise TimeoutError(f"Services did not become healthy within {timeout_s}s: {last_status}")


def http_get(path: str, timeout: float = 10.0) -> tuple[int, dict | str]:
    try:
        with urllib.request.urlopen(f"{API_BASE}{path}", timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body


def http_post_json(path: str, payload: dict, timeout: float = 120.0) -> tuple[int, dict | str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}{path}", data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body


def main():
    parser = argparse.ArgumentParser(description="Docker Compose smoke test")
    parser.add_argument("--keep-up", action="store_true", help="Leave containers running after the test")
    parser.add_argument("--timeout", type=int, default=240, help="Seconds to wait for healthy services")
    args = parser.parse_args()

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "compose_file": str(COMPOSE_FILE.relative_to(REPO_ROOT)),
        "scope_note": (
            "No GPU, no HF_TOKEN in this stack -- LLM inference is expected to "
            "fail gracefully, not to succeed. Real GPU-backed inference evidence "
            "is in evaluation/results/ and notebooks/ (Kaggle T4)."
        ),
        "steps": [],
    }
    overall_ok = True

    def record(name: str, ok: bool, detail):
        nonlocal overall_ok
        overall_ok = overall_ok and ok
        report["steps"].append({"step": name, "ok": ok, "detail": detail})
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")

    build = compose("up", "-d", "--build", capture_output=True, text=True)
    record(
        "docker compose up -d --build",
        build.returncode == 0,
        build.stdout[-2000:] + build.stderr[-2000:],
    )

    if build.returncode == 0:
        try:
            services = wait_for_healthy(args.timeout)
            record("all services healthy", True, services)
        except TimeoutError as e:
            record("all services healthy", False, str(e))
            overall_ok = False

        if overall_ok:
            status, body = http_get("/health")
            record("GET /health", status == 200, {"status": status, "body": body})

            filing_text = SAMPLE_FILING_TEXT.read_text(encoding="utf-8")
            status, body = http_post_json(
                "/extract",
                {"filing_id": "smoke-test-001", "text": filing_text},
            )
            # A well-formed response (2xx-4xx JSON, not 5xx/hang/crash) is the
            # pass condition here -- see module docstring for why LLM failure
            # itself is not a smoke-test failure.
            well_formed = status < 500 and isinstance(body, dict)
            record(
                "POST /extract (real HTTP request, real containers)",
                well_formed,
                {"status": status, "body": body},
            )

    if not args.keep_up:
        down = compose("down", "-v", capture_output=True, text=True)
        record("docker compose down -v", down.returncode == 0, down.stdout[-1000:] + down.stderr[-1000:])

    out_dir = REPO_ROOT / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "docker_smoke_test.json"
    report["overall_ok"] = overall_ok
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\n{'PASSED' if overall_ok else 'FAILED'}")
    print(f"Report written to {out_path.relative_to(REPO_ROOT)}")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
