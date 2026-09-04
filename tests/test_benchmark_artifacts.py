"""Regression coverage for the artifact-writing paths of the CLI scripts
this session gated behind explicit flags: confirms they still actually run
end-to-end and produce a well-formed JSON artifact, not just that argparse
accepts the flag.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    # PYTHONIOENCODING=utf-8 avoids a real but unrelated fragility: rich's
    # box-drawing characters (e.g. benchmark.py's "=== Results ===" banner)
    # crash under Windows' legacy console codepage (cp1252) when stdout isn't
    # a real terminal, as it isn't here under subprocess capture. GitHub
    # Actions' ubuntu-latest runners default to UTF-8 and never hit this --
    # setting it explicitly keeps this test's own behavior identical on both,
    # rather than this test depending on which OS happens to run it.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, *args], cwd=cwd, capture_output=True, text=True, timeout=60, env=env,
        encoding="utf-8", errors="replace",
    )


class TestBenchmarkSimulateArtifact:
    def test_simulate_writes_tagged_json(self, tmp_path):
        output_path = tmp_path / "benchmark.simulated.json"
        result = _run(
            ["evaluation/benchmark.py", "--simulate", "--n-docs", "10", "--output", str(output_path)],
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, result.stderr
        assert output_path.exists()

        data = json.loads(output_path.read_text())
        assert data["is_simulated"] is True
        assert "latency" in data
        assert set(data["latency"].keys()) >= {"p50_ms", "p95_ms", "p99_ms"}

    def test_no_flags_refuses_to_run(self):
        result = _run(["evaluation/benchmark.py"], cwd=REPO_ROOT)
        assert result.returncode != 0
        assert "--simulate" in result.stderr

    def test_both_flags_refuses_to_run(self):
        result = _run(
            ["evaluation/benchmark.py", "--server", "http://x", "--simulate"], cwd=REPO_ROOT,
        )
        assert result.returncode != 0

    def test_default_output_path_is_distinctly_named(self):
        """No --output given: must land at results/benchmark.simulated.json,
        never results/benchmark.json -- a real run's filename. This exact
        gap (a Makefile target hardcoding --output to the "real" filename,
        silently reintroducing indistinguishable fabricated output) was
        found and fixed this session precisely because it wasn't covered
        by a test until now."""
        default_path = REPO_ROOT / "results" / "benchmark.simulated.json"
        real_named_path = REPO_ROOT / "results" / "benchmark.json"
        for p in (default_path, real_named_path):
            p.unlink(missing_ok=True)
        try:
            result = _run(["evaluation/benchmark.py", "--simulate", "--n-docs", "5"], cwd=REPO_ROOT)
            assert result.returncode == 0, result.stderr
            assert default_path.exists()
            assert not real_named_path.exists()
        finally:
            default_path.unlink(missing_ok=True)


class TestEvaluatePlaceholderArtifact:
    def test_generate_sample_metrics_writes_tagged_json(self, tmp_path):
        output_path = tmp_path / "metrics.json"
        result = _run(
            ["evaluation/evaluate.py", "--generate-sample-metrics", "--output", str(output_path)],
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(output_path.read_text())
        assert data["is_fabricated_placeholder"] is True

    def test_no_flags_refuses_to_run(self):
        result = _run(["evaluation/evaluate.py"], cwd=REPO_ROOT)
        assert result.returncode != 0
        assert "generate-sample-metrics" in result.stderr

    def test_default_output_path_is_distinctly_named(self):
        """Same class of gap as benchmark.py's equivalent test: no --output
        given must land at a name that can't be confused with a real run's
        results/metrics.json."""
        default_path = REPO_ROOT / "results" / "metrics.fabricated_placeholder.json"
        real_named_path = REPO_ROOT / "results" / "metrics.json"
        for p in (default_path, real_named_path):
            p.unlink(missing_ok=True)
        try:
            result = _run(["evaluation/evaluate.py", "--generate-sample-metrics"], cwd=REPO_ROOT)
            assert result.returncode == 0, result.stderr
            assert default_path.exists()
            assert not real_named_path.exists()
        finally:
            default_path.unlink(missing_ok=True)


class TestParserTelemetryReportArtifact:
    def test_writes_well_formed_report(self, tmp_path, monkeypatch):
        result = _run(["evaluation/parser_telemetry_report.py"], cwd=REPO_ROOT)
        assert result.returncode == 0, result.stderr

        out_path = REPO_ROOT / "evaluation" / "results" / "parser_telemetry_report.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())

        assert "malformed_corpus" in data
        assert "clean_output_baseline" in data
        assert data["malformed_corpus"]["n_cases"] > 0
        assert 0.0 <= data["malformed_corpus"]["recovery_rate"] <= 1.0


class TestMonitorDemoArtifact:
    def test_demo_writes_tagged_json(self, tmp_path):
        output_path = tmp_path / "report.json"
        result = _run(
            ["monitoring/monitor.py", "--demo", "--output", str(output_path)], cwd=REPO_ROOT,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(output_path.read_text())
        assert data["is_demo_data"] is True

    def test_no_flags_refuses_to_run(self):
        result = _run(["monitoring/monitor.py"], cwd=REPO_ROOT)
        assert result.returncode != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
