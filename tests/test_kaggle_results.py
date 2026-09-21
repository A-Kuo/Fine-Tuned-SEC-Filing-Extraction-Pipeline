"""Tests for scripts/fetch_kaggle_results.py's pure metrics-conversion logic.

Only build_metrics_summary()/_downsample() are exercised -- the download/glue
layer talks to the live Kaggle API and is intentionally out of scope here
(no live network/credentials in this test suite, matching tests/test_database.py's
mock-only convention).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.fetch_kaggle_results as fetch
from scripts.fetch_kaggle_results import (
    build_metrics_summary,
    _downsample,
    assert_real_training_occurred,
    collect_run_extras,
)


def _log(n: int) -> list[dict]:
    return [{"step": i, "epoch": i / 10, "loss": 2.0 - i * 0.01} for i in range(n)]


class TestBuildMetricsSummaryNormalCase:
    def test_final_train_loss_from_hf_metrics(self):
        summary = build_metrics_summary(
            {"train_loss": 0.42, "epoch": 3.0, "train_runtime": 500.0},
            _log(100),
            status="complete", kernel_slug="u/s", git_commit_sha="abc123",
        )
        assert summary["final_train_loss"] == 0.42
        assert summary["final_epoch"] == 3.0
        assert summary["total_steps"] == 99

    def test_duration_from_runtime_when_no_timestamps(self):
        summary = build_metrics_summary(
            {"train_runtime": 500.0}, [],
            status="complete", kernel_slug="u/s", git_commit_sha="abc123",
        )
        assert summary["duration_s"] == 500.0

    def test_duration_from_timestamps_when_present(self):
        summary = build_metrics_summary(
            {"train_runtime": 999.0}, [],
            status="complete", kernel_slug="u/s", git_commit_sha="abc123",
            started_at="2026-08-26T00:00:00Z", completed_at="2026-08-26T01:00:00Z",
        )
        assert summary["duration_s"] == pytest.approx(3600.0)

    def test_loss_curve_downsampled_and_capped(self):
        summary = build_metrics_summary(
            {}, _log(500), status="complete", kernel_slug="u/s", git_commit_sha="abc123",
        )
        assert len(summary["loss_curve"]) <= 50
        assert summary["loss_curve"][0]["step"] == 0
        assert summary["loss_curve"][-1]["step"] == 499

    def test_raw_hf_metrics_passthrough(self):
        raw = {"train_loss": 0.1, "some_unmapped_field": "value"}
        summary = build_metrics_summary(
            raw, [], status="complete", kernel_slug="u/s", git_commit_sha="abc123",
        )
        assert summary["raw_hf_metrics"] == raw

    def test_schema_version_and_identity_fields(self):
        summary = build_metrics_summary(
            {}, [], status="complete", kernel_slug="user/slug", git_commit_sha="deadbeef",
            kernel_version=7,
        )
        assert summary["schema_version"] == 1
        assert summary["kernel_slug"] == "user/slug"
        assert summary["git_commit_sha"] == "deadbeef"
        assert summary["kernel_version"] == 7


class TestBuildMetricsSummaryMissingKeyRobustness:
    def test_missing_train_loss_falls_back_to_last_log_entry(self):
        summary = build_metrics_summary(
            {"epoch": 2.0},  # no train_loss key -- simulates a transformers version drift
            _log(10),
            status="complete", kernel_slug="u/s", git_commit_sha="abc123",
        )
        assert summary["final_train_loss"] == pytest.approx(2.0 - 9 * 0.01)

    def test_completely_empty_training_metrics_does_not_raise(self):
        summary = build_metrics_summary(
            {}, [], status="complete", kernel_slug="u/s", git_commit_sha="abc123",
        )
        assert summary["final_train_loss"] is None
        assert summary["final_epoch"] is None
        assert summary["duration_s"] is None
        assert summary["raw_hf_metrics"] == {}

    def test_training_log_entries_without_loss_key_are_excluded(self):
        log = [{"step": 0, "epoch": 0.0}, {"step": 1, "epoch": 0.1, "loss": 1.5}]
        summary = build_metrics_summary(
            {}, log, status="complete", kernel_slug="u/s", git_commit_sha="abc123",
        )
        assert len(summary["loss_curve"]) == 1
        assert summary["loss_curve"][0]["loss"] == 1.5


class TestBuildMetricsSummaryEmptyLog:
    def test_empty_training_log_yields_empty_loss_curve(self):
        summary = build_metrics_summary(
            {"train_loss": 0.9}, [], status="complete", kernel_slug="u/s", git_commit_sha="abc123",
        )
        assert summary["loss_curve"] == []
        assert summary["total_steps"] is None


class TestStatusPassthrough:
    @pytest.mark.parametrize("status", ["complete", "error", "timeout"])
    def test_status_preserved_verbatim(self, status):
        summary = build_metrics_summary(
            {}, [], status=status, kernel_slug="u/s", git_commit_sha="abc123",
        )
        assert summary["status"] == status


class TestDownsample:
    def test_no_downsampling_needed(self):
        points = [{"step": i} for i in range(10)]
        assert _downsample(points, 50) == points

    def test_downsampling_includes_first_and_last(self):
        points = [{"step": i} for i in range(1000)]
        result = _downsample(points, 50)
        assert len(result) <= 50
        assert result[0]["step"] == 0
        assert result[-1]["step"] == 999

    def test_max_points_of_two(self):
        points = [{"step": i} for i in range(100)]
        result = _downsample(points, 2)
        assert result[0]["step"] == 0
        assert result[-1]["step"] == 99

    def test_empty_points(self):
        assert _downsample([], 50) == []


class TestAssertRealTrainingOccurred:
    """Regression coverage for the silent-fake-success bug: a crashed Kaggle
    kernel (e.g. missing training data) used to produce an all-null
    metrics.json and exit 0, letting kaggle_training.yml's commit step land
    a hollow result under a green checkmark."""

    def test_raises_systemexit_when_note_present(self):
        summary = {"status": "complete", "note": "training_metrics.json/training_log.json not found in kernel output"}
        with pytest.raises(SystemExit) as exc_info:
            assert_real_training_occurred(summary)
        assert exc_info.value.code == 1

    def test_no_exit_when_real_metrics_present(self):
        summary = build_metrics_summary(
            {"train_loss": 0.42, "train_runtime": 500.0}, _log(67),
            status="complete", kernel_slug="u/s", git_commit_sha="abc123",
        )
        assert "note" not in summary
        assert_real_training_occurred(summary)  # must not raise


class TestExtras:
    def test_extras_are_merged_as_top_level_keys(self):
        summary = build_metrics_summary(
            {"train_loss": 0.4}, _log(10),
            status="complete", kernel_slug="u/s", git_commit_sha="abc123",
            extras={"adapter_eval": {"n_examples": 30}, "run_info": {"train_examples": 90}},
        )
        assert summary["adapter_eval"] == {"n_examples": 30}
        assert summary["run_info"] == {"train_examples": 90}
        assert summary["final_train_loss"] == 0.4

    def test_no_extras_leaves_the_original_schema_unchanged(self):
        summary = build_metrics_summary(
            {}, [], status="complete", kernel_slug="u/s", git_commit_sha="abc123",
        )
        assert "adapter_eval" not in summary and "run_info" not in summary


class TestCollectRunExtras:
    def _write(self, root, relative, payload):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    def test_reports_are_found_wherever_the_kernel_put_them(self, tmp_path):
        self._write(tmp_path, "repo/models/llama-sec-v1/training_run_info.json", {"train_examples": 90})
        self._write(tmp_path, "reports/run_report.json", {"mode": "full"})
        self._write(tmp_path, "reports/adapter_eval.json", {"n_examples": 30})
        self._write(tmp_path, "reports/hf_upload.json", {"status": "uploaded"})
        self._write(tmp_path, "reports/environment.json", {
            "python": "3.12.1", "git_commit": "abc", "packages": {"trl": "0.12.2"},
            "gpu": {"name": "Tesla T4"}, "data_files": {}, "collected_at": "2026-09-20T00:00:00Z",
        })

        extras = collect_run_extras(tmp_path)

        assert extras["run_info"] == {"train_examples": 90}
        assert extras["run_report"] == {"mode": "full"}
        assert extras["adapter_eval"] == {"n_examples": 30}
        assert extras["hf_upload"] == {"status": "uploaded"}
        assert extras["environment"]["packages"] == {"trl": "0.12.2"}
        assert "collected_at" not in extras["environment"]

    def test_missing_reports_are_simply_absent(self, tmp_path):
        self._write(tmp_path, "reports/run_report.json", {"mode": "smoke"})
        assert set(collect_run_extras(tmp_path)) == {"run_report"}


class TestLogsOnly:
    """The workflow runs this even after a failed kernel, so it must never
    write results or exit non-zero."""

    def _run(self, monkeypatch, tmp_path, api):
        monkeypatch.setattr(fetch, "DOWNLOAD_DIR", tmp_path / "out")
        monkeypatch.setattr(fetch, "RESULTS_DIR", tmp_path / "results")
        monkeypatch.setattr(fetch, "HISTORY_DIR", tmp_path / "results" / "history")
        monkeypatch.setattr(fetch, "_get_kaggle_api", lambda: api)
        monkeypatch.setattr(sys, "argv", ["fetch_kaggle_results.py", "--logs-only", "--kernel-slug", "u/s"])
        fetch.main()

    def test_downloads_output_and_writes_no_results(self, monkeypatch, tmp_path):
        calls = []

        class Api:
            def kernels_output(self, slug, path, force):
                calls.append((slug, path, force))

        self._run(monkeypatch, tmp_path, Api())

        assert calls == [("u/s", str(tmp_path / "out"), True)]
        assert not (tmp_path / "results").exists()

    def test_a_download_failure_is_a_warning_not_an_exit(self, monkeypatch, tmp_path):
        class Api:
            def kernels_output(self, slug, path, force):
                raise RuntimeError("kernel never started")

        self._run(monkeypatch, tmp_path, Api())  # must not raise

    def test_a_kaggle_auth_exit_is_also_swallowed(self, monkeypatch, tmp_path):
        def failing_api():
            raise SystemExit("Kaggle authentication failed")

        monkeypatch.setattr(fetch, "DOWNLOAD_DIR", tmp_path / "out")
        monkeypatch.setattr(fetch, "_get_kaggle_api", failing_api)
        monkeypatch.setattr(sys, "argv", ["fetch_kaggle_results.py", "--logs-only", "--kernel-slug", "u/s"])
        fetch.main()  # must not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
