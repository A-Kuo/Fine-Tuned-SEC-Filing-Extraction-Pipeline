"""Tests for scripts/kaggle_kernel/train_kernel.py.

Regression coverage for the bug where the Kaggle kernel ran
training/train.py directly against a training data file that was never
generated (config.yaml's kaggle.dataset_id is empty, and its
local_fallback, data/sec_filings_train.jsonl, is gitignored -- absent from
this script's fresh git clone) -- training crashed with FileNotFoundError
almost immediately -- plus the run modes, TRL pin, and the rule that steps
after training can never fail a run whose training succeeded. Mocks
subprocess.run entirely; no real clone/install/train happens in this test.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.kaggle_kernel.train_kernel as kernel
from scripts.kaggle_kernel.train_kernel import main


@pytest.fixture(autouse=True)
def report_dir(tmp_path, monkeypatch):
    """Keep run_report.json out of the real /kaggle/working path."""
    path = tmp_path / "reports"
    monkeypatch.setattr(kernel, "REPORT_DIR", str(path))
    return path


def _run_main(mode=None, fail_when=None):
    """Run main() with subprocess mocked. fail_when(cmd) -> True makes that
    command raise CalledProcessError. Returns the list of commands run."""
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        assert kwargs.get("check") is True, f"subprocess without check=True: {cmd}"
        if fail_when and fail_when(cmd):
            raise subprocess.CalledProcessError(1, cmd)
        return MagicMock(returncode=0)

    with patch("scripts.kaggle_kernel.train_kernel.subprocess.run", side_effect=fake_run), \
         patch("scripts.kaggle_kernel.train_kernel._load_kaggle_secrets"):
        main(mode)
    return commands


def _index(commands, fragment):
    return next(i for i, cmd in enumerate(commands) if fragment in " ".join(map(str, cmd)))


def _has(commands, fragment):
    return any(fragment in " ".join(map(str, cmd)) for cmd in commands)


class TestDataGenerationOrder:
    def test_download_and_format_run_before_training(self):
        commands = _run_main()

        assert any("git" in cmd[0] and "clone" in cmd for cmd in commands)
        assert _index(commands, "scripts/download_dataset.py") < _index(commands, "scripts/format_data.py") < _index(commands, "training/train.py")

    def test_all_subprocess_calls_use_check_true(self):
        """A silent data-generation failure must not be allowed to fall
        through to training/train.py's own (uninformative) crash. _run_main
        asserts check=True on every call."""
        _run_main()


class TestEnvironmentPin:
    def test_trl_is_pinned_after_requirements_and_before_training(self):
        commands = _run_main()

        requirements = _index(commands, "-r requirements.txt")
        pin = _index(commands, "trl==0.12.2")
        assert requirements < pin < _index(commands, "training/train.py")

    def test_environment_is_recorded_before_training(self):
        commands = _run_main()
        assert _index(commands, "collect_environment.py") < _index(commands, "training/train.py")


class TestRunModes:
    def test_smoke_trains_briefly_evaluates_two_examples_and_never_uploads(self):
        commands = _run_main("smoke")

        train = commands[_index(commands, "training/train.py")]
        assert train[-4:] == ["--max_samples", "8", "--num_epochs", "1"]
        evaluation = commands[_index(commands, "evaluation/eval_adapter.py")]
        assert evaluation[evaluation.index("--extra") + 1] == "0"
        assert evaluation[evaluation.index("--limit") + 1] == "2"
        assert not _has(commands, "upload_adapter_hf.py")

    def test_full_trains_on_everything_evaluates_and_uploads(self):
        commands = _run_main("full")

        train = commands[_index(commands, "training/train.py")]
        assert "--max_samples" not in train and "--num_epochs" not in train
        evaluation = commands[_index(commands, "evaluation/eval_adapter.py")]
        assert "--limit" not in evaluation
        assert _index(commands, "training/train.py") < _index(commands, "eval_adapter.py") < _index(commands, "upload_adapter_hf.py")

    def test_default_mode_is_the_module_constant(self):
        assert kernel.RUN_MODE in kernel.MODES

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(SystemExit, match="Unknown RUN_MODE"):
            _run_main("turbo")


class TestFailureSemantics:
    def test_a_failed_evaluation_does_not_fail_a_successful_training_run(self, report_dir):
        _run_main("full", fail_when=lambda cmd: "eval_adapter.py" in " ".join(map(str, cmd)))

        report = json.loads((report_dir / "run_report.json").read_text())
        assert report["steps"]["training"]["status"] == "ok"
        assert report["steps"]["evaluation"]["status"] == "failed"

    def test_a_failed_upload_does_not_fail_a_successful_training_run(self, report_dir):
        _run_main("full", fail_when=lambda cmd: "upload_adapter_hf.py" in " ".join(map(str, cmd)))

        report = json.loads((report_dir / "run_report.json").read_text())
        assert report["steps"]["training"]["status"] == "ok"
        assert report["steps"]["hf_upload"]["status"] == "failed"

    def test_a_failed_environment_capture_does_not_stop_training(self, report_dir):
        commands = _run_main(fail_when=lambda cmd: "collect_environment.py" in " ".join(map(str, cmd)))
        assert _has(commands, "training/train.py")

    def test_a_failed_training_step_fails_the_kernel_and_is_reported(self, report_dir):
        with pytest.raises(subprocess.CalledProcessError):
            _run_main(fail_when=lambda cmd: "training/train.py" in " ".join(map(str, cmd)))

        report = json.loads((report_dir / "run_report.json").read_text())
        assert report["steps"]["training"]["status"] == "failed"

    def test_evaluation_and_upload_never_run_after_failed_training(self):
        commands = []

        def fake_run(cmd, **kwargs):
            commands.append(cmd)
            if "training/train.py" in " ".join(map(str, cmd)):
                raise subprocess.CalledProcessError(1, cmd)
            return MagicMock(returncode=0)

        with patch("scripts.kaggle_kernel.train_kernel.subprocess.run", side_effect=fake_run), \
             patch("scripts.kaggle_kernel.train_kernel._load_kaggle_secrets"), \
             pytest.raises(subprocess.CalledProcessError):
            main("full")

        assert not _has(commands, "eval_adapter.py")
        assert not _has(commands, "upload_adapter_hf.py")

    def test_a_setup_failure_is_labelled_setup_not_training(self, report_dir):
        with pytest.raises(subprocess.CalledProcessError):
            _run_main(fail_when=lambda cmd: "requirements.txt" in " ".join(map(str, cmd)))

        report = json.loads((report_dir / "run_report.json").read_text())
        assert report["steps"]["setup"]["status"] == "failed"
        assert "training" not in report["steps"]


class TestRunReport:
    def test_report_records_mode_and_every_stage(self, report_dir):
        _run_main("smoke")

        report = json.loads((report_dir / "run_report.json").read_text())
        assert report["mode"] == "smoke"
        assert report["steps"]["training"]["status"] == "ok"
        assert report["steps"]["evaluation"]["status"] == "ok"
        assert report["steps"]["hf_upload"] == {"status": "skipped", "reason": "smoke mode never uploads"}
        assert "started_at" in report and "finished_at" in report


class TestSecrets:
    def test_available_secrets_are_loaded_and_missing_ones_are_ignored(self, monkeypatch):
        for name in kernel.SECRET_NAMES:
            monkeypatch.delenv(name, raising=False)

        class FakeClient:
            def get_secret(self, name):
                if name == "HF_TOKEN":
                    return "hf_read"
                if name == "HF_ADAPTER_REPO":
                    return "me/adapter"
                raise RuntimeError("secret not found")

        fake_module = ModuleType("kaggle_secrets")
        fake_module.UserSecretsClient = FakeClient
        monkeypatch.setitem(sys.modules, "kaggle_secrets", fake_module)

        kernel._load_kaggle_secrets()

        import os

        assert os.environ["HF_TOKEN"] == "hf_read"
        assert os.environ["HF_ADAPTER_REPO"] == "me/adapter"
        assert "HF_WRITE_TOKEN" not in os.environ
        assert "MLFLOW_TRACKING_URI" not in os.environ

    def test_every_documented_secret_is_requested(self):
        assert {"HF_TOKEN", "HF_WRITE_TOKEN", "HF_ADAPTER_REPO", "MLFLOW_TRACKING_URI"} <= set(kernel.SECRET_NAMES)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
