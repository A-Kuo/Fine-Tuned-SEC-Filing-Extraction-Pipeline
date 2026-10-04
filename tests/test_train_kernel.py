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

# The real functions, kept so the tests that exercise them directly are not
# affected by the autouse stub below.
REAL_LOAD_SECRETS = kernel._load_kaggle_secrets
REAL_PREFLIGHT = kernel._preflight_hf_access


@pytest.fixture(autouse=True)
def report_dir(tmp_path, monkeypatch):
    """Keep run_report.json out of the real /kaggle/working path."""
    path = tmp_path / "reports"
    monkeypatch.setattr(kernel, "REPORT_DIR", str(path))
    return path


@pytest.fixture(autouse=True)
def no_real_secrets_or_network(monkeypatch):
    """main() would otherwise read Kaggle secrets and call Hugging Face."""
    monkeypatch.setattr(kernel, "_load_kaggle_secrets", lambda: {})
    monkeypatch.setattr(kernel, "_preflight_hf_access", lambda *a, **k: (True, "stubbed"))


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

    with patch("scripts.kaggle_kernel.train_kernel.subprocess.run", side_effect=fake_run):
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

        REAL_LOAD_SECRETS()

        import os

        assert os.environ["HF_TOKEN"] == "hf_read"
        assert os.environ["HF_ADAPTER_REPO"] == "me/adapter"
        assert "HF_WRITE_TOKEN" not in os.environ
        assert "MLFLOW_TRACKING_URI" not in os.environ

    def test_every_documented_secret_is_requested(self):
        assert {"HF_TOKEN", "HF_WRITE_TOKEN", "HF_ADAPTER_REPO", "MLFLOW_TRACKING_URI"} <= set(kernel.SECRET_NAMES)


class TestSecretReporting:
    """A secret that isn't attached to the notebook used to look identical to
    one that loaded fine -- the first symptom was a 401 minutes later."""

    def _install_client(self, monkeypatch, behaviour):
        class FakeClient:
            def get_secret(self, name):
                return behaviour(name)

        fake_module = ModuleType("kaggle_secrets")
        fake_module.UserSecretsClient = FakeClient
        monkeypatch.setitem(sys.modules, "kaggle_secrets", fake_module)

    def test_reports_loaded_and_not_loaded_per_secret_name(self, monkeypatch):
        for name in kernel.SECRET_NAMES:
            monkeypatch.delenv(name, raising=False)

        def behaviour(name):
            if name == "HF_TOKEN":
                return "hf_the_real_secret_value"
            raise RuntimeError("Secret not attached to this notebook")

        self._install_client(monkeypatch, behaviour)

        status = REAL_LOAD_SECRETS()

        assert status["HF_TOKEN"] == "loaded"
        assert status["HF_WRITE_TOKEN"].startswith("not loaded (RuntimeError")
        assert "not attached" in status["HF_WRITE_TOKEN"]

    def test_secret_values_are_never_printed_or_returned(self, monkeypatch, capsys):
        for name in kernel.SECRET_NAMES:
            monkeypatch.delenv(name, raising=False)
        self._install_client(monkeypatch, lambda name: "hf_the_real_secret_value")

        status = REAL_LOAD_SECRETS()

        output = capsys.readouterr().out
        assert "hf_the_real_secret_value" not in output
        assert "hf_the_real_secret_value" not in json.dumps(status)
        assert "[kernel] secret HF_TOKEN: loaded" in output

    def test_a_client_that_cannot_even_be_constructed_is_reported_for_every_secret(self, monkeypatch):
        class BrokenClient:
            def __init__(self):
                raise PermissionError("no secrets service")

        fake_module = ModuleType("kaggle_secrets")
        fake_module.UserSecretsClient = BrokenClient
        monkeypatch.setitem(sys.modules, "kaggle_secrets", fake_module)

        status = REAL_LOAD_SECRETS()

        assert set(status) == set(kernel.SECRET_NAMES)
        assert all(v.startswith("not loaded (PermissionError") for v in status.values())

    def test_off_kaggle_it_reports_what_the_environment_already_has(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "kaggle_secrets", None)  # `import` raises ImportError
        monkeypatch.setenv("HF_TOKEN", "x")
        monkeypatch.delenv("HF_WRITE_TOKEN", raising=False)

        status = REAL_LOAD_SECRETS()

        assert status["HF_TOKEN"] == "present in environment"
        assert status["HF_WRITE_TOKEN"] == "not set"


class FakeHttpError(Exception):
    def __init__(self, status_code, message="http error"):
        super().__init__(message)
        self.response = type("Resp", (), {"status_code": status_code})()


class GatedRepoError(FakeHttpError):
    pass


def _install_hf(monkeypatch, whoami=None, download=None):
    """A fake huggingface_hub whose whoami()/hf_hub_download() behave as given."""
    calls = {"download": []}

    class FakeHfApi:
        def __init__(self, token=None):
            self.token = token

        def whoami(self):
            if isinstance(whoami, Exception):
                raise whoami
            return whoami if whoami is not None else {"name": "austin"}

    def fake_download(repo_id, filename, token=None):
        calls["download"].append((repo_id, filename))
        if isinstance(download, Exception):
            raise download
        return "/cache/config.json"

    module = ModuleType("huggingface_hub")
    module.HfApi = FakeHfApi
    module.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


class TestPreflightHfAccess:
    def test_no_token_fails_with_the_attach_to_notebook_hint(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)

        ok, message = REAL_PREFLIGHT()

        assert ok is False
        assert "HF_TOKEN is not set" in message
        assert "Add-ons > Secrets" in message
        assert "attached" in message

    def test_a_token_hugging_face_rejects_is_reported_as_invalid(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_bad")
        _install_hf(monkeypatch, whoami=FakeHttpError(401))

        ok, message = REAL_PREFLIGHT()

        assert ok is False
        assert "rejected by Hugging Face" in message
        assert "settings/tokens" in message

    def test_a_valid_token_without_model_access_names_the_account_and_the_license_page(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_ok")
        _install_hf(monkeypatch, whoami={"name": "austin"}, download=GatedRepoError(401))

        ok, message = REAL_PREFLIGHT()

        assert ok is False
        assert "valid (Hugging Face account 'austin')" in message
        assert "https://huggingface.co/meta-llama/Llama-3.1-8B" in message
        assert "fine-grained" in message

    def test_access_ok_is_reported_with_the_account_name(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_ok")
        calls = _install_hf(monkeypatch, whoami={"name": "austin"})

        ok, message = REAL_PREFLIGHT()

        assert ok is True
        assert "access OK" in message and "austin" in message
        assert calls["download"] == [("meta-llama/Llama-3.1-8B", "config.json")]

    def test_the_token_value_never_appears_in_any_message(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_super_secret_value")
        _install_hf(monkeypatch, whoami={"name": "austin"}, download=GatedRepoError(403))

        _, message = REAL_PREFLIGHT()

        assert "hf_super_secret_value" not in message

    def test_a_network_problem_does_not_block_the_run(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_ok")
        _install_hf(monkeypatch, whoami=ConnectionError("dns failure"))

        ok, message = REAL_PREFLIGHT()

        assert ok is True
        assert "could not verify" in message

    def test_a_network_problem_during_the_model_check_does_not_block_either(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_ok")
        _install_hf(monkeypatch, whoami={"name": "austin"}, download=TimeoutError("slow"))

        ok, message = REAL_PREFLIGHT()

        assert ok is True
        assert "could not check access" in message

    def test_a_missing_huggingface_hub_does_not_block_the_run(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_ok")
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)

        ok, message = REAL_PREFLIGHT()

        assert ok is True
        assert "not importable" in message


class TestPreflightGatesTheRun:
    def test_a_failed_preflight_stops_before_cloning_or_installing_anything(self, monkeypatch, report_dir):
        monkeypatch.setattr(kernel, "_preflight_hf_access", lambda *a, **k: (False, "HF_TOKEN is not set."))
        commands = []

        def fake_run(cmd, **kwargs):
            commands.append(cmd)
            return MagicMock(returncode=0)

        with patch("scripts.kaggle_kernel.train_kernel.subprocess.run", side_effect=fake_run), \
             pytest.raises(SystemExit, match="Preflight failed: HF_TOKEN is not set"):
            main("full")

        assert commands == []  # no git clone, no pip install: it failed in seconds, not minutes

    def test_a_failed_preflight_is_recorded_in_the_run_report(self, monkeypatch, report_dir):
        monkeypatch.setattr(kernel, "_load_kaggle_secrets", lambda: {"HF_TOKEN": "not loaded (RuntimeError)"})
        monkeypatch.setattr(kernel, "_preflight_hf_access", lambda *a, **k: (False, "no access"))

        with patch("scripts.kaggle_kernel.train_kernel.subprocess.run"), pytest.raises(SystemExit):
            main("smoke")

        report = json.loads((report_dir / "run_report.json").read_text())
        assert report["steps"]["preflight"] == {"status": "failed", "detail": "no access"}
        assert report["secrets"] == {"HF_TOKEN": "not loaded (RuntimeError)"}
        assert "training" not in report["steps"]

    def test_a_passing_preflight_lets_the_run_continue_and_is_recorded(self, report_dir):
        commands = _run_main("smoke")

        assert _has(commands, "training/train.py")
        report = json.loads((report_dir / "run_report.json").read_text())
        assert report["steps"]["preflight"] == {"status": "ok", "detail": "stubbed"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
