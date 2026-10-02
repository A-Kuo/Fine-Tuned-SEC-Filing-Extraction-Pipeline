"""Tests for training/train.py's MLFlow handling and run provenance.

Regression coverage for two ways MLFlow used to sink a training run: with no
DagsHub token the code still pointed MLFlow at the remote DagsHub server,
unauthenticated, so mlflow.start_run() failed before any training step; and
log_artifacts/register_model ran after the adapter was saved but could still
crash the process and mark a good run failed.

training/train.py imports torch/trl/peft/mlflow at import time, none of which
this test environment needs. They are replaced with small fakes for the
duration of this module (the same technique tests/test_inference.py uses), so
this exercises train.py's own logic without a GPU or any downloads.
"""

import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class FakeMlflow(ModuleType):
    """Records every call; individual methods can be made to fail."""

    def __init__(self):
        super().__init__("mlflow")
        self.calls: list[tuple] = []
        self.fail: set[str] = set()
        self._uri = "file:///default"

    def _record(self, _method, *args, **kwargs):
        self.calls.append((_method, args, kwargs))
        if _method in self.fail:
            raise RuntimeError(f"{_method} unavailable")

    def set_tracking_uri(self, uri):
        self._uri = uri
        self._record("set_tracking_uri", uri)

    def get_tracking_uri(self):
        return self._uri

    def set_experiment(self, name):
        self._record("set_experiment", name)

    def log_params(self, params):
        self._record("log_params", params)

    def log_metrics(self, metrics):
        self._record("log_metrics", metrics)

    def log_artifacts(self, path, artifact_path=None):
        self._record("log_artifacts", path, artifact_path=artifact_path)

    def register_model(self, uri, name):
        self._record("register_model", uri, name=name)

    @contextmanager
    def start_run(self, run_name=None):
        self._record("start_run", run_name=run_name)
        yield SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

    def called(self, name) -> bool:
        return any(c[0] == name for c in self.calls)


def _fake_torch(cuda: bool = False):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            get_device_name=lambda i: "Tesla T4",
            max_memory_allocated=lambda: 11_500_000_000,
        ),
    )


@pytest.fixture(scope="module")
def train_module():
    fake_mlflow = FakeMlflow()
    fakes = {
        "mlflow": fake_mlflow,
        "torch": _fake_torch(),
        "datasets": SimpleNamespace(load_dataset=lambda *a, **k: None),
        "peft": SimpleNamespace(LoraConfig=object, TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
                                get_peft_model=lambda m, c: m, prepare_model_for_kbit_training=lambda m: m),
        "transformers": SimpleNamespace(AutoModelForCausalLM=object, AutoTokenizer=object,
                                        BitsAndBytesConfig=object, TrainingArguments=object),
        "trl": SimpleNamespace(SFTConfig=object, SFTTrainer=object),
        "training.callbacks": SimpleNamespace(MetricsCallback=object, EarlyStoppingOnLoss=object),
        "training.data_collator": SimpleNamespace(FinancialDataCollator=object),
    }
    with patch.dict(sys.modules, fakes):
        sys.modules.pop("training.train", None)
        import training.train as module

        yield module


@pytest.fixture
def mlflow_fake(train_module):
    fake = FakeMlflow()
    with patch.object(train_module, "mlflow", fake):
        yield fake


def _config(**mlflow_overrides):
    mlflow_cfg = {
        "tracking_uri": "",
        "experiment_name": "exp",
        "run_name_prefix": "test-run",
        "registered_model_name": "model-x",
        "register_model": False,
        "dagshub_repo_owner": "owner",
        "dagshub_repo_name": "repo",
        **mlflow_overrides,
    }
    return {"mlflow": mlflow_cfg}


class TestConfigureMlflow:
    def test_no_uri_and_no_token_uses_a_local_store_never_dagshub(self, train_module, mlflow_fake, tmp_path, monkeypatch):
        monkeypatch.delenv("DAGSHUB_USER_TOKEN", raising=False)

        out_dir = tmp_path / "nested" / "output"  # does not exist yet
        info = train_module.configure_mlflow(_config(), str(out_dir))

        assert info.kind == "local"
        assert not info.registry_capable
        uri = mlflow_fake.get_tracking_uri()
        # Not a plain file:// store: newer MLflow puts that backend into
        # maintenance mode and raises on every use (confirmed on a real
        # Kaggle run) -- SQLite is the zero-config local fallback instead.
        assert uri.startswith("sqlite:///") and uri.endswith("mlflow.db")
        assert "dagshub" not in uri
        assert mlflow_fake.called("set_experiment")
        assert out_dir.is_dir()  # must create the output dir so sqlite has somewhere to write

    def test_an_explicit_remote_uri_is_used_and_can_hold_a_registry(self, train_module, mlflow_fake, tmp_path):
        info = train_module.configure_mlflow(_config(tracking_uri="https://mlflow.example.com"), str(tmp_path))

        assert info.kind == "remote"
        assert info.registry_capable
        assert mlflow_fake.get_tracking_uri() == "https://mlflow.example.com"

    @pytest.mark.parametrize("uri", ["sqlite:///mlflow.db", "postgresql://u@h/db", "databricks"])
    def test_database_backed_stores_count_as_registry_capable(self, train_module, mlflow_fake, tmp_path, uri):
        assert train_module.configure_mlflow(_config(tracking_uri=uri), str(tmp_path)).registry_capable

    def test_an_explicit_file_uri_is_local(self, train_module, mlflow_fake, tmp_path):
        info = train_module.configure_mlflow(_config(tracking_uri="file:///tmp/runs"), str(tmp_path))
        assert info.kind == "local"

    def test_explicit_uri_wins_over_a_dagshub_token(self, train_module, mlflow_fake, tmp_path, monkeypatch):
        monkeypatch.setenv("DAGSHUB_USER_TOKEN", "tok")
        info = train_module.configure_mlflow(_config(tracking_uri="https://mlflow.example.com"), str(tmp_path))
        assert info.uri == "https://mlflow.example.com"

    def test_dagshub_is_used_only_when_a_token_is_set(self, train_module, mlflow_fake, tmp_path, monkeypatch):
        monkeypatch.setenv("DAGSHUB_USER_TOKEN", "tok")
        init_calls = []

        def init(**kwargs):
            init_calls.append(kwargs)
            mlflow_fake._uri = "https://dagshub.com/owner/repo.mlflow"

        monkeypatch.setitem(sys.modules, "dagshub", SimpleNamespace(init=init))

        info = train_module.configure_mlflow(_config(), str(tmp_path))

        assert init_calls == [{"repo_owner": "owner", "repo_name": "repo", "mlflow": True}]
        assert info.kind == "remote"

    def test_a_failing_dagshub_init_falls_back_to_a_local_store(self, train_module, mlflow_fake, tmp_path, monkeypatch):
        monkeypatch.setenv("DAGSHUB_USER_TOKEN", "tok")

        def init(**kwargs):
            raise RuntimeError("401")

        monkeypatch.setitem(sys.modules, "dagshub", SimpleNamespace(init=init))

        info = train_module.configure_mlflow(_config(), str(tmp_path))

        assert info.kind == "local"
        assert mlflow_fake.get_tracking_uri().startswith("sqlite:///")


class TestRedactUri:
    def test_credentials_are_removed(self, train_module):
        assert train_module._redact_uri("https://user:s3cret@mlflow.example.com:5000/x") == "https://mlflow.example.com:5000/x"

    def test_uris_without_credentials_are_unchanged(self, train_module):
        assert train_module._redact_uri("https://mlflow.example.com/x") == "https://mlflow.example.com/x"
        assert train_module._redact_uri("file:///tmp/mlruns") == "file:///tmp/mlruns"


class TestFinishMlflowRun:
    LOCAL = staticmethod(lambda tm: tm.TrackingInfo("file:///tmp/mlruns", "local"))
    REMOTE = staticmethod(lambda tm: tm.TrackingInfo("https://mlflow.example.com", "remote"))

    def test_a_local_store_logs_metrics_and_skips_artifacts_and_registry(self, train_module, mlflow_fake):
        status = train_module.finish_mlflow_run(
            self.LOCAL(train_module), {"register_model": True}, "run-1", "out", {"train_loss": 0.5, "epoch": 3.0, "note": "x", "flag": True},
        )

        assert status["metrics_logged"] == "ok"
        assert status["artifacts"].startswith("skipped")
        assert status["registry"].startswith("skipped")
        assert not mlflow_fake.called("log_artifacts")
        assert not mlflow_fake.called("register_model")
        logged = next(c for c in mlflow_fake.calls if c[0] == "log_metrics")[1][0]
        assert logged == {"train_loss": 0.5, "epoch": 3.0}

    def test_a_remote_store_uploads_artifacts_but_registers_only_when_enabled(self, train_module, mlflow_fake):
        status = train_module.finish_mlflow_run(self.REMOTE(train_module), {"register_model": False}, "run-1", "out", {})

        assert status["artifacts"] == "ok"
        assert status["registry"] == "skipped: mlflow.register_model is false"
        assert mlflow_fake.called("log_artifacts")
        assert not mlflow_fake.called("register_model")

    def test_registration_runs_when_enabled_on_a_remote_store(self, train_module, mlflow_fake):
        status = train_module.finish_mlflow_run(
            self.REMOTE(train_module), {"register_model": True, "registered_model_name": "model-x"}, "run-9", "out", {},
        )

        assert status["registry"] == "registered"
        call = next(c for c in mlflow_fake.calls if c[0] == "register_model")
        assert call[1] == ("runs:/run-9/adapter",)
        assert call[2]["name"] == "model-x"

    @pytest.mark.parametrize("failing", ["log_metrics", "log_artifacts", "register_model"])
    def test_no_mlflow_failure_can_raise_out_of_the_run(self, train_module, mlflow_fake, failing):
        mlflow_fake.fail.add(failing)

        status = train_module.finish_mlflow_run(
            self.REMOTE(train_module), {"register_model": True, "registered_model_name": "m"}, "run-1", "out", {"train_loss": 1.0},
        )  # must not raise

        key = {"log_metrics": "metrics_logged", "log_artifacts": "artifacts", "register_model": "registry"}[failing]
        assert status[key].startswith("failed")

    def test_credentials_in_the_uri_never_reach_the_status(self, train_module, mlflow_fake):
        tracking = train_module.TrackingInfo("https://user:hunter2@mlflow.example.com", "remote")
        status = train_module.finish_mlflow_run(tracking, {}, "run-1", "out", {})
        assert "hunter2" not in json.dumps(status)


class TestBuildRunInfo:
    def _config(self):
        return {
            "model": {"max_seq_length": 2048},
            "training": {"num_epochs": 3, "batch_size": 1, "gradient_accumulation_steps": 4, "learning_rate": "5.0e-4"},
            "lora": {"r": 16},
        }

    def test_records_what_was_trained_on_and_how(self, train_module, tmp_path):
        data = tmp_path / "train.jsonl"
        data.write_text('{"a": 1}\n')

        info = train_module.build_run_info(
            config=self._config(), model_name="meta-llama/Llama-3.1-8B", data_path=str(data),
            mlflow_status={"tracking": "local"}, train_examples=90, trainable=42_000_000, total=8_000_000_000,
        )

        assert info["base_model"] == "meta-llama/Llama-3.1-8B"
        assert info["dataset_sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
        assert info["train_examples"] == 90
        assert info["effective_batch_size"] == 4
        assert info["learning_rate"] == 5e-4
        assert info["trainable_params"] == 42_000_000
        assert info["mlflow"] == {"tracking": "local"}
        assert "gpu" not in info

    def test_a_missing_data_file_gives_a_null_hash_not_a_crash(self, train_module, tmp_path):
        info = train_module.build_run_info(
            config=self._config(), model_name="m", data_path=str(tmp_path / "gone.jsonl"),
            mlflow_status={}, train_examples=0, trainable=1, total=2,
        )
        assert info["dataset_sha256"] is None

    def test_gpu_details_are_included_when_cuda_is_available(self, train_module, tmp_path):
        with patch.object(train_module, "torch", _fake_torch(cuda=True)):
            info = train_module.build_run_info(
                config=self._config(), model_name="m", data_path=str(tmp_path / "x"),
                mlflow_status={}, train_examples=1, trainable=1, total=2,
            )
        assert info["gpu"] == "Tesla T4"
        assert info["peak_gpu_memory_gb"] == 11.5


class TestTrainRunsToCompletionDespiteMlflowFailures:
    """The point of the whole change: the adapter and metrics get written and
    train() returns, whatever MLFlow does after the adapter is saved."""

    def _wire(self, train_module, tmp_path, monkeypatch, config):
        data = tmp_path / "train.jsonl"
        data.write_text('{"a": 1}\n')

        saved = []

        class FakeModel:
            def save_pretrained(self, out):
                saved.append(("model", out))
                Path(out).mkdir(parents=True, exist_ok=True)

            def parameters(self):
                return [SimpleNamespace(numel=lambda: 1000, requires_grad=True),
                        SimpleNamespace(numel=lambda: 9000, requires_grad=False)]

        class FakeTokenizer:
            def save_pretrained(self, out):
                saved.append(("tokenizer", out))

        class FakeTrainer:
            def __init__(self, **kwargs):
                pass

            def train(self):
                return SimpleNamespace(metrics={"train_loss": 0.9, "epoch": 3.0})

        monkeypatch.setattr(train_module, "load_config", lambda: config)
        monkeypatch.setattr(train_module, "resolve_dataset_path", lambda cfg, override=None: str(data))
        monkeypatch.setattr(train_module, "create_bnb_config", lambda cfg: object())
        monkeypatch.setattr(train_module, "load_base_model", lambda *a, **k: (FakeModel(), FakeTokenizer()))
        monkeypatch.setattr(train_module, "create_lora_config", lambda cfg: object())
        monkeypatch.setattr(train_module, "get_peft_model", lambda model, lora: model)
        monkeypatch.setattr(train_module, "prepare_dataset", lambda *a, **k: [1, 2, 3])
        monkeypatch.setattr(train_module, "to_text_dataset", lambda ds, tok: ds)
        monkeypatch.setattr(train_module, "create_training_args", lambda *a, **k: SimpleNamespace())
        monkeypatch.setattr(train_module, "SFTTrainer", FakeTrainer)
        monkeypatch.setattr(train_module, "MetricsCallback", lambda: object())
        monkeypatch.setattr(train_module, "EarlyStoppingOnLoss", lambda **k: object())
        return saved

    def _full_config(self, **mlflow_overrides):
        cfg = _config(**mlflow_overrides)
        cfg.update({
            "model": {"base_model": "meta-llama/Llama-3.1-8B", "max_seq_length": 2048},
            "training": {"num_epochs": 3, "batch_size": 1, "gradient_accumulation_steps": 4,
                         "learning_rate": 5e-4, "output_dir": "unused"},
            "lora": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05},
            "kaggle": {},
            "data": {"max_train_samples": None},
        })
        return cfg

    def test_local_tracking_writes_the_adapter_metrics_and_run_info(self, train_module, mlflow_fake, tmp_path, monkeypatch):
        monkeypatch.delenv("DAGSHUB_USER_TOKEN", raising=False)
        config = self._full_config()
        saved = self._wire(train_module, tmp_path, monkeypatch, config)
        out = tmp_path / "adapter"

        metrics = train_module.train(output_dir=str(out))

        assert metrics == {"train_loss": 0.9, "epoch": 3.0}
        assert [s[0] for s in saved] == ["model", "tokenizer"]
        assert json.loads((out / "training_metrics.json").read_text()) == metrics
        info = json.loads((out / "training_run_info.json").read_text())
        assert info["train_examples"] == 3
        assert info["trainable_params"] == 1000 and info["total_params"] == 10000
        assert info["mlflow"]["tracking"] == "local"

    def test_every_mlflow_bookkeeping_failure_still_lets_training_finish(self, train_module, mlflow_fake, tmp_path, monkeypatch):
        mlflow_fake.fail.update({"log_metrics", "log_artifacts", "register_model"})
        config = self._full_config(tracking_uri="https://mlflow.example.com", register_model=True)
        self._wire(train_module, tmp_path, monkeypatch, config)
        out = tmp_path / "adapter"

        metrics = train_module.train(output_dir=str(out))  # must not raise

        assert metrics["train_loss"] == 0.9
        assert (out / "training_metrics.json").exists()
        info = json.loads((out / "training_run_info.json").read_text())
        assert info["mlflow"]["registry"].startswith("failed")
        assert info["mlflow"]["artifacts"].startswith("failed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
