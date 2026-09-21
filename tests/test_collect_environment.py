"""Tests for scripts/kaggle_kernel/collect_environment.py: the record of what a
training run executed on must never crash the run it is describing."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.kaggle_kernel.collect_environment as env

REAL_GPU_INFO = env.gpu_info


@pytest.fixture(autouse=True)
def no_real_torch_import(monkeypatch):
    """Importing the real torch takes tens of seconds; only TestGpuInfo needs
    gpu_info's real code, and it injects a fake torch instead."""
    monkeypatch.setattr(env, "gpu_info", lambda: {"cuda_available": False})


class TestPackageVersions:
    def test_installed_packages_report_a_version_and_missing_ones_report_null(self):
        versions = env.package_versions(["pytest", "definitely-not-an-installed-package"])
        assert isinstance(versions["pytest"], str) and versions["pytest"]
        assert versions["definitely-not-an-installed-package"] is None

    def test_every_library_that_affects_training_is_tracked(self):
        assert {"torch", "transformers", "peft", "trl", "bitsandbytes"} <= set(env.PACKAGES)


class TestFileFingerprint:
    def test_size_and_hash_of_a_file(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_bytes(b'{"a": 1}\n')
        assert env.file_fingerprint(f) == {"bytes": 9, "sha256": hashlib.sha256(b'{"a": 1}\n').hexdigest()}

    def test_a_missing_file_is_null_not_an_error(self, tmp_path):
        assert env.file_fingerprint(tmp_path / "nope") is None


class TestGitCommit:
    def test_a_directory_that_is_not_a_repository_gives_null(self, tmp_path):
        assert env.git_commit(tmp_path) is None


class TestCollect:
    def test_report_has_the_expected_sections_and_tracks_the_data_files(self, tmp_path, monkeypatch):
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "sec_filings_train.jsonl").write_bytes(b"row\n")
        monkeypatch.setattr(env, "git_commit", lambda repo_dir: "abc123")
        monkeypatch.setattr(env, "gpu_info", lambda: {"cuda_available": True, "name": "Tesla T4"})

        report = env.collect(tmp_path)

        assert report["git_commit"] == "abc123"
        assert report["gpu"]["name"] == "Tesla T4"
        assert set(report["packages"]) == set(env.PACKAGES)
        assert report["data_files"]["data/sec_filings_train.jsonl"]["bytes"] == 4
        assert report["data_files"]["data/sec_filings_test.jsonl"] is None
        assert report["python"] and report["collected_at"]

    def test_it_is_json_serializable(self, tmp_path):
        assert json.loads(json.dumps(env.collect(tmp_path)))["schema_version"] == 1


class TestGpuInfo:
    def _fake_torch(self, available: bool):
        from types import SimpleNamespace

        props = SimpleNamespace(name="Tesla T4", total_memory=15_640_000_000)
        return SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: available,
                device_count=lambda: 2,
                get_device_properties=lambda i: props,
            ),
            version=SimpleNamespace(cuda="12.1"),
        )

    def test_reports_the_device_when_cuda_is_available(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", self._fake_torch(True))
        assert REAL_GPU_INFO() == {
            "cuda_available": True, "device_count": 2, "name": "Tesla T4",
            "total_memory_gb": 15.6, "torch_cuda_version": "12.1",
        }

    def test_reports_no_cuda_without_a_gpu(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", self._fake_torch(False))
        assert REAL_GPU_INFO() == {"cuda_available": False}

    def test_a_missing_torch_is_not_an_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
        assert REAL_GPU_INFO()["cuda_available"] is False


class TestMain:
    def test_writes_the_report_creating_parent_directories(self, tmp_path, monkeypatch):
        out = tmp_path / "nested" / "environment.json"
        monkeypatch.setattr(sys, "argv", ["collect_environment.py", "--repo-dir", str(tmp_path), "--out", str(out)])

        env.main()

        assert json.loads(out.read_text())["schema_version"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
