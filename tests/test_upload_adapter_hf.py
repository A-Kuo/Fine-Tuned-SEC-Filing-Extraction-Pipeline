"""Tests for scripts/upload_adapter_hf.py with a fake Hub client: no network,
no token, nothing is really uploaded."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.upload_adapter_hf import (
    UPLOAD_ALLOWLIST,
    adapter_files,
    build_readme,
    resolve_settings,
    upload,
)


class FakeApi:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))
        if self.fail_on == "create_repo":
            raise RuntimeError("403 Forbidden: token is read-only")

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))
        if self.fail_on == "upload_folder":
            raise RuntimeError("network down")


@pytest.fixture
def adapter_dir(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    for name in ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json", "training_run_info.json"):
        (d / name).write_text("x")
    # Things that must never be uploaded:
    (d / "checkpoint-22").mkdir()
    (d / "checkpoint-22" / "optimizer.pt").write_text("huge")
    (d / "mlruns").mkdir()
    (d / "stray_notes.txt").write_text("private")
    return d


SETTINGS = {"repo": "someone/sec-adapter", "token": "hf_fake", "private": True}


class TestResolveSettings:
    def test_repo_precedence_override_then_env_then_config(self):
        cfg = {"huggingface": {"adapter_repo": "cfg/repo", "private": True}}
        assert resolve_settings(cfg, {}, repo_override="cli/repo")["repo"] == "cli/repo"
        assert resolve_settings(cfg, {"HF_ADAPTER_REPO": "env/repo"})["repo"] == "env/repo"
        assert resolve_settings(cfg, {})["repo"] == "cfg/repo"
        assert resolve_settings({}, {})["repo"] == ""

    def test_write_token_wins_over_read_token(self):
        assert resolve_settings({}, {"HF_WRITE_TOKEN": "w", "HF_TOKEN": "r"})["token"] == "w"
        assert resolve_settings({}, {"HF_TOKEN": "r"})["token"] == "r"
        assert resolve_settings({}, {})["token"] == ""

    def test_private_by_default(self):
        assert resolve_settings({}, {})["private"] is True
        assert resolve_settings({"huggingface": {"private": False}}, {})["private"] is False


class TestFileSelection:
    def test_only_allowlisted_files_that_exist_are_chosen(self, adapter_dir):
        files = adapter_files(adapter_dir)
        assert set(files) == {"adapter_model.safetensors", "adapter_config.json", "tokenizer.json", "training_run_info.json"}
        assert "stray_notes.txt" not in files
        assert not any("checkpoint" in f or "mlruns" in f for f in files)


class TestUpload:
    def test_uploads_only_the_allowlist_to_a_private_repo(self, adapter_dir):
        api = FakeApi()

        result = upload(adapter_dir, SETTINGS, api=api)

        assert result["status"] == "uploaded"
        create = dict(api.calls[0][1])
        assert create["repo_id"] == "someone/sec-adapter"
        assert create["private"] is True
        assert create["exist_ok"] is True
        push = api.calls[1][1]
        assert set(push["allow_patterns"]) == set(UPLOAD_ALLOWLIST) | {"README.md"}
        assert "hf_fake" not in str(result)

    def test_a_model_card_is_written_alongside_the_adapter(self, adapter_dir):
        upload(adapter_dir, SETTINGS, api=FakeApi())
        readme = (adapter_dir / "README.md").read_text()
        assert "base_model: meta-llama/Llama-3.1-8B" in readme
        assert "has **not** been measured" in readme

    @pytest.mark.parametrize(
        "settings,reason_fragment",
        [
            ({"repo": "", "token": "t", "private": True}, "no repo configured"),
            ({"repo": "not-a-valid-id", "token": "t", "private": True}, "owner/name"),
            ({"repo": "a/b", "token": "", "private": True}, "no HF_WRITE_TOKEN"),
        ],
    )
    def test_missing_prerequisites_are_skipped_with_a_reason_and_no_upload(self, adapter_dir, settings, reason_fragment):
        api = FakeApi()
        result = upload(adapter_dir, settings, api=api)
        assert result["status"] == "skipped"
        assert reason_fragment in result["reason"]
        assert api.calls == []

    def test_no_adapter_weights_means_nothing_to_upload(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text("x")
        api = FakeApi()
        result = upload(tmp_path, SETTINGS, api=api)
        assert result["status"] == "skipped"
        assert "no adapter weights" in result["reason"]
        assert api.calls == []

    def test_dry_run_uploads_nothing(self, adapter_dir):
        api = FakeApi()
        result = upload(adapter_dir, SETTINGS, dry_run=True, api=api)
        assert result["status"] == "skipped"
        assert result["files"]
        assert api.calls == []

    @pytest.mark.parametrize("failing_call", ["create_repo", "upload_folder"])
    def test_hub_errors_become_a_failed_status_not_an_exception(self, adapter_dir, failing_call):
        result = upload(adapter_dir, SETTINGS, api=FakeApi(fail_on=failing_call))
        assert result["status"] == "failed"
        assert result["reason"]


class TestReadme:
    def test_model_card_quotes_no_metrics(self):
        readme = build_readme("a/b")
        assert "%" not in readme
        assert "synthetic" in readme
        assert "a/b" in readme


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
