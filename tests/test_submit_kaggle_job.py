"""Tests for scripts/submit_kaggle_job.py's run-mode handling.

Kaggle uploads only the kernel's metadata and code file, so the run mode is
written into a throwaway copy of the code. These tests check that copy is
correct, the checked-in file is never modified, and the temp dir is cleaned
up. The Kaggle API itself is replaced by a recording fake.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.submit_kaggle_job as submit
from scripts.submit_kaggle_job import RUN_MODES, prepare_kernel_dir, push_kernel

REAL_KERNEL_DIR = Path(submit.KERNEL_DIR)


class RecordingApi:
    def __init__(self):
        self.pushed_from = None
        self.pushed_files: dict[str, str] = {}

    def kernels_push(self, folder):
        self.pushed_from = folder
        for f in Path(folder).iterdir():
            self.pushed_files[f.name] = f.read_text(encoding="utf-8")


class TestPrepareKernelDir:
    @pytest.mark.parametrize("mode", RUN_MODES)
    def test_the_copy_has_the_requested_mode(self, mode):
        tmp = prepare_kernel_dir(mode)
        try:
            code = (tmp / "train_kernel.py").read_text(encoding="utf-8")
            assert f'RUN_MODE = "{mode}"' in code
            assert code.count("RUN_MODE = ") == 1
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_only_the_files_kaggle_uses_are_copied_and_metadata_is_untouched(self):
        tmp = prepare_kernel_dir("smoke")
        try:
            assert sorted(p.name for p in tmp.iterdir()) == ["kernel-metadata.json", "train_kernel.py"]
            assert (tmp / "kernel-metadata.json").read_text(encoding="utf-8") == (
                REAL_KERNEL_DIR / "kernel-metadata.json"
            ).read_text(encoding="utf-8")
            assert json.loads((tmp / "kernel-metadata.json").read_text())["code_file"] == "train_kernel.py"
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_checked_in_kernel_file_is_never_modified(self):
        before = (REAL_KERNEL_DIR / "train_kernel.py").read_text(encoding="utf-8")
        import shutil

        shutil.rmtree(prepare_kernel_dir("smoke"), ignore_errors=True)
        assert (REAL_KERNEL_DIR / "train_kernel.py").read_text(encoding="utf-8") == before

    def test_only_the_mode_line_differs_between_modes(self):
        smoke, full = prepare_kernel_dir("smoke"), prepare_kernel_dir("full")
        try:
            a = (smoke / "train_kernel.py").read_text(encoding="utf-8").splitlines()
            b = (full / "train_kernel.py").read_text(encoding="utf-8").splitlines()
            differing = [(x, y) for x, y in zip(a, b) if x != y]
            assert len(a) == len(b)
            assert len(differing) == 1
            assert differing[0][0].startswith('RUN_MODE = "smoke"')
            assert differing[0][1].startswith('RUN_MODE = "full"')
        finally:
            import shutil

            shutil.rmtree(smoke, ignore_errors=True)
            shutil.rmtree(full, ignore_errors=True)

    def test_an_unknown_mode_is_rejected_before_anything_is_written(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            prepare_kernel_dir("turbo")

    def test_a_kernel_file_without_exactly_one_mode_line_is_an_error(self, tmp_path, monkeypatch):
        (tmp_path / "kernel-metadata.json").write_text("{}")
        (tmp_path / "train_kernel.py").write_text("print('no mode here')\n")
        monkeypatch.setattr(submit, "KERNEL_DIR", tmp_path)

        with pytest.raises(RuntimeError, match="exactly one RUN_MODE"):
            prepare_kernel_dir("smoke")

        (tmp_path / "train_kernel.py").write_text('RUN_MODE = "full"\nRUN_MODE = "full"\n')
        with pytest.raises(RuntimeError, match="exactly one RUN_MODE"):
            prepare_kernel_dir("smoke")


class TestPushKernel:
    CONFIG = {"kaggle": {"kernel_slug": "user/slug"}}

    def test_pushes_the_patched_copy_not_the_checked_in_folder(self):
        api = RecordingApi()

        slug = push_kernel(api, self.CONFIG, "smoke")

        assert slug == "user/slug"
        assert Path(api.pushed_from) != REAL_KERNEL_DIR
        assert 'RUN_MODE = "smoke"' in api.pushed_files["train_kernel.py"]

    def test_defaults_to_a_full_run(self):
        api = RecordingApi()
        push_kernel(api, self.CONFIG)
        assert 'RUN_MODE = "full"' in api.pushed_files["train_kernel.py"]

    def test_the_temp_copy_is_removed_afterwards_even_if_the_push_fails(self):
        seen = {}

        class FailingApi:
            def kernels_push(self, folder):
                seen["folder"] = folder
                raise RuntimeError("Kaggle is down")

        with pytest.raises(RuntimeError, match="Kaggle is down"):
            push_kernel(FailingApi(), self.CONFIG, "full")

        assert not Path(seen["folder"]).exists()


class TestCommandLine:
    def test_mode_defaults_to_full_for_backwards_compatibility(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["submit_kaggle_job.py", "--wait"])
        assert submit.parse_args().mode == "full"

    def test_smoke_mode_can_be_requested(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["submit_kaggle_job.py", "--wait", "--mode", "smoke"])
        assert submit.parse_args().mode == "smoke"

    def test_an_invalid_mode_is_rejected_by_argparse(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["submit_kaggle_job.py", "--mode", "turbo"])
        with pytest.raises(SystemExit):
            submit.parse_args()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
