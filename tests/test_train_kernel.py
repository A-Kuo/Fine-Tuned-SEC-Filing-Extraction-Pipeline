"""Tests for scripts/kaggle_kernel/train_kernel.py's data-generation fix.

Regression coverage for the bug where the Kaggle kernel ran
training/train.py directly against a training data file that was never
generated (config.yaml's kaggle.dataset_id is empty, and its
local_fallback, data/sec_filings_train.jsonl, is gitignored -- absent from
this script's fresh git clone) -- training crashed with FileNotFoundError
almost immediately. Mocks subprocess.run entirely; no real clone/install/
train happens in this test.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.kaggle_kernel.train_kernel import main


class TestDataGenerationOrder:
    def test_download_and_format_run_before_training(self):
        with patch("scripts.kaggle_kernel.train_kernel.subprocess.run") as mock_run, \
             patch("scripts.kaggle_kernel.train_kernel._load_kaggle_secrets"):
            mock_run.return_value = MagicMock(returncode=0)
            main()

        commands = [call.args[0] for call in mock_run.call_args_list]
        assert any("git" in cmd[0] and "clone" in cmd for cmd in commands)
        assert any("scripts/download_dataset.py" in cmd for cmd in commands)
        assert any("scripts/format_data.py" in cmd for cmd in commands)
        assert any("training/train.py" in cmd for cmd in commands)

        download_idx = next(i for i, cmd in enumerate(commands) if "scripts/download_dataset.py" in cmd)
        format_idx = next(i for i, cmd in enumerate(commands) if "scripts/format_data.py" in cmd)
        train_idx = next(i for i, cmd in enumerate(commands) if "training/train.py" in cmd)

        assert download_idx < format_idx < train_idx

    def test_all_subprocess_calls_use_check_true(self):
        """A silent data-generation failure must not be allowed to fall
        through to training/train.py's own (uninformative) crash."""
        with patch("scripts.kaggle_kernel.train_kernel.subprocess.run") as mock_run, \
             patch("scripts.kaggle_kernel.train_kernel._load_kaggle_secrets"):
            mock_run.return_value = MagicMock(returncode=0)
            main()

        for call in mock_run.call_args_list:
            assert call.kwargs.get("check") is True
