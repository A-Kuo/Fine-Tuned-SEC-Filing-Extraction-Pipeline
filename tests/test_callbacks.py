"""Tests for training/callbacks.py.

Regression coverage for a real bug hit on a live Kaggle T4 run: HF Trainer's
final on_log() call (after training completes) logs aggregate metrics
(train_loss, train_runtime, ...) with no "learning_rate" key, so `lr` is
legitimately None there -- MetricsCallback.on_log() crashed formatting it as
`{lr:.2e}`, losing the whole run's result one line before it would have been
saved.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.callbacks import EarlyStoppingOnLoss, MetricsCallback


def _state(step=1, epoch=1.0):
    return SimpleNamespace(global_step=step, epoch=epoch)


class TestMetricsCallbackOnLog:
    def test_mid_training_log_with_learning_rate(self):
        cb = MetricsCallback()
        cb.on_train_begin(None, _state(), None)
        cb.on_log(None, _state(step=10, epoch=0.5), None, logs={"loss": 1.234, "learning_rate": 5e-4})
        assert cb.metrics_log[-1]["step"] == 10

    def test_final_summary_log_has_no_learning_rate_key(self):
        """This is the exact shape of HF Trainer's post-training on_log() call
        that crashed in production: train_loss present, learning_rate absent."""
        cb = MetricsCallback()
        cb.on_train_begin(None, _state(), None)

        # Must not raise.
        cb.on_log(
            None, _state(step=930, epoch=3.0), None,
            logs={"train_loss": 0.842, "train_runtime": 500.0},
        )

        assert cb.metrics_log[-1]["train_loss"] == 0.842

    def test_logs_none_is_a_noop(self):
        cb = MetricsCallback()
        cb.on_train_begin(None, _state(), None)
        cb.on_log(None, _state(), None, logs=None)
        assert cb.metrics_log == []

    def test_on_train_end_writes_metrics_log(self, tmp_path):
        cb = MetricsCallback()
        cb.on_train_begin(None, _state(), None)
        cb.on_log(None, _state(step=1, epoch=0.1), None, logs={"loss": 2.0, "learning_rate": 1e-4})

        args = SimpleNamespace(output_dir=str(tmp_path))
        cb.on_train_end(args, _state(), None)

        out = tmp_path / "training_log.json"
        assert out.exists()


class TestEarlyStoppingOnLoss:
    def test_no_loss_key_is_a_noop(self):
        cb = EarlyStoppingOnLoss(patience=2, min_delta=0.01)
        control = SimpleNamespace(should_training_stop=False)
        cb.on_log(None, _state(), control, logs={"learning_rate": 1e-4})
        assert control.should_training_stop is False

    def test_stops_after_patience_exceeded_with_no_improvement(self):
        cb = EarlyStoppingOnLoss(patience=2, min_delta=0.01, smoothing=1.0)
        control = SimpleNamespace(should_training_stop=False)
        for _ in range(4):
            cb.on_log(None, _state(), control, logs={"loss": 1.0})
        assert control.should_training_stop is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
