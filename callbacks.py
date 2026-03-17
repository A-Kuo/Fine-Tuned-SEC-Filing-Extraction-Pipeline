"""Training callbacks for QLoRA fine-tuning.

Custom callbacks for monitoring training progress, early stopping,
and logging metrics that matter for production deployment.
"""

import json
import time
from pathlib import Path

from loguru import logger
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class MetricsCallback(TrainerCallback):
    """Log training metrics with timestamps for post-training analysis.

    Tracks loss curve, learning rate schedule, and throughput. These metrics
    feed into the monitoring dashboard and help diagnose training issues
    (e.g., loss spikes indicate data quality problems).
    """

    def __init__(self):
        self.metrics_log = []
        self.start_time = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.start_time = time.time()
        logger.info("Training started")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return

        entry = {
            "step": state.global_step,
            "epoch": state.epoch,
            "elapsed_seconds": time.time() - self.start_time if self.start_time else 0,
            **{k: v for k, v in logs.items() if isinstance(v, (int, float))},
        }
        self.metrics_log.append(entry)

        # Log key metrics
        loss = logs.get("loss", logs.get("train_loss"))
        lr = logs.get("learning_rate")
        if loss is not None:
            logger.info(
                f"Step {state.global_step} | "
                f"Loss: {loss:.4f} | "
                f"LR: {lr:.2e}" + (f" | Epoch: {state.epoch:.2f}" if state.epoch else "")
            )

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        elapsed = time.time() - self.start_time if self.start_time else 0
        logger.info(f"Training complete in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

        # Save full metrics log
        metrics_path = Path(args.output_dir) / "training_log.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_log, f, indent=2)
        logger.info(f"Metrics log saved to {metrics_path}")


class EarlyStoppingOnLoss(TrainerCallback):
    """Stop training if loss plateaus.

    Monitors training loss and stops if it hasn't improved by `min_delta`
    for `patience` consecutive logging steps. This prevents wasting compute
    on a converged model—important when GPU hours cost real money.

    The math: we track the exponential moving average of loss to smooth noise:
        EMA_t = α × loss_t + (1 - α) × EMA_{t-1}
    and stop if EMA hasn't decreased by min_delta in `patience` steps.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.01, smoothing: float = 0.3):
        self.patience = patience
        self.min_delta = min_delta
        self.smoothing = smoothing
        self.best_loss = float("inf")
        self.ema_loss = None
        self.wait = 0

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return

        loss = logs.get("loss")
        if loss is None:
            return

        # Update EMA
        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = self.smoothing * loss + (1 - self.smoothing) * self.ema_loss

        # Check improvement
        if self.ema_loss < self.best_loss - self.min_delta:
            self.best_loss = self.ema_loss
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            logger.warning(
                f"Early stopping: loss plateaued at {self.ema_loss:.4f} "
                f"for {self.patience} steps (best: {self.best_loss:.4f})"
            )
            control.should_training_stop = True
