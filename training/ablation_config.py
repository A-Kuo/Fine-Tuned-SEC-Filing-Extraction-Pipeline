"""Named training configurations for ablation comparisons: LoRA rank, target
modules, context-truncation strategy, prompt style, numeric-normalization
approach.

Deliberately dependency-free (no torch/peft/trl imports) -- unlike
training/train.py, this needs to be importable and testable without a
GPU-capable environment. Each AblationConfig produces both a config_overrides
dict (for training.train.train()'s config_overrides parameter) and a
deterministic config_id, so a result row in results/ablation_results.jsonl
can always be traced back to exactly what was run -- same spirit as
src/core/dataset_manifest.py's checksums.

This module only DEFINES the configs to compare; running them needs a GPU
(see scripts/kaggle_kernel/run_ablation_sweep.py) and cannot happen in a
CPU-only environment.

Honesty note on what's actually wired vs. just defined: `lora_r` and
`target_modules` genuinely change training behavior today --
training/train.py::create_lora_config() reads config["lora"]["r"] and
config["lora"]["target_modules"] directly. `truncation_strategy`,
`prompt_style`, and `numeric_normalization` are DEFINED here as ablation
axes worth comparing, but nothing in training/train.py currently reads
config["training"]["truncation_strategy"] or the other two -- setting them
via to_config_overrides() has no effect yet. Treat ABLATION_CONFIGS as a
real, runnable rank/target-module comparison plus three placeholder axes,
not a 5-axis sweep that's ready today. Wiring the other three in is future
work, not claimed as done here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class AblationConfig:
    name: str
    lora_r: int = 16
    lora_alpha: int = 32
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )
    truncation_strategy: Literal["truncate_end", "truncate_start", "sliding_window"] = "truncate_end"
    prompt_style: Literal["chat_template", "alpaca"] = "chat_template"
    numeric_normalization: Literal["raw_string", "normalized_float"] = "raw_string"

    @property
    def config_id(self) -> str:
        """Deterministic hash over every field that defines this config --
        two AblationConfigs with identical field values always get the same
        id, regardless of instantiation order."""
        canonical = json.dumps(
            {
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "target_modules": sorted(self.target_modules),
                "truncation_strategy": self.truncation_strategy,
                "prompt_style": self.prompt_style,
                "numeric_normalization": self.numeric_normalization,
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    def to_config_overrides(self) -> dict:
        """Shape matching training.train.train()'s config_overrides param --
        a nested dict merged into the loaded config.yaml before training."""
        return {
            "lora": {
                "r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "target_modules": list(self.target_modules),
            },
            "training": {
                "truncation_strategy": self.truncation_strategy,
                "numeric_normalization": self.numeric_normalization,
            },
            "model": {
                "prompt_style": self.prompt_style,
            },
        }


NARROW_TARGET_MODULES = ("q_proj", "v_proj")
ALL_LINEAR_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)

# The 2-3 configs requested for comparison. baseline_all_linear_r16 matches
# config.yaml's current defaults -- the other two isolate one variable each
# (target-module scope, LoRA rank) so a difference in results.json can be
# attributed to a single changed axis, not several at once.
ABLATION_CONFIGS: tuple[AblationConfig, ...] = (
    AblationConfig(
        name="baseline_all_linear_r16",
        lora_r=16, target_modules=ALL_LINEAR_TARGET_MODULES,
    ),
    AblationConfig(
        name="narrow_scope_r16",
        lora_r=16, target_modules=NARROW_TARGET_MODULES,
    ),
    AblationConfig(
        name="all_linear_r32",
        lora_r=32, lora_alpha=64, target_modules=ALL_LINEAR_TARGET_MODULES,
    ),
)
