"""Tiny, dependency-free dict helpers.

Deliberately has zero imports beyond stdlib -- training/train.py needs
torch/peft/trl (not in requirements-ci.txt, "no torch/vLLM/GPU" by design),
so anything it imports can't be unit-tested on the CI runner. Pulling
deep_merge() out here means it (and anything else that should stay
testable without a GPU-capable environment) gets real test coverage.
"""

from __future__ import annotations


def deep_merge(base: dict, overrides: dict) -> None:
    """In-place recursive merge of `overrides` into `base` (dicts merge
    key-by-key; any other type, including lists, is replaced wholesale)."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
