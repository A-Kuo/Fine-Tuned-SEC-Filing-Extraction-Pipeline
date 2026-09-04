"""Tests for training/ablation_config.py. Dependency-free by design (see its
module docstring) -- these must be able to run without torch/peft/trl
installed, same as training/train.py's own CI-testability constraint."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.ablation_config import (
    ABLATION_CONFIGS,
    ALL_LINEAR_TARGET_MODULES,
    AblationConfig,
)


class TestConfigId:
    def test_deterministic(self):
        c = AblationConfig(name="x", lora_r=16)
        assert c.config_id == AblationConfig(name="x", lora_r=16).config_id

    def test_id_independent_of_name(self):
        """The id identifies the CONFIG, not the label -- two configs with
        the same hyperparameters but different names should collide, which
        is itself useful (it flags an accidental duplicate)."""
        c1 = AblationConfig(name="alpha", lora_r=16)
        c2 = AblationConfig(name="beta", lora_r=16)
        assert c1.config_id == c2.config_id

    def test_different_rank_different_id(self):
        c1 = AblationConfig(name="x", lora_r=16)
        c2 = AblationConfig(name="x", lora_r=32)
        assert c1.config_id != c2.config_id

    def test_target_module_order_does_not_matter(self):
        c1 = AblationConfig(name="x", target_modules=("q_proj", "v_proj"))
        c2 = AblationConfig(name="x", target_modules=("v_proj", "q_proj"))
        assert c1.config_id == c2.config_id

    def test_all_three_named_configs_have_distinct_ids(self):
        ids = [c.config_id for c in ABLATION_CONFIGS]
        assert len(ids) == len(set(ids))


class TestToConfigOverrides:
    def test_lora_section_reflects_rank_and_modules(self):
        c = AblationConfig(name="x", lora_r=8, target_modules=("q_proj",))
        overrides = c.to_config_overrides()
        assert overrides["lora"]["r"] == 8
        assert overrides["lora"]["target_modules"] == ["q_proj"]

    def test_merges_cleanly_via_deep_merge(self):
        """The overlay this produces must actually work with
        src/core/dict_utils.py::deep_merge, which is what
        training.train.train()'s config_overrides parameter uses."""
        from src.core.dict_utils import deep_merge

        base_config = {
            "lora": {"r": 16, "lora_alpha": 32, "target_modules": list(ALL_LINEAR_TARGET_MODULES)},
            "training": {"num_epochs": 3},
            "model": {"base_model": "meta-llama/Llama-3.1-8B"},
        }
        narrow = AblationConfig(name="narrow", target_modules=("q_proj", "v_proj"))
        deep_merge(base_config, narrow.to_config_overrides())

        assert base_config["lora"]["target_modules"] == ["q_proj", "v_proj"]
        assert base_config["training"]["num_epochs"] == 3  # untouched key survives
        assert base_config["model"]["base_model"] == "meta-llama/Llama-3.1-8B"  # untouched


class TestAblationConfigsRoster:
    def test_baseline_matches_repo_defaults(self):
        """config.yaml's lora section (r=16, all 7 linear layers) should
        match the baseline config, so it's a genuine no-op comparison point."""
        baseline = next(c for c in ABLATION_CONFIGS if c.name == "baseline_all_linear_r16")
        assert baseline.lora_r == 16
        assert set(baseline.target_modules) == set(ALL_LINEAR_TARGET_MODULES)

    def test_narrow_scope_isolates_target_modules_only(self):
        """Comparing narrow_scope vs baseline should isolate exactly one
        variable (target_modules) -- rank must be held constant."""
        baseline = next(c for c in ABLATION_CONFIGS if c.name == "baseline_all_linear_r16")
        narrow = next(c for c in ABLATION_CONFIGS if c.name == "narrow_scope_r16")
        assert narrow.lora_r == baseline.lora_r
        assert set(narrow.target_modules) != set(baseline.target_modules)

    def test_at_least_two_configs_for_comparison(self):
        assert len(ABLATION_CONFIGS) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
