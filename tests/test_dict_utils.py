"""Tests for src/core/dict_utils.py. Deliberately dependency-free (see the
module docstring for why) so this stays testable regardless of what ML
libraries are or aren't installed in a given environment."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.dict_utils import deep_merge


class TestDeepMerge:
    def test_simple_scalar_override(self):
        base = {"a": 1, "b": 2}
        deep_merge(base, {"b": 20})
        assert base == {"a": 1, "b": 20}

    def test_adds_new_key(self):
        base = {"a": 1}
        deep_merge(base, {"c": 3})
        assert base == {"a": 1, "c": 3}

    def test_nested_dict_merges_key_by_key(self):
        base = {"lora": {"r": 16, "alpha": 32}}
        deep_merge(base, {"lora": {"r": 8}})
        assert base == {"lora": {"r": 8, "alpha": 32}}

    def test_list_is_replaced_wholesale_not_merged(self):
        base = {"lora": {"target_modules": ["q_proj", "v_proj"]}}
        deep_merge(base, {"lora": {"target_modules": ["all-linear"]}})
        assert base["lora"]["target_modules"] == ["all-linear"]

    def test_deeply_nested_merge(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        deep_merge(base, {"a": {"b": {"c": 99}}})
        assert base == {"a": {"b": {"c": 99, "d": 2}}}

    def test_empty_overrides_is_a_noop(self):
        base = {"a": 1}
        deep_merge(base, {})
        assert base == {"a": 1}

    def test_overriding_a_dict_value_with_a_non_dict_replaces_it(self):
        base = {"a": {"b": 1}}
        deep_merge(base, {"a": "now a string"})
        assert base == {"a": "now a string"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
