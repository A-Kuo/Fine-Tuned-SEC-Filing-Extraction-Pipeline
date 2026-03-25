"""Tests for the A/B routing module."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ab_router import assign_for_request, ABAssignment


def _cfg(enabled=False, split=0.1, primary="models/v1", challenger="models/v2"):
    return {
        "model": {"adapter_path": primary},
        "ab_test": {
            "enabled": enabled,
            "primary_model_path": primary,
            "challenger_model_path": challenger,
            "traffic_split": split,
        },
    }


class TestABRouterDisabled:
    def test_returns_primary_when_disabled(self):
        cfg = _cfg(enabled=False)
        result = assign_for_request("filing-123", None, cfg)
        assert result.use_challenger is False
        assert result.model_version_label == "primary"

    def test_falls_back_to_adapter_path(self):
        cfg = {"model": {"adapter_path": "models/fallback"}, "ab_test": {}}
        result = assign_for_request(None, None, cfg)
        assert result.model_path == "models/fallback"
        assert result.use_challenger is False

    def test_header_override_ignored_when_disabled(self):
        cfg = _cfg(enabled=False)
        result = assign_for_request("fid", "challenger", cfg)
        assert result.use_challenger is False


class TestABRouterEnabled:
    def test_header_override_challenger(self):
        cfg = _cfg(enabled=True)
        result = assign_for_request("fid", "challenger", cfg)
        assert result.use_challenger is True
        assert result.model_version_label == "challenger"

    def test_header_override_primary(self):
        cfg = _cfg(enabled=True, split=1.0)  # would always pick challenger by hash
        result = assign_for_request("fid", "primary", cfg)
        assert result.use_challenger is False
        assert result.model_version_label == "primary"

    def test_deterministic_by_filing_id(self):
        cfg = _cfg(enabled=True, split=0.5)
        r1 = assign_for_request("stable-id-xyz", None, cfg)
        r2 = assign_for_request("stable-id-xyz", None, cfg)
        assert r1.use_challenger == r2.use_challenger

    def test_full_split_always_challenger(self):
        """With split=1.0 and no header override, all requests go to challenger."""
        cfg = _cfg(enabled=True, split=1.0)
        for i in range(20):
            r = assign_for_request(f"fid-{i}", None, cfg)
            assert r.use_challenger is True

    def test_zero_split_always_primary(self):
        """With split=0.0, all requests go to primary."""
        cfg = _cfg(enabled=True, split=0.0)
        for i in range(20):
            r = assign_for_request(f"fid-{i}", None, cfg)
            assert r.use_challenger is False

    def test_split_approximates_ratio(self):
        """With split=0.5, roughly half of a large sample goes to challenger."""
        cfg = _cfg(enabled=True, split=0.5)
        results = [assign_for_request(f"filing-{i:06d}", None, cfg) for i in range(1000)]
        challenger_count = sum(1 for r in results if r.use_challenger)
        # Expect between 40% and 60% (deterministic hash, not true random)
        assert 400 <= challenger_count <= 600, f"Got {challenger_count}/1000 challenger"

    def test_none_filing_id_uses_default_key(self):
        cfg = _cfg(enabled=True)
        r1 = assign_for_request(None, None, cfg)
        r2 = assign_for_request(None, None, cfg)
        assert r1.use_challenger == r2.use_challenger

    def test_model_paths_correctly_assigned(self):
        cfg = _cfg(enabled=True, primary="models/p", challenger="models/c", split=1.0)
        r = assign_for_request("any-id", None, cfg)
        assert r.model_path == "models/c"
        assert r.use_challenger is True

    def test_primary_model_paths_correctly_assigned(self):
        cfg = _cfg(enabled=True, primary="models/p", challenger="models/c", split=0.0)
        r = assign_for_request("any-id", None, cfg)
        assert r.model_path == "models/p"
        assert r.use_challenger is False


class TestABAssignmentDataclass:
    def test_fields_accessible(self):
        a = ABAssignment(use_challenger=True, model_path="models/v2", model_version_label="challenger")
        assert a.use_challenger is True
        assert a.model_path == "models/v2"
        assert a.model_version_label == "challenger"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
