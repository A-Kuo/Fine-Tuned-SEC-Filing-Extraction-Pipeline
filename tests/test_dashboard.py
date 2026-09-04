"""Tests for monitoring/dashboard.py's data-source tagging.

Regression coverage for a real bug: _load_dashboard_data_cached() used to
silently return generate_demo_data()'s fabricated numbers on ANY exception
(DB down, empty tables, connection refused -- anything), with no way for the
caller or the viewer to tell the numbers on screen weren't real. It now
returns a data_source tag ("live"/"demo") the UI uses to render a persistent
warning banner instead of rendering demo data as if it were live.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.dashboard import _load_dashboard_data_cached, generate_demo_data


class TestDataSourceTagging:
    def test_db_unavailable_tags_as_demo(self):
        with patch("src.storage.database.DatabaseManager.from_config", side_effect=Exception("no db")):
            result = _load_dashboard_data_cached.__wrapped__(days=30)
        *_, data_source = result
        assert data_source == "demo"

    def test_demo_result_still_has_the_expected_shape(self):
        with patch("src.storage.database.DatabaseManager.from_config", side_effect=Exception("no db")):
            result = _load_dashboard_data_cached.__wrapped__(days=30)
        accuracy_history, latencies, statuses, cache_stats, data_source = result
        assert isinstance(accuracy_history, list)
        assert isinstance(latencies, list)
        assert isinstance(cache_stats, dict)

    def test_live_db_tags_as_live(self):
        mock_db = MagicMock()
        mock_db.get_daily_extraction_counts.return_value = [{"date": "2026-01-01", "accuracy": 0.9, "sample_size": 10}]
        mock_db.get_stats.return_value = {"storage": {}}
        mock_db.get_recent_extraction_logs.return_value = [{"status": "success", "latency_ms": 300}]
        mock_db.cache.get_stats.return_value = {"available": True, "hit_rate": 0.5, "hits": 5, "misses": 5, "used_memory_mb": 1}

        with patch("src.storage.database.DatabaseManager.from_config", return_value=mock_db):
            result = _load_dashboard_data_cached.__wrapped__(days=30)

        *_, data_source = result
        assert data_source == "live"

    def test_empty_history_from_live_db_falls_back_to_demo(self):
        """get_daily_extraction_counts() returning [] (e.g. empty table, not
        a connection failure) must ALSO be treated as "not real data", not
        silently rendered as a live-but-empty chart."""
        mock_db = MagicMock()
        mock_db.get_daily_extraction_counts.return_value = []

        with patch("src.storage.database.DatabaseManager.from_config", return_value=mock_db):
            result = _load_dashboard_data_cached.__wrapped__(days=30)

        *_, data_source = result
        assert data_source == "demo"


class TestGenerateDemoData:
    def test_returns_four_tuple(self):
        result = generate_demo_data()
        assert len(result) == 4

    def test_deterministic(self):
        """seed(42) means repeated calls produce identical demo data --
        important so the "this is fake" banner corresponds to numbers a
        developer can reproduce exactly when debugging the dashboard."""
        r1 = generate_demo_data()
        r2 = generate_demo_data()
        assert r1[0] == r2[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
