"""Tests for database layer.

Tests the financial value parsing, cache logic, and storage operations.
Uses mocks so tests run without Docker/Redis/PostgreSQL.
"""

import json
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import RedisCache, PostgresStorage, DatabaseManager
from src.postprocessing import ExtractionResult


# ─── Financial Value Parsing ─────────────────────────────────────────────────

class TestFinancialParsing:
    """Test _parse_financial which converts '$383.3 billion' → 383300000000.0"""

    def test_billions(self):
        assert PostgresStorage._parse_financial("$383.3 billion") == pytest.approx(383.3e9)

    def test_millions(self):
        assert PostgresStorage._parse_financial("$12.1 million") == pytest.approx(12.1e6)

    def test_trillions(self):
        assert PostgresStorage._parse_financial("$1.1 trillion") == pytest.approx(1.1e12)

    def test_plain_number(self):
        assert PostgresStorage._parse_financial("$5.23") == pytest.approx(5.23)

    def test_with_commas(self):
        assert PostgresStorage._parse_financial("$1,234,567") == pytest.approx(1234567.0)

    def test_none_input(self):
        assert PostgresStorage._parse_financial(None) is None

    def test_empty_string(self):
        assert PostgresStorage._parse_financial("") is None

    def test_no_digits(self):
        assert PostgresStorage._parse_financial("not a number") is None

    def test_negative_notation(self):
        # Should still extract the numeric part
        result = PostgresStorage._parse_financial("$-12.5 billion")
        assert result == pytest.approx(12.5e9)  # Extracts magnitude


# ─── Redis Cache Logic ───────────────────────────────────────────────────────

class TestRedisCache:
    def test_unavailable_get_returns_none(self):
        """When Redis is down, get() returns None (graceful degradation)."""
        cache = RedisCache("localhost", 6379)
        cache._available = False
        assert cache.get("test-id") is None

    def test_unavailable_set_returns_false(self):
        cache = RedisCache("localhost", 6379)
        cache._available = False
        assert cache.set("test-id", {"data": "test"}) is False

    def test_get_set_with_mock(self):
        """Test get/set with a mocked Redis client."""
        cache = RedisCache("localhost", 6379)
        cache._available = True
        cache._client = MagicMock()

        # Test set
        cache._client.setex = MagicMock(return_value=True)
        assert cache.set("filing-123", {"company_name": "Apple"}) is True
        cache._client.setex.assert_called_once()

        # Test get (cache hit)
        cache._client.get = MagicMock(return_value='{"company_name": "Apple"}')
        result = cache.get("filing-123")
        assert result == {"company_name": "Apple"}

        # Test get (cache miss)
        cache._client.get = MagicMock(return_value=None)
        assert cache.get("nonexistent") is None

    def test_delete_with_mock(self):
        cache = RedisCache("localhost", 6379)
        cache._available = True
        cache._client = MagicMock()
        cache._client.delete = MagicMock(return_value=1)
        assert cache.delete("filing-123") is True

    def test_stats_unavailable(self):
        cache = RedisCache("localhost", 6379)
        cache._available = False
        stats = cache.get_stats()
        assert stats["available"] is False


# ─── DatabaseManager Read-Through Caching ────────────────────────────────────

class TestDatabaseManager:
    def _make_manager(self) -> DatabaseManager:
        """Create a DatabaseManager with mocked backends."""
        cache = RedisCache("localhost", 6379)
        cache._available = True
        cache._client = MagicMock()

        storage = PostgresStorage("localhost", 5432, "user", "pass", "db")
        storage._available = True
        storage._connection = MagicMock()

        return DatabaseManager(cache, storage)

    def test_get_cache_hit(self):
        """Read-through: cache hit → return from Redis, don't hit PostgreSQL."""
        mgr = self._make_manager()

        cached_data = {"company_name": "Apple", "filing_type": "10-K"}
        mgr.cache._client.get = MagicMock(return_value=json.dumps(cached_data))

        result = mgr.get_extraction("filing-123")
        assert result == cached_data

        # PostgreSQL should NOT have been queried
        mgr.storage._connection.cursor.assert_not_called()

    def test_get_cache_miss_db_hit(self):
        """Read-through: cache miss → query PostgreSQL → cache result."""
        mgr = self._make_manager()

        # Cache miss
        mgr.cache._client.get = MagicMock(return_value=None)

        # PostgreSQL hit
        mock_cursor = MagicMock()
        mock_cursor.fetchone = MagicMock(return_value=(
            "filing-123", "Apple Inc.", "10-K", "2023-11-03",
            383.3e9, 90e9, 500e9, 200e9, 5.23,
            0.96, 421, "v1",
        ))
        mgr.storage._connection.cursor = MagicMock(return_value=mock_cursor)

        result = mgr.get_extraction("filing-123")
        assert result is not None
        assert result["company_name"] == "Apple Inc."

        # Verify result was cached
        mgr.cache._client.setex.assert_called_once()

    def test_get_both_miss(self):
        """Read-through: both miss → return None."""
        mgr = self._make_manager()

        mgr.cache._client.get = MagicMock(return_value=None)

        mock_cursor = MagicMock()
        mock_cursor.fetchone = MagicMock(return_value=None)
        mgr.storage._connection.cursor = MagicMock(return_value=mock_cursor)

        result = mgr.get_extraction("nonexistent")
        assert result is None

    def test_store_writes_both_tiers(self):
        """Write-through: store writes to both PostgreSQL and Redis."""
        mgr = self._make_manager()

        mock_cursor = MagicMock()
        mgr.storage._connection.cursor = MagicMock(return_value=mock_cursor)
        mgr.cache._client.setex = MagicMock(return_value=True)

        result = ExtractionResult(
            filing_id="filing-123",
            company_name="Apple Inc.",
            filing_type="10-K",
            date="2023-11-03",
        )

        mgr.store_extraction(
            filing_id="filing-123",
            result=result,
            confidence=0.96,
            latency_ms=421,
            model_version="v1",
        )

        # Both backends should have been written to
        assert mock_cursor.execute.call_count >= 2  # INSERT extractions + INSERT logs
        mgr.cache._client.setex.assert_called_once()


# ─── Graceful Degradation ────────────────────────────────────────────────────

class TestGracefulDegradation:
    def test_no_redis_still_works(self):
        """System works without Redis (just slower reads)."""
        cache = RedisCache("localhost", 6379)
        cache._available = False  # Redis down

        storage = PostgresStorage("localhost", 5432, "user", "pass", "db")
        storage._available = True
        storage._connection = MagicMock()

        mock_cursor = MagicMock()
        mock_cursor.fetchone = MagicMock(return_value=(
            "filing-123", "Apple", "10-K", "2023-11-03",
            None, None, None, None, None, 0.9, 400, "v1",
        ))
        storage._connection.cursor = MagicMock(return_value=mock_cursor)

        mgr = DatabaseManager(cache, storage)
        result = mgr.get_extraction("filing-123")
        assert result is not None

    def test_no_postgres_still_caches(self):
        """Without PostgreSQL, cache still serves reads."""
        cache = RedisCache("localhost", 6379)
        cache._available = True
        cache._client = MagicMock()
        cache._client.get = MagicMock(return_value='{"company_name": "Apple"}')

        storage = PostgresStorage("localhost", 5432, "user", "pass", "db")
        storage._available = False  # PostgreSQL down

        mgr = DatabaseManager(cache, storage)
        result = mgr.get_extraction("filing-123")
        assert result == {"company_name": "Apple"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
