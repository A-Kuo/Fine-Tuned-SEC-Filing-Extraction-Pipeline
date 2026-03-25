"""Tests for the circuit breaker state machine."""

import time
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreakerInitialState:
    def test_starts_closed(self):
        cb = CircuitBreaker("svc")
        assert cb.state == CircuitState.CLOSED

    def test_allows_requests_when_closed(self):
        cb = CircuitBreaker("svc")
        assert cb.allow() is True


class TestCircuitBreakerFailures:
    def test_opens_after_threshold(self):
        cb = CircuitBreaker("svc", failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_does_not_open_before_threshold(self):
        cb = CircuitBreaker("svc", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_blocks_requests_when_open(self):
        cb = CircuitBreaker("svc", failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow() is False

    def test_failure_in_half_open_reopens(self):
        cb = CircuitBreaker("svc", failure_threshold=1, reset_timeout_s=0.0)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # Timeout expires — next allow() transitions to HALF_OPEN
        assert cb.allow() is True
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerRecovery:
    def test_resets_to_closed_on_success(self):
        cb = CircuitBreaker("svc", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # Force half-open by manipulating timeout
        cb._last_failure_time = 0.0
        assert cb.allow() is True
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_success_resets_failure_counter(self):
        cb = CircuitBreaker("svc", failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # After success, need full threshold again to open
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_reset_timeout(self):
        cb = CircuitBreaker("svc", failure_threshold=1, reset_timeout_s=0.0)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # With 0s timeout, next allow() should move to HALF_OPEN
        allowed = cb.allow()
        assert allowed is True
        assert cb.state == CircuitState.HALF_OPEN


class TestCircuitBreakerDegradedResponse:
    def test_degraded_response_no_cache(self):
        cb = CircuitBreaker("svc")
        d = cb.degraded_response()
        assert d["service"] == "svc"
        assert d["cached"] is None

    def test_degraded_response_caches_last_success(self):
        cb = CircuitBreaker("svc")
        cb.record_success(cached_response={"ticker": "AAPL"})
        d = cb.degraded_response()
        assert d["cached"] == {"ticker": "AAPL"}

    def test_cached_response_updates_on_new_success(self):
        cb = CircuitBreaker("svc")
        cb.record_success(cached_response={"ticker": "AAPL"})
        cb.record_success(cached_response={"ticker": "MSFT"})
        assert cb.degraded_response()["cached"]["ticker"] == "MSFT"


class TestCircuitBreakerStateReporting:
    def test_state_value_strings(self):
        cb = CircuitBreaker("svc", failure_threshold=1)
        assert cb.state.value == "closed"
        cb.record_failure()
        assert cb.state.value == "open"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
