"""Simple circuit breaker for downstream HTTP services."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Open circuit after `failure_threshold` consecutive failures; retry after `reset_timeout_s`."""

    name: str
    failure_threshold: int = 5
    reset_timeout_s: float = 60.0
    _failures: int = 0
    _last_failure_time: float = 0.0
    _state: CircuitState = field(default=CircuitState.CLOSED)
    _last_success_response: object | None = None

    def allow(self) -> bool:
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.reset_timeout_s:
                self._state = CircuitState.HALF_OPEN
                return True
            return False
        # HALF_OPEN: allow one trial
        return True

    def record_success(self, cached_response: object | None = None) -> None:
        self._failures = 0
        self._state = CircuitState.CLOSED
        if cached_response is not None:
            self._last_success_response = cached_response

    def record_failure(self) -> None:
        self._failures += 1
        self._last_failure_time = time.monotonic()
        if self._failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
        elif self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN

    @property
    def state(self) -> CircuitState:
        return self._state

    def degraded_response(self) -> dict:
        return {
            "service": self.name,
            "state": self._state.value,
            "cached": self._last_success_response,
        }


def check_http_health(url: str, timeout: float = 5.0) -> bool:
    """Return True if GET url returns 2xx."""
    try:
        import httpx

        r = httpx.get(url, timeout=timeout)
        return 200 <= r.status_code < 300
    except Exception:
        return False
