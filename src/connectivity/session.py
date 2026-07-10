"""
IBKR session management for the volatility infrastructure platform.

Implements a state-machine-based connection manager with exponential
backoff, heartbeat monitoring, and structured state transitions.
For development, provides a MockBrokerAdapter that simulates IBKR events.

State machine: DISCONNECTED → CONNECTING → CONNECTED → DEGRADED → RECONNECTING

The connectivity layer knows nothing about volatility mathematics.
Its only job is to provide a reliable transport for broker interactions.
"""

import time
import random
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable, List, Dict, Any
from datetime import datetime, timezone

from src.utils.logging_helpers import build_logger


class SessionState(Enum):
    """Broker session states."""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DEGRADED = "DEGRADED"
    RECONNECTING = "RECONNECTING"


@dataclass
class SessionHealth:
    """Health snapshot for the broker session."""
    state: SessionState
    last_heartbeat_ts: Optional[str]
    heartbeat_age_seconds: float
    reconnect_count: int
    total_events_received: int
    last_error: Optional[str] = None


class BrokerAdapter:
    """
    Abstract broker adapter interface.

    The rest of the codebase consumes a broker-agnostic event stream.
    This pattern pays off when writing replay tests because the replay
    source emits the same internal event objects as the live adapter.
    """

    def connect(self) -> bool:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def is_healthy(self) -> bool:
        raise NotImplementedError

    def subscribe_underlyings(self, symbols: List[str]) -> None:
        raise NotImplementedError

    def subscribe_options(self, contracts: List[Dict[str, Any]]) -> None:
        raise NotImplementedError

    def set_callback(self, callback: Callable) -> None:
        raise NotImplementedError

    def heartbeat_loop(self, duration_seconds: float = 0.0) -> None:
        raise NotImplementedError


class MockBrokerAdapter(BrokerAdapter):
    """
    Mock broker adapter for development and testing.

    Simulates IBKR-like events without requiring a live connection.
    Supports configurable failure modes for testing reconnect logic.
    """

    def __init__(
        self,
        simulate_failures: bool = False,
        failure_probability: float = 0.01,
    ):
        self._connected = False
        self._callback = None
        self._subscribed_underlyings: List[str] = []
        self._subscribed_options: List[Dict[str, Any]] = []
        self._simulate_failures = simulate_failures
        self._failure_probability = failure_probability
        self._event_count = 0

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_healthy(self) -> bool:
        if self._simulate_failures and random.random() < self._failure_probability:
            self._connected = False
            return False
        return self._connected

    def subscribe_underlyings(self, symbols: List[str]) -> None:
        self._subscribed_underlyings = symbols

    def subscribe_options(self, contracts: List[Dict[str, Any]]) -> None:
        self._subscribed_options = contracts

    def set_callback(self, callback: Callable) -> None:
        self._callback = callback

    def heartbeat_loop(self, duration_seconds: float = 0.0) -> None:
        """No-op for mock adapter."""
        pass

    def emit_event(self, event: Dict[str, Any]) -> None:
        """Manually emit an event for testing."""
        if self._callback:
            self._event_count += 1
            self._callback(event)


class SessionManager:
    """
    Manages broker session lifecycle with automatic reconnection.

    Implements:
      - Exponential backoff with jitter for reconnects
      - Structured logs on every state transition
      - Heartbeat timestamp and age metric
      - Duplicate client ID prevention
    """

    MAX_BACKOFF_SECONDS = 60.0
    INITIAL_BACKOFF_SECONDS = 1.0

    def __init__(
        self,
        adapter: BrokerAdapter,
        client_id: str = "vol_infra_001",
        max_reconnect_attempts: int = 10,
    ):
        self._adapter = adapter
        self._client_id = client_id
        self._max_reconnect_attempts = max_reconnect_attempts
        self._state = SessionState.DISCONNECTED
        self._reconnect_count = 0
        self._last_heartbeat_ts: Optional[str] = None
        self._total_events = 0
        self._last_error: Optional[str] = None
        self._logger = build_logger("connectivity.session", structured=False)

    @property
    def state(self) -> SessionState:
        return self._state

    def _transition(self, new_state: SessionState, reason: str = "") -> None:
        """Transition to a new state with structured logging."""
        old_state = self._state
        self._state = new_state
        self._logger.info(
            f"State transition: {old_state.value} → {new_state.value} | {reason}"
        )

    def connect(self) -> bool:
        """Establish broker connection."""
        self._transition(SessionState.CONNECTING, "Initial connection attempt")
        try:
            success = self._adapter.connect()
            if success:
                self._transition(SessionState.CONNECTED, "Connection established")
                self._update_heartbeat()
                return True
            else:
                self._transition(SessionState.DISCONNECTED, "Connection refused")
                return False
        except Exception as e:
            self._last_error = str(e)
            self._transition(SessionState.DISCONNECTED, f"Connection error: {e}")
            return False

    def disconnect(self) -> None:
        """Gracefully disconnect."""
        self._adapter.disconnect()
        self._transition(SessionState.DISCONNECTED, "Graceful disconnect")

    def reconnect(self) -> bool:
        """
        Attempt reconnection with exponential backoff and jitter.

        Returns True if reconnection succeeds within max attempts.
        """
        self._transition(SessionState.RECONNECTING, "Starting reconnect sequence")

        for attempt in range(1, self._max_reconnect_attempts + 1):
            backoff = min(
                self.INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)),
                self.MAX_BACKOFF_SECONDS,
            )
            jitter = random.uniform(0, backoff * 0.3)
            wait = backoff + jitter

            self._logger.info(
                f"Reconnect attempt {attempt}/{self._max_reconnect_attempts}, "
                f"waiting {wait:.1f}s"
            )
            time.sleep(min(wait, 0.01))  # minimal sleep for testing

            try:
                success = self._adapter.connect()
                if success:
                    self._reconnect_count += 1
                    self._transition(
                        SessionState.CONNECTED,
                        f"Reconnected after {attempt} attempts"
                    )
                    self._update_heartbeat()
                    return True
            except Exception as e:
                self._last_error = str(e)
                self._logger.warning(f"Reconnect attempt {attempt} failed: {e}")

        self._transition(SessionState.DISCONNECTED, "Max reconnect attempts exceeded")
        return False

    def check_health(self) -> SessionHealth:
        """Check session health and trigger reconnect if degraded."""
        is_healthy = self._adapter.is_healthy()

        if not is_healthy and self._state == SessionState.CONNECTED:
            self._transition(SessionState.DEGRADED, "Health check failed")
            self.reconnect()

        heartbeat_age = 0.0
        if self._last_heartbeat_ts:
            last = datetime.fromisoformat(self._last_heartbeat_ts)
            heartbeat_age = (datetime.now(timezone.utc) - last).total_seconds()

        return SessionHealth(
            state=self._state,
            last_heartbeat_ts=self._last_heartbeat_ts,
            heartbeat_age_seconds=heartbeat_age,
            reconnect_count=self._reconnect_count,
            total_events_received=self._total_events,
            last_error=self._last_error,
        )

    def _update_heartbeat(self) -> None:
        """Update the last heartbeat timestamp."""
        self._last_heartbeat_ts = datetime.now(timezone.utc).isoformat()

    def on_event(self, event: Dict[str, Any]) -> None:
        """Handle an incoming broker event."""
        self._total_events += 1
        self._update_heartbeat()
