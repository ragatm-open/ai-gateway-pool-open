"""Per-node availability signals: circuit breaker, saturation hints, in-flight counts.

SCOPE: everything here is **local to one router process**. Two pool processes
(or uvicorn workers) each keep their own breaker states and in-flight counters;
nothing here is a distributed or GPU-level measurement. Shared, durable state
(job affinity) lives in Redis, not here.

Circuit breaker (per node)::

    CLOSED --(failure_threshold consecutive transport failures)--> OPEN
    OPEN   --(open_seconds elapsed)--> HALF_OPEN (exactly one trial request)
    HALF_OPEN --success--> CLOSED          (backoff reset)
    HALF_OPEN --failure--> OPEN            (open period doubled, capped)

Only TRANSPORT failures count (connection refused / connect timeout / connection
lost before any response byte). Any HTTP answer from the Gateway — including
4xx and ``503 resource_saturated`` — proves the node is reachable and closes the
breaker. Saturation is tracked separately as a short, Retry-After bounded hint.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum


class BreakerState(StrEnum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass(frozen=True)
class BreakerSettings:
    failure_threshold: int = 3
    open_seconds: float = 15.0
    max_open_seconds: float = 300.0
    # Upper bound for a saturation hint, whatever Retry-After says.
    saturation_max_seconds: float = 30.0


@dataclass
class _NodeAvailability:
    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    open_until: float = 0.0
    open_period: float = 0.0
    trial_in_flight: bool = False
    last_failure: str | None = None
    saturated_until: dict[str, float] = field(default_factory=dict)  # service -> monotonic
    in_flight: int = 0


class AvailabilityTracker:
    def __init__(
        self,
        settings: BreakerSettings | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings or BreakerSettings()
        self._clock = clock
        self._lock = threading.Lock()
        self._nodes: dict[str, _NodeAvailability] = {}

    def _get(self, node_id: str) -> _NodeAvailability:
        return self._nodes.setdefault(node_id, _NodeAvailability())

    # ------------------------------------------------------------------ breaker

    def breaker_state(self, node_id: str) -> BreakerState:
        with self._lock:
            n = self._get(node_id)
            if n.state is BreakerState.OPEN and self._clock() >= n.open_until:
                n.state = BreakerState.HALF_OPEN
                n.trial_in_flight = False
            return n.state

    def admits(self, node_id: str) -> bool:
        """Whether a NEW request may be sent to the node right now (no side effect)."""

        state = self.breaker_state(node_id)
        if state is BreakerState.OPEN:
            return False
        if state is BreakerState.HALF_OPEN:
            with self._lock:
                return not self._get(node_id).trial_in_flight
        return True

    def begin(self, node_id: str) -> bool:
        """Called when a request is actually dispatched to the node.

        Returns True when this request took the single HALF_OPEN trial: its
        caller MUST then pass ``trial=True`` to :meth:`end` (any exit path).
        """

        state = self.breaker_state(node_id)
        with self._lock:
            n = self._get(node_id)
            n.in_flight += 1
            if state is BreakerState.HALF_OPEN and not n.trial_in_flight:
                n.trial_in_flight = True
                return True
            return False

    def end(self, node_id: str, *, trial: bool = False) -> None:
        """Request finished. A trial that ended WITHOUT a verdict (no success and
        no transport failure recorded: client gone, malformed client body, local
        pool timeout, cancellation...) is released: the breaker stays HALF_OPEN
        and the next request may try again — never stuck until a restart."""

        with self._lock:
            n = self._get(node_id)
            n.in_flight = max(0, n.in_flight - 1)
            if trial and n.state is BreakerState.HALF_OPEN:
                n.trial_in_flight = False

    def record_probe_success(self, node_id: str) -> None:
        """A health probe got an HTTP answer: the node is reachable. Closes a
        breaker whose open period is over (HALF_OPEN) when no request trial is
        pending; never shortens an OPEN period (backoff kept)."""

        state = self.breaker_state(node_id)
        with self._lock:
            n = self._get(node_id)
            if state is BreakerState.HALF_OPEN and not n.trial_in_flight:
                n.state = BreakerState.CLOSED
                n.consecutive_failures = 0
                n.open_period = 0.0

    def record_success(self, node_id: str) -> None:
        with self._lock:
            n = self._get(node_id)
            n.state = BreakerState.CLOSED
            n.consecutive_failures = 0
            n.open_period = 0.0
            n.trial_in_flight = False

    def record_transport_failure(self, node_id: str, reason: str) -> BreakerState:
        with self._lock:
            n = self._get(node_id)
            n.last_failure = reason
            n.consecutive_failures += 1
            reopen = n.state is BreakerState.HALF_OPEN
            if reopen or n.consecutive_failures >= self.settings.failure_threshold:
                n.open_period = min(
                    self.settings.max_open_seconds,
                    (n.open_period * 2) if reopen and n.open_period else self.settings.open_seconds,
                )
                n.state = BreakerState.OPEN
                n.open_until = self._clock() + n.open_period
                n.trial_in_flight = False
            return n.state

    # ------------------------------------------------------------------ saturation

    def record_saturation(self, node_id: str, service: str, retry_after: float | None) -> None:
        seconds = min(
            self.settings.saturation_max_seconds,
            retry_after if retry_after and retry_after > 0 else 1.0,
        )
        with self._lock:
            self._get(node_id).saturated_until[service] = self._clock() + seconds

    def saturated(self, node_id: str, service: str) -> bool:
        with self._lock:
            until = self._get(node_id).saturated_until.get(service, 0.0)
            return self._clock() < until

    # ------------------------------------------------------------------ views

    def in_flight(self, node_id: str) -> int:
        with self._lock:
            return self._get(node_id).in_flight

    def snapshot(self, node_id: str) -> dict[str, object]:
        state = self.breaker_state(node_id)
        with self._lock:
            n = self._get(node_id)
            now = self._clock()
            return {
                "breaker_state": state.value,
                "consecutive_failures": n.consecutive_failures,
                "open_for_seconds": round(max(0.0, n.open_until - now), 1)
                if state is BreakerState.OPEN
                else 0.0,
                "last_transport_failure": n.last_failure,
                "saturated_services": sorted(s for s, u in n.saturated_until.items() if u > now),
                "in_flight_this_process": n.in_flight,
            }
