"""AIGP-HARDEN-1 / AIGP-NEW-01: the HALF_OPEN trial can never stay stuck.

Deterministic: an injected clock, no real delay.
"""

from __future__ import annotations

import random

from ai_gateway_pool.nodes.availability import AvailabilityTracker, BreakerSettings, BreakerState

N = "node"


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def tracker(
    threshold: int = 1, open_s: float = 10, max_s: float = 40
) -> tuple[AvailabilityTracker, Clock]:
    clock = Clock()
    t = AvailabilityTracker(
        BreakerSettings(failure_threshold=threshold, open_seconds=open_s, max_open_seconds=max_s),
        clock=clock,
    )
    return t, clock


def half_open(t: AvailabilityTracker, clock: Clock) -> None:
    trial = t.begin(N)
    t.record_transport_failure(N, "connect_error")
    t.end(N, trial=trial)
    assert t.breaker_state(N) is BreakerState.OPEN and not t.admits(N)
    clock.t += 10.01
    assert t.breaker_state(N) is BreakerState.HALF_OPEN and t.admits(N)


def test_single_trial_then_neutral_end_releases_it() -> None:
    t, clock = tracker()
    half_open(t, clock)
    assert t.begin(N) is True  # this request IS the trial
    assert not t.admits(N)  # exactly one trial at a time
    t.end(N, trial=True)  # neutral outcome: no success, no transport failure
    assert t.breaker_state(N) is BreakerState.HALF_OPEN
    assert t.admits(N)  # the next request may try again (was: stuck for good)


def test_neutral_end_never_needs_a_restart_whatever_the_number() -> None:
    t, clock = tracker()
    half_open(t, clock)
    for _ in range(100):
        trial = t.begin(N)
        assert trial is True
        t.end(N, trial=trial)
        assert t.admits(N)
    trial = t.begin(N)
    t.record_success(N)
    t.end(N, trial=trial)
    assert t.breaker_state(N) is BreakerState.CLOSED


def test_trial_failure_reopens_with_doubled_period_then_recovers() -> None:
    t, clock = tracker()
    half_open(t, clock)
    trial = t.begin(N)
    t.record_transport_failure(N, "connect_error")
    t.end(N, trial=trial)
    assert t.breaker_state(N) is BreakerState.OPEN
    clock.t += 10.01
    assert t.breaker_state(N) is BreakerState.OPEN  # period doubled to 20 s
    clock.t += 10.0
    assert t.breaker_state(N) is BreakerState.HALF_OPEN
    trial = t.begin(N)
    t.record_success(N)
    t.end(N, trial=trial)
    assert t.breaker_state(N) is BreakerState.CLOSED and t.admits(N)
    # a new failure after recovery starts from the base period again
    t.record_transport_failure(N, "connect_error")
    assert t.breaker_state(N) is BreakerState.OPEN
    clock.t += 10.01
    assert t.breaker_state(N) is BreakerState.HALF_OPEN


def test_non_trial_request_cannot_release_the_pending_trial() -> None:
    """A long request started while CLOSED ends during HALF_OPEN: it must not
    free the slot of the in-flight trial (one trial at a time)."""

    t, clock = tracker(threshold=2)
    old = t.begin(N)  # CLOSED: not a trial
    assert old is False
    t.record_transport_failure(N, "x")
    t.record_transport_failure(N, "x")
    clock.t += 10.01
    trial = t.begin(N)
    assert trial is True and not t.admits(N)
    t.end(N, trial=old)
    assert not t.admits(N)  # still exactly one trial pending
    t.end(N, trial=trial)
    assert t.admits(N)


def test_probe_success_closes_a_half_open_breaker_without_client_traffic() -> None:
    t, clock = tracker()
    half_open(t, clock)
    t.record_probe_success(N)
    assert t.breaker_state(N) is BreakerState.CLOSED


def test_probe_never_shortens_an_open_period_nor_overrides_a_pending_trial() -> None:
    t, clock = tracker()
    trial = t.begin(N)
    t.record_transport_failure(N, "x")
    t.end(N, trial=trial)
    t.record_probe_success(N)
    assert t.breaker_state(N) is BreakerState.OPEN  # backoff kept
    clock.t += 10.01
    trial = t.begin(N)
    t.record_probe_success(N)
    assert t.breaker_state(N) is BreakerState.HALF_OPEN  # the trial decides
    t.record_transport_failure(N, "x")
    t.end(N, trial=trial)
    assert t.breaker_state(N) is BreakerState.OPEN


def test_in_flight_counter_balanced_on_every_path() -> None:
    t, clock = tracker()
    half_open(t, clock)
    for verdict in ("success", "failure", "neutral"):
        clock.t += 100
        t.breaker_state(N)
        trial = t.begin(N)
        if verdict == "success":
            t.record_success(N)
        elif verdict == "failure":
            t.record_transport_failure(N, "x")
        t.end(N, trial=trial)
    assert t.in_flight(N) == 0


def test_random_sequences_never_leave_an_unowned_pending_trial() -> None:
    """Model check: whatever the interleaving of begin / verdicts / end / probes
    / time, once every request ended no trial is left pending, and the node is
    admitted again as soon as the breaker is not OPEN."""

    rng = random.Random(20261108)  # noqa: S311 - deterministic model check, not crypto
    for _ in range(300):
        t, clock = tracker(threshold=rng.randint(1, 3))
        live: list[bool] = []
        for _ in range(60):
            op = rng.choice(["begin", "end", "fail", "ok", "tick", "probe"])
            if op == "begin" and t.admits(N):
                live.append(t.begin(N))
            elif op == "end" and live:
                t.end(N, trial=live.pop(rng.randrange(len(live))))
            elif op == "fail" and live:
                t.record_transport_failure(N, "x")
                t.end(N, trial=live.pop(rng.randrange(len(live))))
            elif op == "ok" and live:
                t.record_success(N)
                t.end(N, trial=live.pop(rng.randrange(len(live))))
            elif op == "tick":
                clock.t += rng.choice([1, 5, 11, 50])
            elif op == "probe":
                t.record_probe_success(N)
        for trial in live:
            t.end(N, trial=trial)
        assert t.in_flight(N) == 0
        clock.t += 10_000
        assert t.breaker_state(N) is not BreakerState.OPEN
        assert t.admits(N), "a node with no request in flight must be admitted again"
