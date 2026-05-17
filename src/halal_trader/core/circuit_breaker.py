"""Adapter circuit breaker state machine.

Auxiliary primitive complementing the bot's resilience layer.
Broker / screener / LLM-provider adapters call out over the
network; when an upstream is failing (rate-limited, down, slow),
the bot needs to stop hammering it after N consecutive failures
and let it recover before retrying. The classic three-state
circuit breaker pattern: CLOSED (normal), OPEN (rejecting all
calls), HALF_OPEN (probing one or two calls to see if the upstream
recovered).

Picked a pure-Python snapshot-based engine over a stateful object
with internal timers because (a) the bot is single-threaded async
on the cycle path; passing `now` explicitly makes time-based
transitions trivially testable without freezegun / sleep / clocks;
(b) the snapshot can be persisted to DB across restarts (a 5-minute
cooldown that started 30s before a crash should still have ~4.5
minutes left after recovery — pure functions over snapshots make
this work for free); (c) operators can read the current state +
counters without grabbing a lock or interrupting the adapter.

Pinned semantics:
- **Forward state machine: CLOSED → OPEN → HALF_OPEN → {CLOSED |
  OPEN}.** The only way back to CLOSED is via successful HALF_OPEN
  probes. CLOSED never goes directly to HALF_OPEN.
- **OPEN → HALF_OPEN is time-driven (cooldown_seconds).** The
  cooldown timer starts when the breaker opens; `tick(now)`
  transitions to HALF_OPEN when `now - opened_at >= cooldown`.
  Boundary inclusive at the cooldown.
- **HALF_OPEN → CLOSED requires `half_open_probe_count` consecutive
  successes.** A single failure in HALF_OPEN re-opens the breaker
  (with a fresh cooldown).
- **`is_call_allowed` is the load-bearing gate.** Returns True for
  CLOSED + HALF_OPEN; False only for OPEN. Adapters consult this
  BEFORE making the actual call.
- **Render output never includes the underlying call's arguments
  or response.** Only the breaker's state + counters; the
  adapter-side logger handles the raw API call.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import Enum


class BreakerState(str, Enum):
    """Circuit breaker state.

    Pinned string values for JSON / DB persistence stability.
    Forward-only: CLOSED → OPEN → HALF_OPEN → {CLOSED | OPEN}.
    """

    CLOSED = "closed"  # Normal operation; calls pass through
    OPEN = "open"  # Rejecting all calls; cooldown in progress
    HALF_OPEN = "half_open"  # Probing; limited calls to test recovery


class CallOutcome(str, Enum):
    """Outcome of an adapter call.

    Pinned string values. SUCCESS / FAILURE.
    """

    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class BreakerPolicy:
    """Operator-tunable circuit breaker policy.

    Defaults: 5 consecutive failures → OPEN; 60s cooldown; 2
    consecutive HALF_OPEN successes → CLOSED. The defaults are
    tuned for "transient network blip" (a 60s cooldown is enough
    for a rate-limit window to clear) and "real outage" (5
    consecutive failures is past the noise floor).
    """

    failure_threshold: int = 5
    cooldown_seconds: float = 60.0
    half_open_probe_count: int = 2

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be > 0")
        if self.half_open_probe_count < 1:
            raise ValueError("half_open_probe_count must be >= 1")


@dataclass(frozen=True)
class BreakerSnapshot:
    """Persistable circuit breaker state snapshot.

    `consecutive_failures` counts CLOSED-state failures since the
    last success; resets to 0 on a CLOSED success or a CLOSED →
    HALF_OPEN → CLOSED cycle. `opened_at` is set when the breaker
    transitions to OPEN; consulted by `tick` to compute cooldown
    expiry. `half_open_successes` counts HALF_OPEN probe successes;
    resets to 0 on every state transition.

    The snapshot is persistable: same fields → same future behavior
    (assuming same policy + clock).
    """

    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    opened_at: datetime | None = None
    half_open_successes: int = 0

    def __post_init__(self) -> None:
        if self.consecutive_failures < 0:
            raise ValueError("consecutive_failures must be non-negative")
        if self.half_open_successes < 0:
            raise ValueError("half_open_successes must be non-negative")
        if self.state is BreakerState.OPEN and self.opened_at is None:
            raise ValueError("OPEN state requires opened_at")
        if self.opened_at is not None and self.opened_at.tzinfo is None:
            raise ValueError("opened_at must be timezone-aware")


def is_call_allowed(snapshot: BreakerSnapshot) -> bool:
    """Whether the breaker permits a call right now.

    CLOSED and HALF_OPEN allow calls; OPEN rejects.
    Load-bearing gate the adapter consults before making the call.
    """

    return snapshot.state is not BreakerState.OPEN


def record_outcome(
    snapshot: BreakerSnapshot,
    outcome: CallOutcome,
    *,
    now: datetime,
    policy: BreakerPolicy = BreakerPolicy(),
) -> BreakerSnapshot:
    """Update snapshot for a call outcome.

    Transitions:
    - CLOSED + SUCCESS → CLOSED with consecutive_failures=0
    - CLOSED + FAILURE → CLOSED++; if >= threshold → OPEN(now)
    - OPEN + ... → no-op (caller should have respected
      is_call_allowed; defensive identity)
    - HALF_OPEN + SUCCESS → HALF_OPEN++; if >= probe_count → CLOSED
    - HALF_OPEN + FAILURE → OPEN(now), reset probe count
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    if snapshot.state is BreakerState.CLOSED:
        if outcome is CallOutcome.SUCCESS:
            return replace(snapshot, consecutive_failures=0)
        new_failures = snapshot.consecutive_failures + 1
        if new_failures >= policy.failure_threshold:
            return BreakerSnapshot(
                state=BreakerState.OPEN,
                consecutive_failures=new_failures,
                opened_at=now,
                half_open_successes=0,
            )
        return replace(snapshot, consecutive_failures=new_failures)

    if snapshot.state is BreakerState.OPEN:
        # Defensive: caller should have respected is_call_allowed.
        # We don't punish the breaker for a stray outcome.
        return snapshot

    # HALF_OPEN
    if outcome is CallOutcome.SUCCESS:
        new_successes = snapshot.half_open_successes + 1
        if new_successes >= policy.half_open_probe_count:
            return BreakerSnapshot(
                state=BreakerState.CLOSED,
                consecutive_failures=0,
                opened_at=None,
                half_open_successes=0,
            )
        return replace(snapshot, half_open_successes=new_successes)

    # HALF_OPEN + FAILURE → OPEN with fresh cooldown
    return BreakerSnapshot(
        state=BreakerState.OPEN,
        consecutive_failures=snapshot.consecutive_failures,
        opened_at=now,
        half_open_successes=0,
    )


def tick(
    snapshot: BreakerSnapshot,
    *,
    now: datetime,
    policy: BreakerPolicy = BreakerPolicy(),
) -> BreakerSnapshot:
    """Time-driven transition: OPEN → HALF_OPEN once cooldown elapsed.

    Boundary inclusive: `now - opened_at >= cooldown` triggers the
    transition. CLOSED + HALF_OPEN are no-ops (their transitions
    are outcome-driven via `record_outcome`).
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    if snapshot.state is not BreakerState.OPEN:
        return snapshot
    assert snapshot.opened_at is not None  # invariant from __post_init__
    elapsed = now - snapshot.opened_at
    cooldown = timedelta(seconds=policy.cooldown_seconds)
    if elapsed >= cooldown:
        return BreakerSnapshot(
            state=BreakerState.HALF_OPEN,
            consecutive_failures=snapshot.consecutive_failures,
            opened_at=None,
            half_open_successes=0,
        )
    return snapshot


def time_until_retry(
    snapshot: BreakerSnapshot,
    *,
    now: datetime,
    policy: BreakerPolicy = BreakerPolicy(),
) -> timedelta:
    """How long until the breaker enters HALF_OPEN.

    Returns timedelta(0) if not in OPEN state or cooldown elapsed.
    Operator surface: dashboard tile shows "retrying in 23s".
    """

    if snapshot.state is not BreakerState.OPEN:
        return timedelta(0)
    assert snapshot.opened_at is not None
    cooldown = timedelta(seconds=policy.cooldown_seconds)
    remaining = (snapshot.opened_at + cooldown) - now
    if remaining.total_seconds() <= 0:
        return timedelta(0)
    return remaining


_STATE_EMOJI: dict[BreakerState, str] = {
    BreakerState.CLOSED: "🟢",
    BreakerState.OPEN: "🔴",
    BreakerState.HALF_OPEN: "🟡",
}


def render_snapshot(
    snapshot: BreakerSnapshot,
    *,
    name: str,
    now: datetime | None = None,
    policy: BreakerPolicy = BreakerPolicy(),
) -> str:
    """Format breaker state for ops display.

    No-secret-leak: shows only state + counters + retry ETA. The
    adapter-side logger handles raw API call details.
    """

    emoji = _STATE_EMOJI[snapshot.state]
    parts = [f"{emoji} {name}: {snapshot.state.value}"]
    if snapshot.state is BreakerState.CLOSED:
        if snapshot.consecutive_failures > 0:
            parts.append(f"({snapshot.consecutive_failures} recent failures)")
    elif snapshot.state is BreakerState.OPEN:
        if now is not None:
            remaining = time_until_retry(snapshot, now=now, policy=policy)
            if remaining > timedelta(0):
                parts.append(f"(retry in {remaining.total_seconds():.0f}s)")
            else:
                parts.append("(ready to probe)")
    else:  # HALF_OPEN
        parts.append(f"({snapshot.half_open_successes}/{policy.half_open_probe_count} probes)")
    return " ".join(parts)


__all__ = [
    "BreakerPolicy",
    "BreakerSnapshot",
    "BreakerState",
    "CallOutcome",
    "is_call_allowed",
    "record_outcome",
    "render_snapshot",
    "tick",
    "time_until_retry",
]
