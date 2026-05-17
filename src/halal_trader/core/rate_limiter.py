"""Token bucket rate limiter (pure-functional snapshot).

Auxiliary primitive complementing the bot's resilience layer.
Broker / screener / LLM-provider adapters need to stay BELOW the
upstream's rate limit to avoid getting throttled in the first
place. The classic token bucket pattern: a bucket holds up to
`capacity` tokens; tokens refill at `refill_rate_per_sec`; each
call consumes N tokens (typically 1); calls that don't have
enough tokens are denied (with a hint about when the bucket will
have enough).

Picked a pure-functional snapshot-based engine over a stateful
object with internal timers because (a) the cycle path is
single-threaded async; passing `now` explicitly makes time-based
refills trivially testable without freezegun / sleep / clocks;
(b) snapshots are persistable to DB across restarts (a partially-
filled bucket from before a crash should still reflect the
in-flight refill state — pure functions over snapshots make this
work for free); (c) operators can read the current token count
without grabbing a lock or interrupting the adapter.

Distinct from `core/circuit_breaker.py`: the breaker reacts AFTER
a call fails (cascading-failure protection); the rate limiter
prevents the call BEFORE it would fail (proactive throttling).
The two compose: the limiter gates calls; if the limiter denies,
the breaker isn't even consulted.

Distinct from the private `_Bucket` in `web/research_api_keys.py`
(stateful, tied to the research API tier registry); this module
ships the public reusable primitive.

Pinned semantics:
- **Refill is monotonic.** Calling `refill` with a `now` earlier
  than `last_refill_at` is a programming error — rejected.
  Otherwise the bucket fills proportional to elapsed time, capped
  at `capacity`.
- **Consume is atomic refill+spend.** `try_consume(n, now)`
  refills first (advancing last_refill_at to now), then spends.
  If insufficient tokens, the snapshot is returned unchanged
  except for the refill; the caller sees `allowed=False` and a
  hint via `time_until_available`.
- **`n > capacity` always denied.** A request larger than the
  bucket can ever hold is a programming error from the caller
  (or a misconfigured policy); we return `allowed=False` with
  `time_until_available=infinity`-equivalent semantics rather
  than ever allowing it.
- **Tokens are float-valued.** Fractional refills are real (a
  0.5 tokens/sec rate ticking for 1.5 seconds adds 0.75 tokens).
  Consume costs are also float-valued for cost-weighted calls.
- **Render output never includes the underlying adapter call
  details.** Only bucket state + refill rate; the adapter-side
  logger handles raw API call args.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class ConsumeOutcome(str, Enum):
    """Outcome of a try_consume call.

    Pinned string values. ALLOWED / DENIED_INSUFFICIENT /
    DENIED_OVERSIZED.
    """

    ALLOWED = "allowed"  # Tokens consumed; call may proceed
    DENIED_INSUFFICIENT = "denied_insufficient"  # Not enough now; will be later
    DENIED_OVERSIZED = "denied_oversized"  # n > capacity; never allowed


@dataclass(frozen=True)
class BucketPolicy:
    """Operator-tunable token bucket policy.

    `capacity` is the maximum token count (also the burst size).
    `refill_rate_per_sec` is the steady-state token replenishment
    rate. Together they encode the upstream's published rate limit
    (e.g., Binance: 1200 weight per minute → capacity=1200,
    refill_rate_per_sec=20).
    """

    capacity: float
    refill_rate_per_sec: float

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        if self.refill_rate_per_sec <= 0:
            raise ValueError("refill_rate_per_sec must be > 0")
        if not math.isfinite(self.capacity):
            raise ValueError("capacity must be finite")
        if not math.isfinite(self.refill_rate_per_sec):
            raise ValueError("refill_rate_per_sec must be finite")


@dataclass(frozen=True)
class BucketSnapshot:
    """Persistable bucket state snapshot.

    `tokens` is the current token count (0 <= tokens <= capacity
    enforced by `refill` / `try_consume`). `last_refill_at` is
    the timestamp at which `tokens` was last computed; `refill`
    advances it.
    """

    tokens: float
    last_refill_at: datetime

    def __post_init__(self) -> None:
        if self.tokens < 0:
            raise ValueError("tokens must be non-negative")
        if not math.isfinite(self.tokens):
            raise ValueError("tokens must be finite")
        if self.last_refill_at.tzinfo is None:
            raise ValueError("last_refill_at must be timezone-aware")


def full_bucket(*, now: datetime, policy: BucketPolicy) -> BucketSnapshot:
    """Construct a fresh full bucket at `now`.

    Operator entry point — typically called when the limiter is
    first armed or after an explicit operator reset.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return BucketSnapshot(tokens=policy.capacity, last_refill_at=now)


def refill(
    snapshot: BucketSnapshot,
    *,
    now: datetime,
    policy: BucketPolicy,
) -> BucketSnapshot:
    """Advance the bucket to `now`, applying elapsed-time refill.

    Tokens are capped at `capacity`. `now` must be >=
    `snapshot.last_refill_at` (rejecting backwards-clock errors).
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if now < snapshot.last_refill_at:
        raise ValueError("now must be >= last_refill_at (clock went backwards)")

    elapsed = (now - snapshot.last_refill_at).total_seconds()
    new_tokens = min(policy.capacity, snapshot.tokens + elapsed * policy.refill_rate_per_sec)
    return BucketSnapshot(tokens=new_tokens, last_refill_at=now)


def try_consume(
    snapshot: BucketSnapshot,
    n: float,
    *,
    now: datetime,
    policy: BucketPolicy,
) -> tuple[BucketSnapshot, ConsumeOutcome]:
    """Refill then attempt to spend `n` tokens.

    Returns (new_snapshot, outcome). On ALLOWED, tokens are
    spent. On DENIED_INSUFFICIENT, the snapshot reflects the
    refill (so the caller's `time_until_available` is meaningful)
    but no tokens are spent. On DENIED_OVERSIZED, the snapshot
    reflects the refill but the request is fundamentally
    impossible.
    """

    if n <= 0:
        raise ValueError("n must be > 0")
    if not math.isfinite(n):
        raise ValueError("n must be finite")

    refilled = refill(snapshot, now=now, policy=policy)

    if n > policy.capacity:
        return refilled, ConsumeOutcome.DENIED_OVERSIZED

    if refilled.tokens >= n:
        return BucketSnapshot(
            tokens=refilled.tokens - n,
            last_refill_at=refilled.last_refill_at,
        ), ConsumeOutcome.ALLOWED

    return refilled, ConsumeOutcome.DENIED_INSUFFICIENT


def time_until_available(
    snapshot: BucketSnapshot,
    n: float,
    *,
    now: datetime,
    policy: BucketPolicy,
) -> timedelta:
    """How long until `n` tokens will be available.

    Returns timedelta(0) if already available. Returns a sentinel
    `timedelta.max` if `n > capacity` (never available — caller
    should treat as a programming error).
    """

    if n <= 0:
        raise ValueError("n must be > 0")
    if n > policy.capacity:
        return timedelta.max

    refilled = refill(snapshot, now=now, policy=policy)
    if refilled.tokens >= n:
        return timedelta(0)
    deficit = n - refilled.tokens
    seconds = deficit / policy.refill_rate_per_sec
    return timedelta(seconds=seconds)


def fill_ratio(
    snapshot: BucketSnapshot,
    *,
    now: datetime,
    policy: BucketPolicy,
) -> float:
    """Current bucket fill ratio in [0.0, 1.0] (after refill).

    Operator surface: dashboard tile "alpaca limiter at 83%".
    """

    refilled = refill(snapshot, now=now, policy=policy)
    return refilled.tokens / policy.capacity


def _emoji_for_ratio(ratio: float) -> str:
    if ratio >= 0.5:
        return "🟢"
    if ratio >= 0.2:
        return "🟡"
    return "🔴"


def render_snapshot(
    snapshot: BucketSnapshot,
    *,
    name: str,
    now: datetime,
    policy: BucketPolicy,
) -> str:
    """Format bucket state for ops display.

    No-secret-leak: shows only bucket state + refill rate. The
    adapter-side logger handles raw API call details.
    """

    refilled = refill(snapshot, now=now, policy=policy)
    ratio = refilled.tokens / policy.capacity
    emoji = _emoji_for_ratio(ratio)
    return (
        f"{emoji} {name}: {refilled.tokens:.1f}/{policy.capacity:.1f} tokens "
        f"(refill {policy.refill_rate_per_sec:.2f}/s)"
    )


__all__ = [
    "BucketPolicy",
    "BucketSnapshot",
    "ConsumeOutcome",
    "fill_ratio",
    "full_bucket",
    "refill",
    "render_snapshot",
    "time_until_available",
    "try_consume",
]
