"""Idempotency key store for retry-safe broker calls.

Auxiliary primitive complementing the bot's resilience layer.
When the bot submits an order to a broker and the network call
times out, the bot needs to retry — but retrying naively could
double-submit the order if the original request actually reached
the broker. The standard pattern: attach an idempotency key to
each request; the broker (or a local ledger) deduplicates on the
key. This module ships the local-ledger half: deterministic key
generation from (operation, payload) + a forward state machine
PENDING → {SUCCEEDED | FAILED} + a pending-timeout reclaim path
for genuinely stuck requests.

Picked a pure-functional snapshot ledger over a stateful service
because (a) the cycle path is single-threaded async; the ledger
is read + advisory-lock + write in one shot, with no concurrent
mutators; (b) snapshots are persistable to DB for cross-restart
deduplication (a 30-second pending request that crashed the bot
is still pending after restart — pure functions over the dict
make this work for free); (c) operators can read the ledger
state without grabbing a lock or interrupting the executor.

Not in scope: the broker-side idempotency key (each adapter sets
the API's per-call X-Idempotency-Key / clientOrderId header from
`make_idempotency_key`). Different from rate limiting (`web/quotas.py`)
which gates spend; different from circuit breakers
(`core/circuit_breaker.py`) which gates calls during outage.

Pinned semantics:
- **Forward state machine: PENDING → {SUCCEEDED | FAILED}.** Once
  terminal, an entry stays terminal until evicted (TTL expiry).
  No PENDING revival; no SUCCEEDED → FAILED rewrite.
- **Stable key generation.** `make_idempotency_key(operation,
  payload)` is canonical: same inputs in any order → same SHA-256
  hex. The payload is JSON-serialized with `sort_keys=True` so
  dict insertion order doesn't change the key.
- **Pending-timeout reclaim.** If a PENDING entry's
  `last_attempt_at` is older than `pending_timeout_seconds`,
  `decide` returns PROCEED_RETRY (the original caller likely
  crashed; let the retry through). Boundary inclusive at the
  timeout.
- **Replay only on SUCCEEDED.** A FAILED entry returns PROCEED_NEW
  (failures should retry); only SUCCEEDED entries replay the
  cached result.
- **Render output never includes the result payload.** The
  payload is operator data (could contain order details, fill
  prices, etc.); render shows only key + state + attempts +
  timestamps. The DB-side query is the audit trail.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import Enum

# Canonical payload value type — what we accept in the payload
# dict. Keeping this strict lets us serialize deterministically
# (json.dumps with sort_keys=True). Nested dicts/lists not
# supported in v1 — operator flattens before passing.
PayloadValue = str | int | float | bool | None


class IdempotencyState(str, Enum):
    """Idempotency entry state.

    Pinned string values for JSON / DB persistence stability.
    Forward-only: PENDING → {SUCCEEDED | FAILED}.
    """

    PENDING = "pending"  # In-flight; original caller still working
    SUCCEEDED = "succeeded"  # Terminal; replay cached result
    FAILED = "failed"  # Terminal; allow retry


class IdempotencyAction(str, Enum):
    """What the caller should do given the ledger state.

    Pinned string values. PROCEED_NEW = no entry exists, claim
    the key. PROCEED_RETRY = stale PENDING or FAILED entry,
    retry. REPLAY = SUCCEEDED entry, return cached result.
    IN_FLIGHT_REJECT = fresh PENDING entry, caller should back
    off (another worker is on it).
    """

    PROCEED_NEW = "proceed_new"
    PROCEED_RETRY = "proceed_retry"
    REPLAY = "replay"
    IN_FLIGHT_REJECT = "in_flight_reject"


@dataclass(frozen=True)
class IdempotencyPolicy:
    """Operator-tunable idempotency ledger policy.

    Defaults: 300s pending timeout (long enough that a real
    network call can finish; short enough that a crashed caller's
    stuck entry doesn't block retries forever); 86400s (24h)
    entry TTL (long enough to deduplicate same-day retries; short
    enough to keep the ledger small).
    """

    pending_timeout_seconds: float = 300.0
    entry_ttl_seconds: float = 86400.0

    def __post_init__(self) -> None:
        if self.pending_timeout_seconds <= 0:
            raise ValueError("pending_timeout_seconds must be > 0")
        if self.entry_ttl_seconds <= 0:
            raise ValueError("entry_ttl_seconds must be > 0")
        if self.entry_ttl_seconds < self.pending_timeout_seconds:
            raise ValueError("entry_ttl_seconds must be >= pending_timeout_seconds")


@dataclass(frozen=True)
class IdempotencyEntry:
    """One ledger entry.

    `key` is the SHA-256 hex from `make_idempotency_key`. `result`
    is the cached payload for SUCCEEDED entries (None for PENDING
    + FAILED). `attempts` counts how many times the key was
    claimed (PROCEED_NEW + PROCEED_RETRY); `first_seen_at` is
    invariant once set; `last_attempt_at` updates per claim;
    `terminal_at` is set on the SUCCEEDED / FAILED transition.
    """

    key: str
    state: IdempotencyState
    result: str | None
    attempts: int
    first_seen_at: datetime
    last_attempt_at: datetime
    terminal_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise ValueError("key must be non-empty")
        if self.attempts < 1:
            raise ValueError("attempts must be >= 1")
        if self.first_seen_at.tzinfo is None:
            raise ValueError("first_seen_at must be timezone-aware")
        if self.last_attempt_at.tzinfo is None:
            raise ValueError("last_attempt_at must be timezone-aware")
        if self.last_attempt_at < self.first_seen_at:
            raise ValueError("last_attempt_at must be >= first_seen_at")
        if self.state is IdempotencyState.PENDING:
            if self.result is not None:
                raise ValueError("PENDING entries must not have a result")
            if self.terminal_at is not None:
                raise ValueError("PENDING entries must not have terminal_at")
        else:  # SUCCEEDED or FAILED
            if self.terminal_at is None:
                raise ValueError("terminal entries require terminal_at")
            if self.terminal_at.tzinfo is None:
                raise ValueError("terminal_at must be timezone-aware")
            if self.terminal_at < self.last_attempt_at:
                raise ValueError("terminal_at must be >= last_attempt_at")
            if self.state is IdempotencyState.SUCCEEDED and self.result is None:
                raise ValueError("SUCCEEDED entries require a result")
            if self.state is IdempotencyState.FAILED and self.result is not None:
                raise ValueError("FAILED entries must not have a result")


def make_idempotency_key(
    operation: str,
    payload: Mapping[str, PayloadValue],
) -> str:
    """Deterministic SHA-256 hex key from (operation, payload).

    Canonical: dict insertion order doesn't matter; same inputs
    always produce the same key. Operator passes the operation
    name (e.g. "place_order") + a flat payload dict. Nested
    structures not supported — operator flattens (e.g. join with
    `.`) before passing.
    """

    if not operation or not operation.strip():
        raise ValueError("operation must be non-empty")
    canonical = json.dumps(
        {"op": operation, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def decide(
    entries: Mapping[str, IdempotencyEntry],
    key: str,
    *,
    now: datetime,
    policy: IdempotencyPolicy = IdempotencyPolicy(),
) -> IdempotencyAction:
    """Decide what to do for `key` given the current ledger state.

    Pure read; doesn't mutate the ledger. The caller passes the
    decision to `claim` (PROCEED_*) or `replay_result` (REPLAY)
    or backs off (IN_FLIGHT_REJECT).
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    entry = entries.get(key)
    if entry is None:
        return IdempotencyAction.PROCEED_NEW

    if entry.state is IdempotencyState.SUCCEEDED:
        return IdempotencyAction.REPLAY
    if entry.state is IdempotencyState.FAILED:
        return IdempotencyAction.PROCEED_RETRY
    # PENDING — check pending-timeout for stuck-caller reclaim
    elapsed = now - entry.last_attempt_at
    timeout = timedelta(seconds=policy.pending_timeout_seconds)
    if elapsed >= timeout:
        return IdempotencyAction.PROCEED_RETRY
    return IdempotencyAction.IN_FLIGHT_REJECT


def claim(
    entries: Mapping[str, IdempotencyEntry],
    key: str,
    *,
    now: datetime,
) -> dict[str, IdempotencyEntry]:
    """Mark `key` PENDING and return the new ledger.

    If `key` is new, creates a fresh entry. If `key` exists in a
    state where retry is allowed (FAILED or stale PENDING),
    increments attempts + resets to PENDING. Caller is expected
    to have called `decide` first and only invoke `claim` when
    the action was PROCEED_NEW or PROCEED_RETRY.

    Does NOT enforce decide's gate — operator-level discipline.
    But the resulting entry is always in PENDING state.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    new_entries = dict(entries)
    existing = entries.get(key)
    if existing is None:
        new_entries[key] = IdempotencyEntry(
            key=key,
            state=IdempotencyState.PENDING,
            result=None,
            attempts=1,
            first_seen_at=now,
            last_attempt_at=now,
            terminal_at=None,
        )
    else:
        new_entries[key] = IdempotencyEntry(
            key=key,
            state=IdempotencyState.PENDING,
            result=None,
            attempts=existing.attempts + 1,
            first_seen_at=existing.first_seen_at,
            last_attempt_at=now,
            terminal_at=None,
        )
    return new_entries


def record_success(
    entries: Mapping[str, IdempotencyEntry],
    key: str,
    result: str,
    *,
    now: datetime,
) -> dict[str, IdempotencyEntry]:
    """Mark `key` SUCCEEDED with the cached result.

    Raises KeyError if `key` doesn't exist (caller must claim
    first). Raises ValueError if entry is already terminal
    (forward-only state machine).
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not result or not result.strip():
        raise ValueError("result must be non-empty for SUCCEEDED")

    existing = entries.get(key)
    if existing is None:
        raise KeyError(f"no entry for key {key!r}")
    if existing.state is not IdempotencyState.PENDING:
        raise ValueError(
            f"cannot record_success on {existing.state.value} entry (forward-only state machine)"
        )

    new_entries = dict(entries)
    new_entries[key] = replace(
        existing,
        state=IdempotencyState.SUCCEEDED,
        result=result,
        terminal_at=now,
    )
    return new_entries


def record_failure(
    entries: Mapping[str, IdempotencyEntry],
    key: str,
    *,
    now: datetime,
) -> dict[str, IdempotencyEntry]:
    """Mark `key` FAILED.

    Raises KeyError if `key` doesn't exist. Raises ValueError if
    entry is already terminal.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    existing = entries.get(key)
    if existing is None:
        raise KeyError(f"no entry for key {key!r}")
    if existing.state is not IdempotencyState.PENDING:
        raise ValueError(
            f"cannot record_failure on {existing.state.value} entry (forward-only state machine)"
        )

    new_entries = dict(entries)
    new_entries[key] = replace(
        existing,
        state=IdempotencyState.FAILED,
        result=None,
        terminal_at=now,
    )
    return new_entries


def replay_result(
    entries: Mapping[str, IdempotencyEntry],
    key: str,
) -> str:
    """Return the cached result for a SUCCEEDED entry.

    Raises KeyError if `key` doesn't exist; ValueError if entry
    is not SUCCEEDED.
    """

    existing = entries.get(key)
    if existing is None:
        raise KeyError(f"no entry for key {key!r}")
    if existing.state is not IdempotencyState.SUCCEEDED:
        raise ValueError(f"cannot replay {existing.state.value} entry — only SUCCEEDED")
    assert existing.result is not None  # invariant
    return existing.result


def evict_expired(
    entries: Mapping[str, IdempotencyEntry],
    *,
    now: datetime,
    policy: IdempotencyPolicy = IdempotencyPolicy(),
) -> dict[str, IdempotencyEntry]:
    """Drop entries past TTL.

    TTL is computed from `first_seen_at` (NOT terminal_at) so
    long-lived PENDING entries also get cleaned up. Boundary
    inclusive: an entry exactly at TTL is evicted.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    ttl = timedelta(seconds=policy.entry_ttl_seconds)
    keep: dict[str, IdempotencyEntry] = {}
    for key, entry in entries.items():
        if (now - entry.first_seen_at) < ttl:
            keep[key] = entry
    return keep


_STATE_EMOJI: dict[IdempotencyState, str] = {
    IdempotencyState.PENDING: "🕐",
    IdempotencyState.SUCCEEDED: "✅",
    IdempotencyState.FAILED: "❌",
}


def render_entry(entry: IdempotencyEntry) -> str:
    """Format one entry for ops display.

    No-secret-leak: shows only key + state + attempt count +
    timestamps. The result payload (which could contain order
    details, fill prices, etc.) is operator-side query data.
    """

    emoji = _STATE_EMOJI[entry.state]
    return (
        f"{emoji} {entry.key[:12]}… {entry.state.value} "
        f"(attempt {entry.attempts}, first {entry.first_seen_at.isoformat()})"
    )


__all__ = [
    "IdempotencyAction",
    "IdempotencyEntry",
    "IdempotencyPolicy",
    "IdempotencyState",
    "PayloadValue",
    "claim",
    "decide",
    "evict_expired",
    "make_idempotency_key",
    "record_failure",
    "record_success",
    "render_entry",
    "replay_result",
]
