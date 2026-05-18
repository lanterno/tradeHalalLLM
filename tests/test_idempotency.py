"""Tests for the idempotency key store."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from halal_trader.core.idempotency import (
    IdempotencyAction,
    IdempotencyEntry,
    IdempotencyPolicy,
    IdempotencyState,
    claim,
    decide,
    evict_expired,
    make_idempotency_key,
    record_failure,
    record_success,
    render_entry,
    replay_result,
)

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


# --- Enum string-value pins ---------------------------------------------------


def test_state_string_values():
    assert IdempotencyState.PENDING.value == "pending"
    assert IdempotencyState.SUCCEEDED.value == "succeeded"
    assert IdempotencyState.FAILED.value == "failed"


def test_action_string_values():
    assert IdempotencyAction.PROCEED_NEW.value == "proceed_new"
    assert IdempotencyAction.PROCEED_RETRY.value == "proceed_retry"
    assert IdempotencyAction.REPLAY.value == "replay"
    assert IdempotencyAction.IN_FLIGHT_REJECT.value == "in_flight_reject"


# --- Policy validation --------------------------------------------------------


def test_default_policy_pins():
    p = IdempotencyPolicy()
    assert p.pending_timeout_seconds == 300.0
    assert p.entry_ttl_seconds == 86400.0


def test_zero_pending_timeout_rejected():
    with pytest.raises(ValueError, match="pending_timeout_seconds"):
        IdempotencyPolicy(pending_timeout_seconds=0)


def test_zero_ttl_rejected():
    with pytest.raises(ValueError, match="entry_ttl_seconds"):
        IdempotencyPolicy(entry_ttl_seconds=0)


def test_ttl_below_pending_timeout_rejected():
    """Pin: TTL must be >= pending timeout (otherwise pending entries
    get evicted before they can be reclaimed)."""
    with pytest.raises(ValueError, match=">= pending_timeout"):
        IdempotencyPolicy(pending_timeout_seconds=100, entry_ttl_seconds=50)


def test_policy_immutable():
    p = IdempotencyPolicy()
    with pytest.raises(Exception):
        p.pending_timeout_seconds = 99  # type: ignore[misc]


# --- make_idempotency_key -----------------------------------------------------


def test_key_deterministic():
    k1 = make_idempotency_key("place_order", {"symbol": "BTCUSDT", "qty": 0.1})
    k2 = make_idempotency_key("place_order", {"symbol": "BTCUSDT", "qty": 0.1})
    assert k1 == k2


def test_key_dict_order_independent():
    """Pin: same payload in different insertion order → same key."""
    k1 = make_idempotency_key("place_order", {"a": 1, "b": 2})
    k2 = make_idempotency_key("place_order", {"b": 2, "a": 1})
    assert k1 == k2


def test_key_different_operations_differ():
    k1 = make_idempotency_key("place_order", {"x": 1})
    k2 = make_idempotency_key("cancel_order", {"x": 1})
    assert k1 != k2


def test_key_different_payload_differs():
    k1 = make_idempotency_key("place_order", {"x": 1})
    k2 = make_idempotency_key("place_order", {"x": 2})
    assert k1 != k2


def test_key_is_64_char_hex():
    k = make_idempotency_key("op", {"x": 1})
    assert len(k) == 64
    assert all(c in "0123456789abcdef" for c in k)


def test_key_empty_operation_rejected():
    with pytest.raises(ValueError, match="operation"):
        make_idempotency_key("", {"x": 1})


def test_key_whitespace_operation_rejected():
    with pytest.raises(ValueError, match="operation"):
        make_idempotency_key("   ", {"x": 1})


def test_key_supports_payload_value_types():
    """str / int / float / bool / None all serialize."""
    k = make_idempotency_key("op", {"s": "x", "i": 1, "f": 1.5, "b": True, "n": None})
    assert len(k) == 64


# --- IdempotencyEntry validation ---------------------------------------------


def _pending_entry(*, now: datetime = T0) -> IdempotencyEntry:
    return IdempotencyEntry(
        key="abc",
        state=IdempotencyState.PENDING,
        result=None,
        attempts=1,
        first_seen_at=now,
        last_attempt_at=now,
        terminal_at=None,
    )


def _succeeded_entry(*, now: datetime = T0) -> IdempotencyEntry:
    return IdempotencyEntry(
        key="abc",
        state=IdempotencyState.SUCCEEDED,
        result="order_id_123",
        attempts=1,
        first_seen_at=now,
        last_attempt_at=now,
        terminal_at=now,
    )


def _failed_entry(*, now: datetime = T0) -> IdempotencyEntry:
    return IdempotencyEntry(
        key="abc",
        state=IdempotencyState.FAILED,
        result=None,
        attempts=1,
        first_seen_at=now,
        last_attempt_at=now,
        terminal_at=now,
    )


def test_entry_immutable():
    e = _pending_entry()
    with pytest.raises(Exception):
        e.attempts = 99  # type: ignore[misc]


def test_entry_empty_key_rejected():
    with pytest.raises(ValueError, match="key"):
        IdempotencyEntry(
            key="",
            state=IdempotencyState.PENDING,
            result=None,
            attempts=1,
            first_seen_at=T0,
            last_attempt_at=T0,
        )


def test_entry_zero_attempts_rejected():
    with pytest.raises(ValueError, match="attempts"):
        IdempotencyEntry(
            key="abc",
            state=IdempotencyState.PENDING,
            result=None,
            attempts=0,
            first_seen_at=T0,
            last_attempt_at=T0,
        )


def test_entry_naive_datetime_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        IdempotencyEntry(
            key="abc",
            state=IdempotencyState.PENDING,
            result=None,
            attempts=1,
            first_seen_at=datetime(2026, 1, 1),
            last_attempt_at=T0,
        )


def test_entry_last_attempt_before_first_seen_rejected():
    with pytest.raises(ValueError, match="last_attempt_at"):
        IdempotencyEntry(
            key="abc",
            state=IdempotencyState.PENDING,
            result=None,
            attempts=1,
            first_seen_at=T0,
            last_attempt_at=T0 - timedelta(seconds=1),
        )


def test_pending_with_result_rejected():
    """Pin: PENDING entries cannot have a cached result."""
    with pytest.raises(ValueError, match="PENDING entries must not have a result"):
        IdempotencyEntry(
            key="abc",
            state=IdempotencyState.PENDING,
            result="x",
            attempts=1,
            first_seen_at=T0,
            last_attempt_at=T0,
        )


def test_pending_with_terminal_at_rejected():
    with pytest.raises(ValueError, match="PENDING entries must not have terminal_at"):
        IdempotencyEntry(
            key="abc",
            state=IdempotencyState.PENDING,
            result=None,
            attempts=1,
            first_seen_at=T0,
            last_attempt_at=T0,
            terminal_at=T0,
        )


def test_succeeded_without_terminal_at_rejected():
    with pytest.raises(ValueError, match="terminal entries require terminal_at"):
        IdempotencyEntry(
            key="abc",
            state=IdempotencyState.SUCCEEDED,
            result="x",
            attempts=1,
            first_seen_at=T0,
            last_attempt_at=T0,
        )


def test_succeeded_without_result_rejected():
    """Pin: SUCCEEDED requires result (it IS the cached result)."""
    with pytest.raises(ValueError, match="SUCCEEDED entries require a result"):
        IdempotencyEntry(
            key="abc",
            state=IdempotencyState.SUCCEEDED,
            result=None,
            attempts=1,
            first_seen_at=T0,
            last_attempt_at=T0,
            terminal_at=T0,
        )


def test_failed_with_result_rejected():
    """Pin: FAILED entries must NOT have a result."""
    with pytest.raises(ValueError, match="FAILED entries must not have a result"):
        IdempotencyEntry(
            key="abc",
            state=IdempotencyState.FAILED,
            result="x",
            attempts=1,
            first_seen_at=T0,
            last_attempt_at=T0,
            terminal_at=T0,
        )


def test_terminal_at_before_last_attempt_rejected():
    with pytest.raises(ValueError, match="terminal_at"):
        IdempotencyEntry(
            key="abc",
            state=IdempotencyState.SUCCEEDED,
            result="x",
            attempts=1,
            first_seen_at=T0,
            last_attempt_at=T0,
            terminal_at=T0 - timedelta(seconds=1),
        )


# --- decide -------------------------------------------------------------------


def test_decide_no_entry_proceed_new():
    assert decide({}, "abc", now=T0) is IdempotencyAction.PROCEED_NEW


def test_decide_succeeded_replays():
    entries = {"abc": _succeeded_entry()}
    assert decide(entries, "abc", now=T0) is IdempotencyAction.REPLAY


def test_decide_failed_proceeds_retry():
    """Pin: FAILED entries return PROCEED_RETRY (failures should retry)."""
    entries = {"abc": _failed_entry()}
    assert decide(entries, "abc", now=T0) is IdempotencyAction.PROCEED_RETRY


def test_decide_pending_fresh_rejects():
    """Pin: a PENDING entry within timeout window blocks new claims."""
    entries = {"abc": _pending_entry()}
    assert (
        decide(entries, "abc", now=T0 + timedelta(seconds=10)) is IdempotencyAction.IN_FLIGHT_REJECT
    )


def test_decide_pending_at_timeout_boundary_inclusive():
    """Pin: now - last_attempt_at >= timeout triggers PROCEED_RETRY."""
    entries = {"abc": _pending_entry()}
    assert (
        decide(entries, "abc", now=T0 + timedelta(seconds=300)) is IdempotencyAction.PROCEED_RETRY
    )


def test_decide_pending_just_below_timeout_rejects():
    entries = {"abc": _pending_entry()}
    assert (
        decide(entries, "abc", now=T0 + timedelta(seconds=299))
        is IdempotencyAction.IN_FLIGHT_REJECT
    )


def test_decide_pending_far_past_timeout_proceeds():
    entries = {"abc": _pending_entry()}
    assert decide(entries, "abc", now=T0 + timedelta(hours=1)) is IdempotencyAction.PROCEED_RETRY


def test_decide_custom_pending_timeout():
    policy = IdempotencyPolicy(pending_timeout_seconds=30, entry_ttl_seconds=3600)
    entries = {"abc": _pending_entry()}
    assert (
        decide(entries, "abc", now=T0 + timedelta(seconds=29), policy=policy)
        is IdempotencyAction.IN_FLIGHT_REJECT
    )
    assert (
        decide(entries, "abc", now=T0 + timedelta(seconds=30), policy=policy)
        is IdempotencyAction.PROCEED_RETRY
    )


def test_decide_naive_now_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        decide({}, "abc", now=datetime(2026, 1, 1))


# --- claim --------------------------------------------------------------------


def test_claim_creates_new_entry():
    new = claim({}, "abc", now=T0)
    assert "abc" in new
    e = new["abc"]
    assert e.state is IdempotencyState.PENDING
    assert e.attempts == 1
    assert e.first_seen_at == T0
    assert e.last_attempt_at == T0


def test_claim_increments_attempts_on_existing():
    """Pin: claim on existing entry increments attempts + updates last_attempt_at."""
    entries = {"abc": _failed_entry()}
    later = T0 + timedelta(minutes=1)
    new = claim(entries, "abc", now=later)
    e = new["abc"]
    assert e.state is IdempotencyState.PENDING
    assert e.attempts == 2
    assert e.first_seen_at == T0  # invariant
    assert e.last_attempt_at == later


def test_claim_returns_new_dict_not_mutate():
    entries: dict = {}
    new = claim(entries, "abc", now=T0)
    assert "abc" not in entries
    assert "abc" in new


def test_claim_clears_terminal_state():
    """Re-claiming a FAILED entry should set state=PENDING."""
    entries = {"abc": _failed_entry()}
    new = claim(entries, "abc", now=T0 + timedelta(seconds=10))
    assert new["abc"].state is IdempotencyState.PENDING
    assert new["abc"].result is None
    assert new["abc"].terminal_at is None


def test_claim_naive_now_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        claim({}, "abc", now=datetime(2026, 1, 1))


# --- record_success -----------------------------------------------------------


def test_record_success_marks_terminal():
    entries = claim({}, "abc", now=T0)
    later = T0 + timedelta(seconds=5)
    new = record_success(entries, "abc", "order_id_xyz", now=later)
    e = new["abc"]
    assert e.state is IdempotencyState.SUCCEEDED
    assert e.result == "order_id_xyz"
    assert e.terminal_at == later
    assert e.attempts == 1


def test_record_success_unknown_key_raises_keyerror():
    with pytest.raises(KeyError):
        record_success({}, "abc", "x", now=T0)


def test_record_success_on_succeeded_rejected():
    """Pin: forward-only state machine."""
    entries = {"abc": _succeeded_entry()}
    with pytest.raises(ValueError, match="forward-only"):
        record_success(entries, "abc", "y", now=T0 + timedelta(seconds=1))


def test_record_success_on_failed_rejected():
    entries = {"abc": _failed_entry()}
    with pytest.raises(ValueError, match="forward-only"):
        record_success(entries, "abc", "y", now=T0 + timedelta(seconds=1))


def test_record_success_empty_result_rejected():
    entries = claim({}, "abc", now=T0)
    with pytest.raises(ValueError, match="result"):
        record_success(entries, "abc", "", now=T0)


def test_record_success_naive_now_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        record_success({}, "abc", "x", now=datetime(2026, 1, 1))


# --- record_failure -----------------------------------------------------------


def test_record_failure_marks_terminal():
    entries = claim({}, "abc", now=T0)
    later = T0 + timedelta(seconds=5)
    new = record_failure(entries, "abc", now=later)
    e = new["abc"]
    assert e.state is IdempotencyState.FAILED
    assert e.result is None
    assert e.terminal_at == later


def test_record_failure_unknown_key_raises_keyerror():
    with pytest.raises(KeyError):
        record_failure({}, "abc", now=T0)


def test_record_failure_on_terminal_rejected():
    entries = {"abc": _succeeded_entry()}
    with pytest.raises(ValueError, match="forward-only"):
        record_failure(entries, "abc", now=T0 + timedelta(seconds=1))


# --- replay_result ------------------------------------------------------------


def test_replay_returns_cached():
    entries = {"abc": _succeeded_entry()}
    assert replay_result(entries, "abc") == "order_id_123"


def test_replay_unknown_key_raises_keyerror():
    with pytest.raises(KeyError):
        replay_result({}, "abc")


def test_replay_pending_rejected():
    entries = {"abc": _pending_entry()}
    with pytest.raises(ValueError, match="only SUCCEEDED"):
        replay_result(entries, "abc")


def test_replay_failed_rejected():
    """Pin: only SUCCEEDED entries replay; FAILED retries."""
    entries = {"abc": _failed_entry()}
    with pytest.raises(ValueError, match="only SUCCEEDED"):
        replay_result(entries, "abc")


# --- evict_expired ------------------------------------------------------------


def test_evict_drops_old_entries():
    entries = {"abc": _succeeded_entry()}
    after_ttl = T0 + timedelta(hours=25)
    new = evict_expired(entries, now=after_ttl)
    assert "abc" not in new


def test_evict_keeps_fresh_entries():
    entries = {"abc": _succeeded_entry()}
    new = evict_expired(entries, now=T0 + timedelta(hours=12))
    assert "abc" in new


def test_evict_at_ttl_boundary_inclusive():
    """Pin: entry at exactly TTL is evicted."""
    entries = {"abc": _succeeded_entry()}
    new = evict_expired(entries, now=T0 + timedelta(seconds=86400))
    assert "abc" not in new


def test_evict_just_below_ttl_keeps():
    entries = {"abc": _succeeded_entry()}
    new = evict_expired(entries, now=T0 + timedelta(seconds=86399))
    assert "abc" in new


def test_evict_uses_first_seen_not_terminal():
    """Pin: TTL computed from first_seen_at so long-lived PENDING gets cleaned too."""
    entry = IdempotencyEntry(
        key="abc",
        state=IdempotencyState.PENDING,
        result=None,
        attempts=1,
        first_seen_at=T0,
        last_attempt_at=T0 + timedelta(hours=1),
        terminal_at=None,
    )
    entries = {"abc": entry}
    new = evict_expired(entries, now=T0 + timedelta(hours=25))
    assert "abc" not in new


def test_evict_custom_ttl():
    policy = IdempotencyPolicy(pending_timeout_seconds=10, entry_ttl_seconds=60)
    entries = {"abc": _succeeded_entry()}
    new = evict_expired(entries, now=T0 + timedelta(seconds=60), policy=policy)
    assert "abc" not in new


def test_evict_naive_now_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        evict_expired({}, now=datetime(2026, 1, 1))


# --- render -------------------------------------------------------------------


def test_render_pending():
    out = render_entry(_pending_entry())
    assert "🕐" in out
    assert "pending" in out
    assert "attempt 1" in out


def test_render_succeeded():
    out = render_entry(_succeeded_entry())
    assert "✅" in out
    assert "succeeded" in out


def test_render_failed():
    out = render_entry(_failed_entry())
    assert "❌" in out
    assert "failed" in out


def test_render_no_secret_leak():
    """Pin: render output never includes the result payload."""
    e = IdempotencyEntry(
        key="abc",
        state=IdempotencyState.SUCCEEDED,
        result="order_id=xyz123 fill_price=$45000 secret_token=sk_live_abc",
        attempts=1,
        first_seen_at=T0,
        last_attempt_at=T0,
        terminal_at=T0,
    )
    out = render_entry(e)
    assert "order_id" not in out
    assert "sk_live" not in out
    assert "fill_price" not in out
    assert "$45000" not in out


# --- E2E flows ----------------------------------------------------------------


def test_e2e_first_call_succeeds():
    """Happy path: claim → succeed → replay returns cached result."""
    key = make_idempotency_key("place_order", {"symbol": "BTCUSDT", "qty": 0.1})
    entries: dict = {}

    # First attempt
    assert decide(entries, key, now=T0) is IdempotencyAction.PROCEED_NEW
    entries = claim(entries, key, now=T0)
    # ... do the work ...
    entries = record_success(entries, key, "broker_order_id_42", now=T0 + timedelta(seconds=2))

    # Retry returns cached result
    assert decide(entries, key, now=T0 + timedelta(seconds=10)) is IdempotencyAction.REPLAY
    assert replay_result(entries, key) == "broker_order_id_42"


def test_e2e_failure_retries_to_success():
    """Failure path: claim → fail → retry → succeed."""
    key = make_idempotency_key("place_order", {"symbol": "ETHUSDT", "qty": 0.5})
    entries: dict = {}

    entries = claim(entries, key, now=T0)
    entries = record_failure(entries, key, now=T0 + timedelta(seconds=1))

    # Retry path
    assert decide(entries, key, now=T0 + timedelta(seconds=2)) is IdempotencyAction.PROCEED_RETRY
    entries = claim(entries, key, now=T0 + timedelta(seconds=2))
    assert entries[key].attempts == 2
    entries = record_success(entries, key, "order_99", now=T0 + timedelta(seconds=3))
    assert replay_result(entries, key) == "order_99"


def test_e2e_stuck_pending_reclaimed():
    """Stuck-caller path: pending entry past timeout gets reclaimed."""
    key = make_idempotency_key("place_order", {"symbol": "SOLUSDT", "qty": 1.0})
    entries: dict = {}
    entries = claim(entries, key, now=T0)

    # Original caller crashes; 6 minutes later, retry
    later = T0 + timedelta(minutes=6)
    assert decide(entries, key, now=later) is IdempotencyAction.PROCEED_RETRY
    entries = claim(entries, key, now=later)
    assert entries[key].attempts == 2
    # first_seen_at preserved
    assert entries[key].first_seen_at == T0


def test_e2e_concurrent_callers_blocked():
    """Pin: a fresh PENDING entry blocks a concurrent caller."""
    key = make_idempotency_key("place_order", {"symbol": "BTC", "qty": 0.1})
    entries: dict = {}
    entries = claim(entries, key, now=T0)

    # 30 seconds later (well within 300s timeout) — second caller is rejected
    assert (
        decide(entries, key, now=T0 + timedelta(seconds=30)) is IdempotencyAction.IN_FLIGHT_REJECT
    )


def test_e2e_replay_consistency():
    """Pin: same operations → equal final ledger."""
    key = make_idempotency_key("op", {"x": 1})

    def run() -> dict:
        entries: dict = {}
        entries = claim(entries, key, now=T0)
        entries = record_success(entries, key, "result", now=T0 + timedelta(seconds=1))
        return entries

    a = run()
    b = run()
    assert a[key] == b[key]
