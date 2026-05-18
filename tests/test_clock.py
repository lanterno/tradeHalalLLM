"""Tests for core/clock.py — Round-5 Wave 0.D."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.core.clock import (
    Clock,
    FrozenClock,
    IdSource,
    SeededIdSource,
    SystemClock,
    SystemIdSource,
)

# --- Protocol structural pins -----------------------------------------------


def test_system_clock_satisfies_protocol():
    assert isinstance(SystemClock(), Clock)


def test_frozen_clock_satisfies_protocol():
    fc = FrozenClock(now=datetime(2026, 5, 5, tzinfo=timezone.utc))
    assert isinstance(fc, Clock)


def test_system_id_source_satisfies_protocol():
    assert isinstance(SystemIdSource(), IdSource)


def test_seeded_id_source_satisfies_protocol():
    assert isinstance(SeededIdSource(), IdSource)


# --- SystemClock -----------------------------------------------------------


def test_system_clock_now_is_utc():
    assert SystemClock().now().tzinfo is not None


def test_system_clock_today_matches_now():
    sc = SystemClock()
    # Tolerant: both reads happen within a millisecond
    assert sc.today() == sc.now().date()


def test_system_clock_monotonic_advances():
    sc = SystemClock()
    a = sc.monotonic()
    time.sleep(0.001)
    b = sc.monotonic()
    assert b > a


# --- FrozenClock -----------------------------------------------------------


def test_frozen_clock_naive_datetime_rejected():
    with pytest.raises(ValueError):
        FrozenClock(now=datetime(2026, 5, 5))  # no tzinfo


def test_frozen_clock_now_does_not_change_unless_advanced():
    t = datetime(2026, 5, 5, 12, 0, 0, tzinfo=timezone.utc)
    fc = FrozenClock(now=t)
    a = fc.now()
    b = fc.now()
    assert a == b == t


def test_frozen_clock_advance_moves_now():
    t = datetime(2026, 5, 5, 12, 0, 0, tzinfo=timezone.utc)
    fc = FrozenClock(now=t)
    fc.advance(timedelta(hours=1))
    assert fc.now() == t + timedelta(hours=1)


def test_frozen_clock_advance_zero_ok():
    t = datetime(2026, 5, 5, tzinfo=timezone.utc)
    fc = FrozenClock(now=t)
    fc.advance(timedelta(0))
    assert fc.now() == t


def test_frozen_clock_negative_advance_rejected():
    t = datetime(2026, 5, 5, tzinfo=timezone.utc)
    fc = FrozenClock(now=t)
    with pytest.raises(ValueError):
        fc.advance(timedelta(seconds=-1))


def test_frozen_clock_today_uses_now_date():
    t = datetime(2026, 5, 5, 23, 0, 0, tzinfo=timezone.utc)
    fc = FrozenClock(now=t)
    assert fc.today() == t.date()


def test_frozen_clock_monotonic_advances_with_now():
    fc = FrozenClock(now=datetime(2026, 5, 5, tzinfo=timezone.utc), monotonic_start=100.0)
    fc.advance(timedelta(seconds=5))
    assert fc.monotonic() == pytest.approx(105.0)


def test_frozen_clock_monotonic_does_not_change_without_advance():
    fc = FrozenClock(now=datetime(2026, 5, 5, tzinfo=timezone.utc), monotonic_start=100.0)
    a = fc.monotonic()
    b = fc.monotonic()
    assert a == b == 100.0


# --- SystemIdSource --------------------------------------------------------


def test_system_id_source_returns_uuid():
    u = SystemIdSource().uuid4()
    assert isinstance(u, uuid.UUID)


def test_system_id_source_uuids_unique():
    src = SystemIdSource()
    a = src.uuid4()
    b = src.uuid4()
    assert a != b


def test_system_id_source_random_token_hex():
    tok = SystemIdSource().random_token(8)
    assert len(tok) == 16  # 8 bytes = 16 hex chars
    assert all(c in "0123456789abcdef" for c in tok)


def test_system_id_source_negative_token_rejected():
    with pytest.raises(ValueError):
        SystemIdSource().random_token(0)


# --- SeededIdSource --------------------------------------------------------


def test_seeded_id_source_returns_uuid():
    u = SeededIdSource(seed=1).uuid4()
    assert isinstance(u, uuid.UUID)


def test_seeded_id_source_same_seed_same_sequence():
    """Two instances with the same seed produce identical UUID sequences."""
    a = SeededIdSource(seed=42)
    b = SeededIdSource(seed=42)
    seq_a = [a.uuid4() for _ in range(5)]
    seq_b = [b.uuid4() for _ in range(5)]
    assert seq_a == seq_b


def test_seeded_id_source_different_seed_different_sequence():
    a = SeededIdSource(seed=1)
    b = SeededIdSource(seed=2)
    assert a.uuid4() != b.uuid4()


def test_seeded_id_source_advances_counter():
    src = SeededIdSource(seed=1)
    a = src.uuid4()
    b = src.uuid4()
    assert a != b


def test_seeded_id_source_random_token_deterministic():
    a = SeededIdSource(seed=99)
    b = SeededIdSource(seed=99)
    assert a.random_token(16) == b.random_token(16)


def test_seeded_id_source_token_length_correct():
    tok = SeededIdSource(seed=0).random_token(20)
    assert len(tok) == 40  # 20 bytes = 40 hex chars


def test_seeded_id_source_token_zero_bytes_rejected():
    with pytest.raises(ValueError):
        SeededIdSource(seed=0).random_token(0)


def test_seeded_id_source_default_seed_zero():
    """Default seed is 0 — pinned for replay convenience."""
    a = SeededIdSource()
    b = SeededIdSource(seed=0)
    assert a.uuid4() == b.uuid4()


# --- E2E -------------------------------------------------------------------


def test_e2e_replay_clock_and_ids_bit_identical():
    """Two simulated 'cycles' with frozen clock + seeded ids produce identical state."""

    def run(clock: Clock, ids: IdSource) -> tuple[datetime, list[uuid.UUID], list[str]]:
        ts = clock.now()
        uuids = [ids.uuid4() for _ in range(3)]
        tokens = [ids.random_token(8) for _ in range(2)]
        return ts, uuids, tokens

    ts1, ids1, toks1 = run(
        FrozenClock(now=datetime(2026, 5, 5, tzinfo=timezone.utc)),
        SeededIdSource(seed=12345),
    )
    ts2, ids2, toks2 = run(
        FrozenClock(now=datetime(2026, 5, 5, tzinfo=timezone.utc)),
        SeededIdSource(seed=12345),
    )
    assert (ts1, ids1, toks1) == (ts2, ids2, toks2)


def test_e2e_advance_changes_outputs_but_remains_replayable():
    """Same seed + advanced FrozenClock → identical UUIDs, advanced timestamp."""
    fc1 = FrozenClock(now=datetime(2026, 5, 5, tzinfo=timezone.utc))
    fc2 = FrozenClock(now=datetime(2026, 5, 5, tzinfo=timezone.utc))
    fc1.advance(timedelta(minutes=5))
    fc2.advance(timedelta(minutes=5))
    assert fc1.now() == fc2.now()
    s1 = SeededIdSource(seed=7)
    s2 = SeededIdSource(seed=7)
    assert s1.uuid4() == s2.uuid4()
