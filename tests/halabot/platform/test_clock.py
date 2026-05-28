"""Clock — system + fake, the injectable time source (INV-6)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from halabot.platform.clock import Clock, FakeClock, SystemClock


def test_system_clock_is_utc_aware():
    out = SystemClock().now()
    assert out.tzinfo is not None
    assert out.utcoffset() == timedelta(0)


def test_system_clock_satisfies_protocol():
    assert isinstance(SystemClock(), Clock)
    assert isinstance(FakeClock(datetime(2026, 1, 1, tzinfo=UTC)), Clock)


def test_fake_clock_holds_until_advanced():
    t0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)
    c = FakeClock(t0)
    assert c.now() == t0
    assert c.now() == t0  # does not move on its own


def test_fake_clock_advance_moves_forward():
    c = FakeClock(datetime(2026, 5, 28, 12, 0, tzinfo=UTC))
    new = c.advance(timedelta(hours=2, minutes=30))
    assert new == datetime(2026, 5, 28, 14, 30, tzinfo=UTC)
    assert c.now() == new


def test_fake_clock_advance_accepts_negative_delta():
    c = FakeClock(datetime(2026, 5, 28, 12, 0, tzinfo=UTC))
    c.advance(timedelta(minutes=-15))
    assert c.now() == datetime(2026, 5, 28, 11, 45, tzinfo=UTC)


def test_fake_clock_set_jumps_absolute():
    c = FakeClock(datetime(2026, 5, 28, 12, 0, tzinfo=UTC))
    target = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
    c.set(target)
    assert c.now() == target


def test_fake_clock_coerces_naive_to_utc():
    c = FakeClock(datetime(2026, 5, 28, 12, 0))  # naive
    assert c.now().tzinfo is UTC
    c.set(datetime(2026, 6, 1, 0, 0))            # naive
    assert c.now().tzinfo is UTC
