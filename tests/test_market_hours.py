"""Tests for :mod:`market_hours` — trading-day + close-time logic.

This module is the single source of truth for "is the market open?"
across the stocks bot. A wrong answer here means the bot trades on
holidays or skips half-days entirely.
"""

from __future__ import annotations

from datetime import date, time, timedelta, timezone

from halal_trader.market_hours import (
    EARLY_CLOSE,
    EARLY_CLOSE_DATES,
    MARKET_CLOSE,
    MARKET_OPEN,
    MARKET_TZ,
    US_MARKET_HOLIDAYS,
    effective_close_time,
    is_trading_day,
    trading_day_end_utc,
    trading_day_start_utc,
)

# ── Constants ────────────────────────────────────────────────


def test_market_constants():
    """Pinned wire times so a typo doesn't break every cron job."""
    assert MARKET_OPEN == time(9, 30)
    assert MARKET_CLOSE == time(16, 0)
    assert EARLY_CLOSE == time(13, 0)
    assert str(MARKET_TZ) == "America/New_York"


def test_holidays_include_2026_set():
    """Sanity-check the 2026 holiday calendar — these dates can't trade."""
    assert date(2026, 1, 1) in US_MARKET_HOLIDAYS  # New Year's
    assert date(2026, 7, 3) in US_MARKET_HOLIDAYS  # Independence Day observed
    assert date(2026, 12, 25) in US_MARKET_HOLIDAYS  # Christmas


# ── is_trading_day ──────────────────────────────────────────


def test_weekday_is_trading_day():
    """A regular Tuesday in 2026 (no holiday) → trading day."""
    assert is_trading_day(date(2026, 5, 5)) is True


def test_saturday_not_trading_day():
    """Saturday is never a trading day, even on a non-holiday."""
    assert is_trading_day(date(2026, 5, 9)) is False


def test_sunday_not_trading_day():
    assert is_trading_day(date(2026, 5, 10)) is False


def test_holiday_not_trading_day():
    """A weekday on the holiday calendar → not a trading day."""
    assert is_trading_day(date(2026, 12, 25)) is False  # Christmas (Friday)


def test_early_close_day_is_still_trading_day():
    """Early-close days (Christmas Eve etc.) still trade — just shorter."""
    assert is_trading_day(date(2026, 12, 24)) is True


# ── effective_close_time ────────────────────────────────────


def test_regular_day_closes_at_4pm():
    assert effective_close_time(date(2026, 5, 5)) == MARKET_CLOSE


def test_early_close_day_closes_at_1pm():
    """Each documented early-close date returns the 1pm close."""
    for d in EARLY_CLOSE_DATES:
        assert effective_close_time(d) == EARLY_CLOSE, f"{d} should be early-close"


def test_holiday_close_time_falls_through_to_regular():
    """Holidays aren't special-cased — closed-day logic lives in
    `is_trading_day`. Calling close-time on a holiday returns 4pm."""
    assert effective_close_time(date(2026, 12, 25)) == MARKET_CLOSE


# ── UTC boundary helpers ────────────────────────────────────


def test_trading_day_start_returns_midnight_et_in_utc():
    """May 5 2026 midnight ET (EDT, UTC-4) → 04:00 UTC same day."""
    out = trading_day_start_utc(date(2026, 5, 5))
    assert out.tzinfo == timezone.utc
    assert out.hour == 4  # midnight ET = 4:00 UTC during EDT


def test_trading_day_end_returns_next_midnight():
    """End is exclusive: midnight ET on the *following* day."""
    start = trading_day_start_utc(date(2026, 5, 5))
    end = trading_day_end_utc(date(2026, 5, 5))
    assert end - start == timedelta(days=1)


def test_dst_transition_jan_uses_est_offset():
    """In January (EST, UTC-5), midnight ET → 05:00 UTC."""
    out = trading_day_start_utc(date(2026, 1, 15))
    assert out.hour == 5
