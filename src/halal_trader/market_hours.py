"""US stock market hours, holidays, and timezone helpers.

Single source of truth for all market-time logic.  Every module that needs
to know "what time is it on Wall Street?" or "is the market open?" imports
from here instead of rolling its own datetime arithmetic.

Design decisions
----------------
* **America/New_York** is the canonical timezone for all US equity markets
  (NYSE, NASDAQ).  The app targets American stocks only.
* DB timestamps remain UTC — this module provides helpers to convert
  between the two when querying by trading day.
* The holiday / early-close calendar is maintained as a static set.
  It covers 2025-2027 and should be extended annually.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

# ── Constants ────────────────────────────────────────────────────

MARKET_TZ = ZoneInfo("America/New_York")
"""Canonical timezone for US equity markets (Eastern Time)."""

MARKET_OPEN = time(9, 30)
"""Regular market open: 9:30 AM ET."""

MARKET_CLOSE = time(16, 0)
"""Regular market close: 4:00 PM ET."""

EARLY_CLOSE = time(13, 0)
"""Early close time: 1:00 PM ET (used on half-days)."""


# ── NYSE / NASDAQ Holiday Calendar ──────────────────────────────
#
# Sources: NYSE Rule 7.2, NASDAQ Rule 4120(b).
# Both exchanges observe the same holidays.

US_MARKET_HOLIDAYS: frozenset[date] = frozenset(
    {
        # ── 2025 ─────────────────────────────────────────────
        date(2025, 1, 1),  # New Year's Day
        date(2025, 1, 20),  # Martin Luther King Jr. Day
        date(2025, 2, 17),  # Presidents' Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),  # Independence Day
        date(2025, 9, 1),  # Labor Day
        date(2025, 11, 27),  # Thanksgiving
        date(2025, 12, 25),  # Christmas
        # ── 2026 ─────────────────────────────────────────────
        date(2026, 1, 1),  # New Year's Day
        date(2026, 1, 19),  # Martin Luther King Jr. Day
        date(2026, 2, 16),  # Presidents' Day
        date(2026, 4, 3),  # Good Friday
        date(2026, 5, 25),  # Memorial Day
        date(2026, 6, 19),  # Juneteenth
        date(2026, 7, 3),  # Independence Day (observed — July 4 is Saturday)
        date(2026, 9, 7),  # Labor Day
        date(2026, 11, 26),  # Thanksgiving
        date(2026, 12, 25),  # Christmas
        # ── 2027 ─────────────────────────────────────────────
        date(2027, 1, 1),  # New Year's Day
        date(2027, 1, 18),  # Martin Luther King Jr. Day
        date(2027, 2, 15),  # Presidents' Day
        date(2027, 3, 26),  # Good Friday
        date(2027, 5, 31),  # Memorial Day
        date(2027, 6, 18),  # Juneteenth (observed — June 19 is Saturday)
        date(2027, 7, 5),  # Independence Day (observed — July 4 is Sunday)
        date(2027, 9, 6),  # Labor Day
        date(2027, 11, 25),  # Thanksgiving
        date(2027, 12, 24),  # Christmas (observed — Dec 25 is Saturday)
    }
)

EARLY_CLOSE_DATES: frozenset[date] = frozenset(
    {
        # Markets close at 1:00 PM ET on these days.
        # ── 2025 ─────────────────────────────────────────────
        date(2025, 7, 3),  # Day before Independence Day
        date(2025, 11, 28),  # Day after Thanksgiving
        date(2025, 12, 24),  # Christmas Eve
        # ── 2026 ─────────────────────────────────────────────
        date(2026, 7, 2),  # Day before Independence Day (observed)
        date(2026, 11, 27),  # Day after Thanksgiving
        date(2026, 12, 24),  # Christmas Eve
        # ── 2027 ─────────────────────────────────────────────
        date(2027, 7, 2),  # Day before Independence Day (observed)
        date(2027, 11, 26),  # Day after Thanksgiving
        date(2027, 12, 23),  # Day before Christmas (observed)
    }
)


# ── Time helpers ────────────────────────────────────────────────


def now_eastern() -> datetime:
    """Return the current wall-clock time in US/Eastern."""
    return datetime.now(MARKET_TZ)


def today_eastern() -> date:
    """Return today's date in US/Eastern (not the system timezone)."""
    return now_eastern().date()


# ── Trading-day helpers ─────────────────────────────────────────


def is_trading_day(d: date) -> bool:
    """Return ``True`` if *d* is a regular NYSE/NASDAQ trading day.

    A trading day is a weekday that is not a market holiday.
    """
    return d.weekday() < 5 and d not in US_MARKET_HOLIDAYS


def effective_close_time(d: date) -> time:
    """Return the market close time for date *d*.

    Returns 1:00 PM ET for early-close days, 4:00 PM ET otherwise.
    """
    if d in EARLY_CLOSE_DATES:
        return EARLY_CLOSE
    return MARKET_CLOSE


def is_market_open_local() -> bool:
    """Fast, local check for whether the US stock market is open *right now*.

    This does **not** call any external API.  It checks:
    1. Is today a trading day?
    2. Is the current Eastern time between market open and close?

    Use this as a pre-filter; the broker API (``get_clock()``) remains the
    authoritative source for unexpected closures or halts.
    """
    now = now_eastern()
    if not is_trading_day(now.date()):
        return False
    current_time = now.time()
    close = effective_close_time(now.date())
    return MARKET_OPEN <= current_time < close


# ── UTC boundary helpers (for DB queries) ───────────────────────


def trading_day_start_utc(d: date) -> datetime:
    """Return the UTC ``datetime`` corresponding to midnight ET on date *d*.

    Useful for DB queries: ``WHERE timestamp >= trading_day_start_utc(d)``.
    """
    midnight_et = datetime.combine(d, time.min, tzinfo=MARKET_TZ)
    return midnight_et.astimezone(timezone.utc)


def trading_day_end_utc(d: date) -> datetime:
    """Return the UTC ``datetime`` corresponding to midnight ET on date *d + 1*.

    Useful for DB queries: ``WHERE timestamp < trading_day_end_utc(d)``.
    """
    next_midnight_et = datetime.combine(d + timedelta(days=1), time.min, tzinfo=MARKET_TZ)
    return next_midnight_et.astimezone(timezone.utc)
