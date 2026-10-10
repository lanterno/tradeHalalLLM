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
  It covers 2016-2027 (2016-2024 taken from the broker's ``/v2/calendar``,
  so research over that history walks real sessions) and should be extended
  annually; the evening research
  run compares the next 90 days with the broker's calendar
  (:func:`calendar_mismatches`) and alerts on any difference, so a wrong
  entry, or the table running out, is caught months ahead.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

# ── Constants ────────────────────────────────────────────────────

MARKET_TZ_NAME = "America/New_York"
"""The zone's IANA name, for SQL (``AT TIME ZONE``)."""

MARKET_TZ = ZoneInfo(MARKET_TZ_NAME)
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
        # ── 2016 ─────────────────────────────────────────────
        date(2016, 1, 1),  # New Year's Day
        date(2016, 1, 18),  # Martin Luther King Jr. Day
        date(2016, 2, 15),  # Presidents' Day
        date(2016, 3, 25),  # Good Friday
        date(2016, 5, 30),  # Memorial Day
        date(2016, 7, 4),  # Independence Day
        date(2016, 9, 5),  # Labor Day
        date(2016, 11, 24),  # Thanksgiving
        date(2016, 12, 26),  # Christmas (observed — Dec 25 is Sunday)
        # ── 2017 ─────────────────────────────────────────────
        date(2017, 1, 2),  # New Year's Day (observed — Jan 1 is Sunday)
        date(2017, 1, 16),  # Martin Luther King Jr. Day
        date(2017, 2, 20),  # Presidents' Day
        date(2017, 4, 14),  # Good Friday
        date(2017, 5, 29),  # Memorial Day
        date(2017, 7, 4),  # Independence Day
        date(2017, 9, 4),  # Labor Day
        date(2017, 11, 23),  # Thanksgiving
        date(2017, 12, 25),  # Christmas
        # ── 2018 ─────────────────────────────────────────────
        date(2018, 1, 1),  # New Year's Day
        date(2018, 1, 15),  # Martin Luther King Jr. Day
        date(2018, 2, 19),  # Presidents' Day
        date(2018, 3, 30),  # Good Friday
        date(2018, 5, 28),  # Memorial Day
        date(2018, 7, 4),  # Independence Day
        date(2018, 9, 3),  # Labor Day
        date(2018, 11, 22),  # Thanksgiving
        date(2018, 12, 5),  # National Day of Mourning (George H. W. Bush)
        date(2018, 12, 25),  # Christmas
        # ── 2019 ─────────────────────────────────────────────
        date(2019, 1, 1),  # New Year's Day
        date(2019, 1, 21),  # Martin Luther King Jr. Day
        date(2019, 2, 18),  # Presidents' Day
        date(2019, 4, 19),  # Good Friday
        date(2019, 5, 27),  # Memorial Day
        date(2019, 7, 4),  # Independence Day
        date(2019, 9, 2),  # Labor Day
        date(2019, 11, 28),  # Thanksgiving
        date(2019, 12, 25),  # Christmas
        # ── 2020 ─────────────────────────────────────────────
        date(2020, 1, 1),  # New Year's Day
        date(2020, 1, 20),  # Martin Luther King Jr. Day
        date(2020, 2, 17),  # Presidents' Day
        date(2020, 4, 10),  # Good Friday
        date(2020, 5, 25),  # Memorial Day
        date(2020, 7, 3),  # Independence Day (observed — July 4 is Saturday)
        date(2020, 9, 7),  # Labor Day
        date(2020, 11, 26),  # Thanksgiving
        date(2020, 12, 25),  # Christmas
        # ── 2021 ─────────────────────────────────────────────
        date(2021, 1, 1),  # New Year's Day
        date(2021, 1, 18),  # Martin Luther King Jr. Day
        date(2021, 2, 15),  # Presidents' Day
        date(2021, 4, 2),  # Good Friday
        date(2021, 5, 31),  # Memorial Day
        date(2021, 7, 5),  # Independence Day (observed — July 4 is Sunday)
        date(2021, 9, 6),  # Labor Day
        date(2021, 11, 25),  # Thanksgiving
        date(2021, 12, 24),  # Christmas (observed — Dec 25 is Saturday)
        # ── 2022 ─────────────────────────────────────────────
        date(2022, 1, 17),  # Martin Luther King Jr. Day
        date(2022, 2, 21),  # Presidents' Day
        date(2022, 4, 15),  # Good Friday
        date(2022, 5, 30),  # Memorial Day
        date(2022, 6, 20),  # Juneteenth (observed — June 19 is Sunday)
        date(2022, 7, 4),  # Independence Day
        date(2022, 9, 5),  # Labor Day
        date(2022, 11, 24),  # Thanksgiving
        date(2022, 12, 26),  # Christmas (observed — Dec 25 is Sunday)
        # ── 2023 ─────────────────────────────────────────────
        date(2023, 1, 2),  # New Year's Day (observed — Jan 1 is Sunday)
        date(2023, 1, 16),  # Martin Luther King Jr. Day
        date(2023, 2, 20),  # Presidents' Day
        date(2023, 4, 7),  # Good Friday
        date(2023, 5, 29),  # Memorial Day
        date(2023, 6, 19),  # Juneteenth
        date(2023, 7, 4),  # Independence Day
        date(2023, 9, 4),  # Labor Day
        date(2023, 11, 23),  # Thanksgiving
        date(2023, 12, 25),  # Christmas
        # ── 2024 ─────────────────────────────────────────────
        date(2024, 1, 1),  # New Year's Day
        date(2024, 1, 15),  # Martin Luther King Jr. Day
        date(2024, 2, 19),  # Presidents' Day
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),  # Independence Day
        date(2024, 9, 2),  # Labor Day
        date(2024, 11, 28),  # Thanksgiving
        date(2024, 12, 25),  # Christmas
        # ── 2025 ─────────────────────────────────────────────
        date(2025, 1, 1),  # New Year's Day
        date(2025, 1, 9),  # National Day of Mourning (Jimmy Carter)
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
        #
        # NYSE rule of thumb: the 1pm early close applies to July 3
        # ONLY when July 4 falls Tue–Sat (i.e. July 3 is a weekday
        # session); when July 4 falls on a Saturday the holiday is
        # observed Friday July 3 (full closure) and Thursday July 2 is
        # a NORMAL full session — there is no "day before the observed
        # holiday" early close (cf. 2015, 2020 full sessions on July 2).
        # Same for Christmas: Dec 24 early-closes only when it's a
        # weekday session; Dec 23 is never an early close (cf. 2021).
        # Live incident 2026-07-02 12:30 ET: a wrong entry here engaged
        # the close lockout 3h early and would have stopped the cycle
        # AND the SL/TP monitor at 13:00 with positions open. The
        # broker clock said 16:00 all along — keep this table matched
        # to the official NYSE calendar, not derived rules.
        # ── 2016 ─────────────────────────────────────────────
        date(2016, 11, 25),  # Day after Thanksgiving
        # ── 2017 ─────────────────────────────────────────────
        date(2017, 7, 3),  # Day before Independence Day
        date(2017, 11, 24),  # Day after Thanksgiving
        # ── 2018 ─────────────────────────────────────────────
        date(2018, 7, 3),  # Day before Independence Day
        date(2018, 11, 23),  # Day after Thanksgiving
        date(2018, 12, 24),  # Christmas Eve
        # ── 2019 ─────────────────────────────────────────────
        date(2019, 7, 3),  # Day before Independence Day
        date(2019, 11, 29),  # Day after Thanksgiving
        date(2019, 12, 24),  # Christmas Eve
        # ── 2020 ─────────────────────────────────────────────
        date(2020, 11, 27),  # Day after Thanksgiving
        date(2020, 12, 24),  # Christmas Eve
        # ── 2021 ─────────────────────────────────────────────
        date(2021, 11, 26),  # Day after Thanksgiving
        # ── 2022 ─────────────────────────────────────────────
        date(2022, 11, 25),  # Day after Thanksgiving
        # ── 2023 ─────────────────────────────────────────────
        date(2023, 7, 3),  # Day before Independence Day
        date(2023, 11, 24),  # Day after Thanksgiving
        # ── 2024 ─────────────────────────────────────────────
        date(2024, 7, 3),  # Day before Independence Day
        date(2024, 11, 29),  # Day after Thanksgiving
        date(2024, 12, 24),  # Christmas Eve
        # ── 2025 ─────────────────────────────────────────────
        date(2025, 7, 3),  # July 4 is a Friday → July 3 half day
        date(2025, 11, 28),  # Day after Thanksgiving
        date(2025, 12, 24),  # Christmas Eve (Wednesday)
        # ── 2026 ─────────────────────────────────────────────
        # July 4 falls Saturday → observed Fri Jul 3 (closed);
        # Thu Jul 2 is a FULL session, no early close.
        date(2026, 11, 27),  # Day after Thanksgiving
        date(2026, 12, 24),  # Christmas Eve (Thursday)
        # ── 2027 ─────────────────────────────────────────────
        # July 4 falls Sunday → observed Mon Jul 5 (closed);
        # Fri Jul 2 is a FULL session, no early close.
        date(2027, 11, 26),  # Day after Thanksgiving
        # Dec 25, 2027 falls Saturday → observed Fri Dec 24 (closed);
        # Thu Dec 23 is a FULL session, no early close.
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


def next_trading_day(after: date) -> date:
    """The first trading day after ``after``."""
    d = after + timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def previous_trading_day(before: date) -> date:
    """The last trading day before ``before``."""
    d = before - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def trading_days_back(end: date, n: int) -> list[date]:
    """The ``n`` trading days ending on ``end`` (inclusive when it is one), oldest first."""
    out: list[date] = []
    d = end
    while len(out) < n:
        if is_trading_day(d):
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


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
    return midnight_et.astimezone(UTC)


def trading_day_end_utc(d: date) -> datetime:
    """Return the UTC ``datetime`` corresponding to midnight ET on date *d + 1*.

    Useful for DB queries: ``WHERE timestamp < trading_day_end_utc(d)``.
    """
    next_midnight_et = datetime.combine(d + timedelta(days=1), time.min, tzinfo=MARKET_TZ)
    return next_midnight_et.astimezone(UTC)


def calendar_mismatches(broker_days: list[dict[str, str]], start: date, end: date) -> list[str]:
    """Where this module's calendar disagrees with the broker's in [start, end].

    ``broker_days`` is Alpaca's ``/v2/calendar``: one ``{"date", "open",
    "close"}`` per session. A day either side calls a session but the other
    does not, or a session whose close differs, is a mismatch.
    """
    sessions = {
        date.fromisoformat(d["date"]): time.fromisoformat(d["close"])
        for d in broker_days
        if start.isoformat() <= d["date"] <= end.isoformat()
    }
    out = []
    d = start
    while d <= end:
        theirs = sessions.get(d)
        ours = effective_close_time(d) if is_trading_day(d) else None
        if theirs != ours:
            out.append(
                f"{d:%a %d %b %Y}: broker "
                + (f"closes {theirs:%H:%M}" if theirs else "closed")
                + ", market_hours "
                + (f"closes {ours:%H:%M}" if ours else "closed")
            )
        d += timedelta(days=1)
    return out
