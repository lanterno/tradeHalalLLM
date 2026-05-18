"""Tests for the now-based helpers in :mod:`market_hours`.

`test_market_hours.py` covers the static-date helpers (`is_trading_day`,
`effective_close_time`, UTC boundaries). This file pins the
"current time" trio that aren't directly tested:

* `now_eastern` — wall-clock in US/Eastern (the single source of
  truth every cycle uses to ask "is the market open right now?").
* `today_eastern` — bare date in US/Eastern (used by the daily
  rollover check in `crypto/scheduler.py`).
* `is_market_open_local` — combines `is_trading_day` + the open
  window. Pre-filter for the broker's authoritative `get_clock()`.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from halal_trader import market_hours
from halal_trader.market_hours import (
    MARKET_TZ,
    is_market_open_local,
    now_eastern,
    today_eastern,
)

# ── now_eastern ────────────────────────────────────────────


def test_now_eastern_returns_eastern_tz():
    """Helper must return a tz-aware datetime *in* US/Eastern, not
    UTC. Pin so a refactor that loses the .astimezone() doesn't
    silently drift cycle timing."""
    dt = now_eastern()
    assert dt.tzinfo is MARKET_TZ
    assert str(dt.tzinfo) == "America/New_York"


def test_now_eastern_is_close_to_real_now():
    """Smoke: returned time is within 5s of `datetime.now(UTC)`."""
    from datetime import UTC
    from datetime import datetime as _dt

    et = now_eastern()
    utc = _dt.now(UTC)
    delta = abs((et - utc).total_seconds())
    assert delta < 5.0


# ── today_eastern ──────────────────────────────────────────


def test_today_eastern_returns_date_object():
    """Returns a bare `date`, not a datetime — used as a dict key
    against the holiday set."""
    out = today_eastern()
    assert isinstance(out, date)
    assert not isinstance(out, datetime)


def test_today_eastern_uses_eastern_not_system_clock(monkeypatch):
    """If the system runs in UTC and it's 2am ET (= 7am UTC), the
    helper should return *yesterday* if the system thinks it's
    today. Pin so daily rollover doesn't drift on UTC-system hosts."""
    # Simulate UTC machine where it's 2026-04-26 02:00 UTC,
    # which is 2026-04-25 22:00 ET → today_eastern should be Apr 25.
    fake_utc = datetime(2026, 4, 26, 2, 0, tzinfo=ZoneInfo("UTC"))

    class _FakeDt(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fake_utc.replace(tzinfo=None)
            return fake_utc.astimezone(tz)

    monkeypatch.setattr(market_hours, "datetime", _FakeDt)

    out = today_eastern()
    assert out == date(2026, 4, 25)  # ET date, not UTC


# ── is_market_open_local ───────────────────────────────────


def _patch_now(monkeypatch, dt: datetime) -> None:
    """Force `now_eastern` to return ``dt`` (must be tz-aware)."""
    monkeypatch.setattr(market_hours, "now_eastern", lambda: dt)


def test_market_closed_on_weekend(monkeypatch):
    """Saturday morning at 10:30 ET → not open even though it's
    inside the regular open window."""
    saturday = datetime(2026, 4, 25, 10, 30, tzinfo=MARKET_TZ)  # Sat
    _patch_now(monkeypatch, saturday)
    assert is_market_open_local() is False


def test_market_closed_on_sunday(monkeypatch):
    sunday = datetime(2026, 4, 26, 10, 30, tzinfo=MARKET_TZ)
    _patch_now(monkeypatch, sunday)
    assert is_market_open_local() is False


def test_market_closed_on_holiday(monkeypatch):
    """Good Friday 2026 (Apr 3) is a market holiday — even at
    market hours, market is closed."""
    holiday = datetime(2026, 4, 3, 10, 30, tzinfo=MARKET_TZ)
    _patch_now(monkeypatch, holiday)
    assert is_market_open_local() is False


def test_market_open_during_regular_hours(monkeypatch):
    """Tuesday 10:30 ET on a normal trading day → open."""
    tuesday = datetime(2026, 4, 21, 10, 30, tzinfo=MARKET_TZ)
    _patch_now(monkeypatch, tuesday)
    assert is_market_open_local() is True


def test_market_closed_before_open(monkeypatch):
    """9:00 ET on a trading day → still pre-open (open is 9:30)."""
    pre_open = datetime(2026, 4, 21, 9, 0, tzinfo=MARKET_TZ)
    _patch_now(monkeypatch, pre_open)
    assert is_market_open_local() is False


def test_market_open_at_exact_open_time(monkeypatch):
    """9:30:00 ET → open (the comparison is `MARKET_OPEN <= current`,
    inclusive on the lower bound)."""
    at_open = datetime(2026, 4, 21, 9, 30, 0, tzinfo=MARKET_TZ)
    _patch_now(monkeypatch, at_open)
    assert is_market_open_local() is True


def test_market_closed_at_exact_close_time(monkeypatch):
    """16:00:00 ET → closed (the comparison is `current < close`,
    exclusive on the upper bound — pin so a refactor to inclusive
    doesn't change the cycle's view of after-hours)."""
    at_close = datetime(2026, 4, 21, 16, 0, 0, tzinfo=MARKET_TZ)
    _patch_now(monkeypatch, at_close)
    assert is_market_open_local() is False


def test_market_closed_after_regular_close(monkeypatch):
    after_close = datetime(2026, 4, 21, 16, 30, tzinfo=MARKET_TZ)
    _patch_now(monkeypatch, after_close)
    assert is_market_open_local() is False


def test_market_open_uses_early_close_for_half_day(monkeypatch):
    """On 2026-11-27 (day after Thanksgiving), the market closes at
    1pm ET — 13:30 ET should report closed even though regular
    close is 4pm. Pin the early-close branch."""
    early_close_day = datetime(2026, 11, 27, 13, 30, tzinfo=MARKET_TZ)
    _patch_now(monkeypatch, early_close_day)
    assert is_market_open_local() is False


def test_market_open_at_1230_on_early_close_day(monkeypatch):
    """12:30 ET on the same early-close day → still open (12:30 < 13:00)."""
    early_close_day = datetime(2026, 11, 27, 12, 30, tzinfo=MARKET_TZ)
    _patch_now(monkeypatch, early_close_day)
    assert is_market_open_local() is True


def test_market_closed_at_exact_early_close_time(monkeypatch):
    """13:00:00 ET on an early-close day → closed (exclusive upper)."""
    at_early_close = datetime(2026, 11, 27, 13, 0, 0, tzinfo=MARKET_TZ)
    _patch_now(monkeypatch, at_early_close)
    assert is_market_open_local() is False


def test_constants_unchanged():
    """Sanity pins — operator never expects these values to drift."""
    from halal_trader.market_hours import EARLY_CLOSE, MARKET_CLOSE, MARKET_OPEN

    assert MARKET_OPEN == time(9, 30)
    assert MARKET_CLOSE == time(16, 0)
    assert EARLY_CLOSE == time(13, 0)
