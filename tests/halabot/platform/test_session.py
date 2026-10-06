"""The regular session: when a shadow fill could happen, and which bars close in it."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from halabot.platform.session import bar_closes_in_session, is_regular_session, timeframe_interval

HOUR = timedelta(hours=1)


def _utc(y: int, mo: int, d: int, h: int, mi: int = 0) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=UTC)


@pytest.mark.parametrize(
    ("instant", "open_"),
    [
        (_utc(2026, 10, 5, 13, 30), True),  # Monday 09:30 EDT, the open
        (_utc(2026, 10, 5, 13, 29), False),  # a minute before it
        (_utc(2026, 10, 5, 19, 59), True),
        (_utc(2026, 10, 5, 20, 0), False),  # 16:00, the close
        (_utc(2026, 10, 3, 15, 0), False),  # Saturday
        (_utc(2026, 11, 26, 15, 0), False),  # Thanksgiving
        (_utc(2026, 11, 27, 17, 0), True),  # 12:00 EST on the early-close day
        (_utc(2026, 11, 27, 18, 30), False),  # 13:30 EST, after the early close
        (_utc(2026, 1, 5, 14, 30), True),  # 09:30 EST in winter
    ],
)
def test_is_regular_session(instant: datetime, open_: bool) -> None:
    assert is_regular_session(instant) is open_


@pytest.mark.parametrize(
    ("bar_start", "in_session"),
    [
        (_utc(2026, 10, 5, 13, 0), True),  # 09:00 bar closes at 10:00
        (_utc(2026, 10, 5, 19, 0), True),  # 15:00 bar closes at the close
        (_utc(2026, 10, 5, 20, 0), False),  # 16:00 bar: after hours
        (_utc(2026, 10, 5, 12, 0), False),  # 08:00 bar closes at 09:00: pre-market
        (_utc(2026, 10, 3, 15, 0), False),  # Saturday
    ],
)
def test_bar_closes_in_session(bar_start: datetime, in_session: bool) -> None:
    assert bar_closes_in_session(bar_start, HOUR) is in_session


def test_timeframe_interval() -> None:
    assert timeframe_interval("1Hour") == HOUR
    assert timeframe_interval("15Min") == timedelta(minutes=15)
    assert timeframe_interval("1Day") == timedelta(days=1)
    assert timeframe_interval("bogus") is None
