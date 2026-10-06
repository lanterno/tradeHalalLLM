"""The US equity regular session, for deciding when a shadow trade could fill.

The shadow prices every hypothetical fill at the latest bar close. Outside the
regular session (09:30-16:00 ET on an NYSE trading day, 13:00 on early-close
days) that close is either an extended-hours print or a stale one: a weekend
proposal opened and closed at Friday's last price and recorded a 0% "loss".
These helpers say whether an instant is in the session and whether a bar's
close is a regular-session print.

Holidays and early closes come from ``halal_trader.market_hours``, the
repository's one maintained NYSE calendar, rather than a second copy here.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from halal_trader.market_hours import (
    MARKET_OPEN,
    MARKET_TZ,
    effective_close_time,
    is_trading_day,
)


def is_regular_session(instant: datetime) -> bool:
    """True when ``instant`` falls inside an NYSE regular session."""
    et = instant.astimezone(MARKET_TZ)
    day = et.date()
    return is_trading_day(day) and MARKET_OPEN <= et.time() < effective_close_time(day)


def bar_closes_in_session(bar_start: datetime, interval: timedelta) -> bool:
    """True when a bar starting at ``bar_start`` ends inside a regular session.

    A bar's close is the last trade before its end, so the 09:00 hourly bar
    (closing at 10:00) carries a regular-session price and the 16:00 bar does
    not. The end may equal the session close; it may not equal the open.
    """
    end = (bar_start + interval).astimezone(MARKET_TZ)
    day = end.date()
    if not is_trading_day(day):
        return False
    opens = datetime.combine(day, MARKET_OPEN, tzinfo=MARKET_TZ)
    closes = datetime.combine(day, effective_close_time(day), tzinfo=MARKET_TZ)
    return opens < end <= closes


_TIMEFRAME = re.compile(r"^\s*(\d+)\s*(min|minute|hour|day|week)s?\s*$", re.IGNORECASE)
_UNIT = {
    "min": timedelta(minutes=1),
    "minute": timedelta(minutes=1),
    "hour": timedelta(hours=1),
    "day": timedelta(days=1),
    "week": timedelta(weeks=1),
}


def timeframe_interval(timeframe: str) -> timedelta | None:
    """An Alpaca timeframe string ("1Hour", "15Min", "1Day") as a duration."""
    m = _TIMEFRAME.match(timeframe)
    if m is None:
        return None
    return int(m.group(1)) * _UNIT[m.group(2).lower()]
