"""One way for a page to survive a missing piece (web/soft.py), and the hawl dates."""

from __future__ import annotations

from datetime import date

from halal_trader.compliance.zakat import hawl_dates, hawl_period
from halal_trader.web.soft import soft


async def test_a_failed_read_degrades_to_its_fallback() -> None:
    async def ok() -> int:
        return 1

    async def broken() -> int:
        raise RuntimeError("no such table")

    assert await soft("x", ok(), 0) == 1
    assert await soft("x", broken(), 0) == 0


def test_hawl_dates_are_the_last_and_the_next_hawl() -> None:
    hawl = (9, 1)  # 1 Ramadan
    today = date(2026, 10, 8)
    last, following = hawl_dates(hawl, today)
    assert last <= today < following
    assert hawl_period(hawl, following)[0] == last  # one lunar year apart
    assert 350 <= (following - last).days <= 356
