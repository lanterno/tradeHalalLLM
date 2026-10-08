"""market_hours' static calendar against the broker's (market_hours.calendar_mismatches)."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from halal_trader.market_hours import calendar_mismatches

# Thanksgiving week 2026: Thu 26 closed, Fri 27 closes at 13:00.
WEEK = [
    {"date": "2026-11-23", "open": "09:30", "close": "16:00"},
    {"date": "2026-11-24", "open": "09:30", "close": "16:00"},
    {"date": "2026-11-25", "open": "09:30", "close": "16:00"},
    {"date": "2026-11-27", "open": "09:30", "close": "13:00"},
]
MON, SUN = date(2026, 11, 23), date(2026, 11, 29)


def test_agreement_is_no_mismatch() -> None:
    assert calendar_mismatches(WEEK, MON, SUN) == []


def test_a_holiday_the_broker_keeps_but_market_hours_lacks_is_named() -> None:
    broker = [d for d in WEEK if d["date"] != "2026-11-24"]  # the broker closes Tuesday
    assert calendar_mismatches(broker, MON, SUN) == [
        "Tue 24 Nov 2026: broker closed, market_hours closes 16:00"
    ]


def test_a_session_on_a_day_market_hours_calls_a_holiday_is_named() -> None:
    broker = [*WEEK, {"date": "2026-11-26", "open": "09:30", "close": "16:00"}]
    assert calendar_mismatches(broker, MON, SUN) == [
        "Thu 26 Nov 2026: broker closes 16:00, market_hours closed"
    ]


def test_an_early_close_that_differs_is_named() -> None:
    broker = [d if d["date"] != "2026-11-27" else {**d, "close": "16:00"} for d in WEEK]
    assert calendar_mismatches(broker, MON, SUN) == [
        "Fri 27 Nov 2026: broker closes 16:00, market_hours closes 13:00"
    ]


def test_days_past_the_static_table_show_up_as_mismatches() -> None:
    """The table ends in 2027; the broker's 2028 holidays read as sessions here."""
    broker = [{"date": "2028-01-03", "open": "09:30", "close": "16:00"}]
    found = calendar_mismatches(broker, date(2027, 12, 31), date(2028, 1, 3))
    assert found == ["Fri 31 Dec 2027: broker closed, market_hours closes 16:00"]


@pytest.mark.asyncio
async def test_the_research_run_reports_mismatches(monkeypatch) -> None:
    from halal_trader.execution import alpaca_broker
    from halal_trader.research import daily

    class FakeBroker:
        def __init__(self, *_a: object, **_k: object) -> None: ...

        async def get_calendar(self, start: str, end: str) -> list[dict[str, str]]:
            return [d for d in WEEK if d["date"] != "2026-11-24"]

        async def disconnect(self) -> None: ...

    monkeypatch.setattr(alpaca_broker, "AlpacaRestBroker", FakeBroker)
    monkeypatch.setattr(daily, "CALENDAR_HORIZON", SUN - MON)
    settings = SimpleNamespace(
        alpaca=SimpleNamespace(api_key="k", secret_key="s", paper_trade=True)
    )
    problems = await daily._calendar_check(settings, MON)  # type: ignore[arg-type]
    assert problems == [
        "calendar: Tue 24 Nov 2026: broker closed, market_hours closes 16:00 (fix market_hours.py)"
    ]
    no_keys = SimpleNamespace(alpaca=SimpleNamespace(api_key=""))
    assert await daily._calendar_check(no_keys, MON) == []  # type: ignore[arg-type]
