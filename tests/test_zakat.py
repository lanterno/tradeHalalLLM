"""Zakat by both of Dar al-Ifta's methods, the higher chosen; the lunar hawl."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.zakat import (
    Assessment,
    assess,
    hawl_period,
    parse_hawl,
    record,
)


def test_the_hawl_is_a_lunar_year_ending_on_the_hijri_day() -> None:
    # 1 Ramadan: 1447 AH -> 2026-02-18, 1448 AH -> 2027-02-08 (Umm al-Qura).
    start, end = hawl_period(parse_hawl("09-01"), date(2026, 10, 4))
    assert end == date(2026, 2, 18)
    assert 350 <= (end - start).days <= 356
    assert hawl_period((9, 1), date(2027, 2, 8))[1] == date(2027, 2, 8)  # the day itself


def test_a_30th_in_a_29_day_month_falls_back_to_its_last_day() -> None:
    start, end = hawl_period((2, 30), date(2026, 12, 31))
    assert start < end


def test_parse_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        parse_hawl("13-01")


def test_both_methods_are_computed_and_the_higher_chosen() -> None:
    a = Assessment("paper", date(2025, 1, 1), date(2026, 1, 1), 10_000.0, 300.0, 6.0)
    assert a.trade_goods == pytest.approx(250.0)
    assert a.income == pytest.approx(0.025 * 294.0)
    assert (a.chosen, a.amount) == ("trade goods", pytest.approx(250.0))
    # A tiny holding with large dividends (not realistic, but the rule is symmetric).
    b = Assessment("paper", date(2025, 1, 1), date(2026, 1, 1), 100.0, 10_000.0, 0.0)
    assert b.chosen == "income" and b.amount == pytest.approx(250.0)


async def test_assessing_the_paper_account_from_its_fills_closes_and_dividends(
    engine: AsyncEngine,
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO broker_activities (id, activity_type, transaction_time, symbol, side, "
                "qty, price, raw) VALUES ('1', 'FILL', :t, 'MSFT', 'buy', 24, 400, '{}')"
            ),
            {"t": datetime.fromisoformat("2025-07-20 15:00:00-04:00")},
        )
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, volume, "
                "fetched_at) VALUES ('MSFT', '2026-02-17', 'raw', 410, 410, 410, 410, 1, now())"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO purification_accruals (account, dividend_id, symbol, ex_date, "
                "payable_date, shares, dividend, impure_ratio, amount, method, accrued_at) VALUES "
                "('paper', 'd1', 'MSFT', '2025-08-20', '2025-09-11', 24, 21.84, 0.0137, 0.30, "
                "'m', now())"
            )
        )
    a = await assess(engine, "paper", period_start=date(2025, 3, 1), hawl_date=date(2026, 2, 18))
    assert a.market_value == pytest.approx(24 * 410)  # the last close on or before the hawl
    assert a.trade_goods == pytest.approx(0.025 * 24 * 410)
    assert a.income == pytest.approx(0.025 * (21.84 - 0.30))
    await record(engine, a)
    await record(engine, a)  # a re-run replaces, never duplicates
    async with engine.connect() as conn:
        rows = (await conn.execute(text("SELECT chosen, amount FROM zakat_assessments"))).all()
    assert len(rows) == 1 and rows[0].chosen == "trade goods"


async def test_the_evening_run_records_a_passed_hawl_once_and_catches_up_a_weekend(
    engine: AsyncEngine,
) -> None:
    from types import SimpleNamespace

    from halal_trader.research.daily import _zakat

    settings = SimpleNamespace(zakat=SimpleNamespace(hawl_hijri="09-01"))
    # 1 Ramadan 1448 is Monday 2027-02-08; a run on the 10th still records it.
    assert await _zakat(engine, settings, date(2027, 2, 7)) == {}  # before the hawl
    first = await _zakat(engine, settings, date(2027, 2, 10))
    assert set(first) == {"paper", "core"}  # both broker accounts
    assert await _zakat(engine, settings, date(2027, 2, 11)) == {}  # already recorded
    assert await _zakat(engine, settings, date(2027, 6, 1)) == {}  # long past: history
    unset = SimpleNamespace(zakat=SimpleNamespace(hawl_hijri=""))
    assert await _zakat(engine, unset, date(2027, 2, 10)) == {}
