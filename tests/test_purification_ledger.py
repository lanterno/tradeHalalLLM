"""Purification: dividend x impure-income ratio, from positions held at the ex-date."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.purification import (
    DEFAULT_RATIO,
    accrue_book,
    accrue_paper,
    report,
)


async def _dividend(engine, sid: str, symbol: str, ex: date, rate: float) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO dividends (source_id, symbol, ex_date, payable_date, rate, special) "
                "VALUES (:i, :s, :ex, :pay, :r, false)"
            ),
            {"i": sid, "s": symbol, "ex": ex, "pay": ex.replace(day=ex.day + 3), "r": rate},
        )


async def _fill(engine, i: int, symbol: str, side: str, qty: float, when: str) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO broker_activities (id, activity_type, transaction_time, symbol, side, "
                "qty, price, raw) VALUES (:i, 'FILL', :t, :s, :side, :q, 100, "
                "'{}')"
            ),
            {"i": str(i), "t": datetime.fromisoformat(when), "s": symbol, "side": side, "q": qty},
        )


async def _screen(engine, symbol: str, as_of: date, ratio: float | None) -> None:
    metrics = "{}" if ratio is None else '{"impure_income_ratio": %f}' % ratio
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, sic_description, verdict, "
                "reasons, metrics, method, screened_at) VALUES (:a, :s, '', 'halal', '[]', "
                "CAST(:m AS JSONB), 't', now())"
            ),
            {"a": as_of, "s": symbol, "m": metrics},
        )


async def test_the_paper_account_purifies_what_it_held_at_the_ex_date(engine: AsyncEngine) -> None:
    await _fill(engine, 1, "MSFT", "buy", 30, "2026-07-20 15:00:00-04:00")
    await _fill(engine, 2, "MSFT", "sell", 6, "2026-07-21 15:00:00-04:00")
    await _fill(engine, 3, "AAPL", "buy", 10, "2026-08-10 10:00:00-04:00")  # bought on the ex-date
    await _screen(engine, "MSFT", date(2026, 6, 30), 0.02)
    await _dividend(engine, "d1", "MSFT", date(2026, 8, 20), 0.91)
    await _dividend(engine, "d2", "AAPL", date(2026, 8, 10), 0.26)

    (a,) = await accrue_paper(engine, through=date(2026, 10, 1))
    assert (a.symbol, a.shares) == ("MSFT", 24.0)
    assert a.amount == pytest.approx(24 * 0.91 * 0.02)
    assert await accrue_paper(engine, through=date(2026, 10, 1)) == []  # written once

    (line,) = await report(engine, "paper", 2026)
    assert line.amount == pytest.approx(24 * 0.91 * 0.02) and line.assumed == 0


async def test_an_unknown_ratio_assumes_the_five_percent_ceiling(engine: AsyncEngine) -> None:
    await _fill(engine, 1, "XYZ", "buy", 100, "2026-07-01 15:00:00-04:00")
    await _screen(engine, "XYZ", date(2026, 6, 30), None)
    await _dividend(engine, "d1", "XYZ", date(2026, 8, 3), 1.0)
    (a,) = await accrue_paper(engine, through=date(2026, 10, 1))
    assert a.ratio == DEFAULT_RATIO and a.amount == pytest.approx(5.0)
    (line,) = await report(engine, "paper", 2026)
    assert line.assumed == 1


async def test_a_book_purifies_per_notional_from_its_weight_nav_and_close(
    engine: AsyncEngine,
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO forward_books (name, strategy, params, created_at) "
                "VALUES ('core', 'core-strict-cap', '{}', now())"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO forward_book_days (book, day, nav, day_return, turnover, weights, "
                "rebalance_next, recorded_at) VALUES ('core', '2026-08-19', 1.1, 0, 0, "
                "'{\"MSFT\": 0.5}', false, now())"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, volume, "
                "fetched_at) VALUES ('MSFT', '2026-08-19', 'raw', 400, 400, 400, 400, 1, now())"
            )
        )
    await _screen(engine, "MSFT", date(2026, 6, 30), 0.02)
    await _dividend(engine, "d1", "MSFT", date(2026, 8, 20), 0.91)
    (a,) = await accrue_book(engine, "core", through=date(2026, 10, 1), notional=10_000)
    shares = 0.5 * 1.1 * 10_000 / 400
    assert a.shares == pytest.approx(shares)
    assert a.amount == pytest.approx(shares * 0.91 * 0.02)


async def test_a_donation_settles_the_ledger_and_the_compliance_summary_sees_both(
    engine: AsyncEngine,
) -> None:
    from datetime import UTC
    from datetime import datetime as dt

    from sqlmodel.ext.asyncio.session import AsyncSession

    from halal_trader.compliance.purification import mark_paid
    from halal_trader.halal.aaoifi_summary import (
        _sum_purification_accrued,
        _sum_purification_disbursed,
    )

    await _fill(engine, 1, "MSFT", "buy", 24, "2026-07-20 15:00:00-04:00")
    await _screen(engine, "MSFT", date(2026, 6, 30), 0.02)
    await _dividend(engine, "d1", "MSFT", date(2026, 8, 20), 0.91)
    (a,) = await accrue_paper(engine, through=date(2026, 10, 1))

    since = dt(2026, 7, 1, tzinfo=UTC)
    async with AsyncSession(engine) as session:
        assert await _sum_purification_accrued(session, since) == pytest.approx(a.amount)
        assert await _sum_purification_disbursed(session, since) == 0.0

    assert await mark_paid(
        engine, "paper", through=date(2026, 9, 30), paid_to="Local food bank"
    ) == (pytest.approx(a.amount))
    assert await mark_paid(engine, "paper", through=date(2026, 9, 30), paid_to="x") == 0.0
    async with AsyncSession(engine) as session:
        assert await _sum_purification_disbursed(session, since) == pytest.approx(a.amount)


async def test_unpaid_accruals_can_be_recomputed_and_paid_ones_never_change(
    engine: AsyncEngine,
) -> None:
    from halal_trader.compliance.purification import clear_unpaid, mark_paid

    await _fill(engine, 1, "MSFT", "buy", 24, "2026-07-20 15:00:00-04:00")
    await _screen(engine, "MSFT", date(2026, 6, 30), 0.0)
    await _dividend(engine, "d1", "MSFT", date(2026, 8, 20), 0.91)
    await _dividend(engine, "d2", "MSFT", date(2026, 9, 10), 0.91)
    await accrue_paper(engine, through=date(2026, 8, 31))
    await mark_paid(engine, "paper", through=date(2026, 8, 31), paid_to="x")  # d1 settled
    await accrue_paper(engine, through=date(2026, 9, 30))
    async with engine.begin() as conn:  # the screen is corrected after the fact
        await conn.execute(
            text("UPDATE halal_screen_results SET metrics = '{\"impure_income_ratio\": 0.013}'")
        )
    assert await clear_unpaid(engine) == 1
    (redone,) = await accrue_paper(engine, through=date(2026, 9, 30))
    assert redone.dividend_id == "d2" and redone.ratio == 0.013
