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
    metrics = "{}" if ratio is None else f'{{"impure_income_ratio": {ratio:f}}}'
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


async def test_an_unpaid_obligation_stays_outstanding_after_its_quarter(
    engine: AsyncEngine,
) -> None:
    from halal_trader.halal.aaoifi_summary import compute_aaoifi_summary

    await _fill(engine, 1, "MSFT", "buy", 24, "2026-07-20 15:00:00-04:00")
    await _screen(engine, "MSFT", date(2026, 6, 30), 0.02)
    await _dividend(engine, "d1", "MSFT", date(2026, 8, 20), 0.91)  # paid in Q3
    (a,) = await accrue_paper(engine, through=date(2026, 10, 1))
    summary = await compute_aaoifi_summary(engine)  # run in a later quarter
    assert summary.purification_outstanding_usd == pytest.approx(a.amount)
    # Paper accounts rehearse: the amount shows, but it is not money owed.
    assert summary.status == "compliant"


async def test_a_live_accounts_unpaid_purification_needs_attention(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    from halal_trader import config
    from halal_trader.halal.aaoifi_summary import compute_aaoifi_summary

    monkeypatch.setenv("ALPACA_PAPER_TRADE", "false")
    monkeypatch.setattr(config, "_settings", None)
    await _fill(engine, 1, "MSFT", "buy", 24, "2026-07-20 15:00:00-04:00")
    await _screen(engine, "MSFT", date(2026, 6, 30), 0.02)
    await _dividend(engine, "d1", "MSFT", date(2026, 8, 20), 0.91)
    await accrue_paper(engine, through=date(2026, 10, 1))
    assert (await compute_aaoifi_summary(engine)).status == "attention"
    config._settings = None


async def test_each_account_sees_only_its_own_fills(engine: AsyncEngine) -> None:
    from halal_trader.compliance.purification import accrue_account, paper_positions

    await _fill(engine, 1, "MSFT", "buy", 24, "2026-07-20 15:00:00-04:00")  # day-trader
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO broker_activities (id, activity_type, transaction_time, symbol, side, "
                "qty, price, raw, account) VALUES ('c1', 'FILL', :t, 'MSFT', 'buy', 3.5, 400, "
                "'{}', 'core')"
            ),
            {"t": datetime.fromisoformat("2026-07-21 15:40:00-04:00")},
        )
    assert await paper_positions(engine, date(2026, 8, 1)) == {"MSFT": 24.0}
    assert await paper_positions(engine, date(2026, 8, 1), "core") == {"MSFT": 3.5}
    await _screen(engine, "MSFT", date(2026, 6, 30), 0.02)
    await _dividend(engine, "d1", "MSFT", date(2026, 8, 20), 1.0)
    (core,) = await accrue_account(engine, "core", through=date(2026, 10, 1))
    assert core.account == "core" and core.shares == 3.5


async def _core_fill(engine, i: str, side: str, qty: float, when: str) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO broker_activities (id, activity_type, transaction_time, symbol, side, "
                "qty, price, raw, account) VALUES (:i, 'FILL', :t, 'NVDA', :side, :q, 100, '{}', "
                "'core')"
            ),
            {"i": i, "t": datetime.fromisoformat(when), "side": side, "q": qty},
        )


async def _snapshot(engine, taken: str, qty: float, *, ledger_synced: str) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO account_snapshots (account, taken_at, equity, cash, positions) "
                "VALUES ('core', :t, 1, 0, CAST(:p AS JSONB))"
            ),
            {"t": datetime.fromisoformat(taken), "p": f'[{{"symbol": "NVDA", "qty": {qty}}}]'},
        )
        await conn.execute(
            text(
                "INSERT INTO heartbeats (component, beat_at, detail) "
                "VALUES ('stock.ledger', :t, '{}')"
            ),
            {"t": datetime.fromisoformat(ledger_synced)},
        )


async def test_a_split_is_counted_as_the_broker_counts_it(engine: AsyncEngine) -> None:
    """Fills alone know 10 shares bought; a 4-for-1 split made them 40. The
    broker's snapshot, moved back across later fills, says 40 at the ex-date."""
    from halal_trader.compliance.purification import held_shares, paper_positions

    await _core_fill(engine, "n1", "buy", 10, "2026-06-01 15:40:00-04:00")
    # after the split: 40 held; then 5 more bought after the dividend's ex-date
    await _core_fill(engine, "n2", "buy", 5, "2026-08-25 15:40:00-04:00")
    await _snapshot(
        engine, "2026-09-01 16:59:00-04:00", 45, ledger_synced="2026-09-01 16:30:00-04:00"
    )
    before = date(2026, 8, 20)
    assert await paper_positions(engine, before, "core") == {"NVDA": 10.0}
    # The ledger last synced before the snapshot: its fills may be missing.
    assert await held_shares(engine, before, "core") == {"NVDA": 10.0}
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE heartbeats SET beat_at = :t WHERE component = 'stock.ledger'"),
            {"t": datetime.fromisoformat("2026-09-02 16:30:00-04:00")},
        )
    assert await held_shares(engine, before, "core") == {"NVDA": 40.0}

    from halal_trader.compliance.purification import accrue_account

    await _screen(engine, "NVDA", date(2026, 6, 30), 0.01)
    await _dividend(engine, "dn", "NVDA", before, 0.01)
    (a,) = await accrue_account(engine, "core", through=date(2026, 10, 1))
    assert a.shares == 40.0


async def test_a_snapshot_beside_a_fill_is_not_trusted(engine: AsyncEngine) -> None:
    from halal_trader.compliance.purification import held_shares

    await _core_fill(engine, "n1", "buy", 10, "2026-08-25 15:40:00-04:00")
    # taken a few seconds around the fill: which came first cannot be told
    await _snapshot(
        engine, "2026-08-25 15:40:30-04:00", 99, ledger_synced="2026-08-25 16:30:00-04:00"
    )
    assert await held_shares(engine, date(2026, 8, 26), "core") == {"NVDA": 10.0}


async def test_zakat_values_the_brokers_shares(engine: AsyncEngine) -> None:
    from halal_trader.compliance.zakat import assess

    await _core_fill(engine, "n1", "buy", 10, "2026-06-01 15:40:00-04:00")
    await _snapshot(
        engine, "2026-09-01 16:59:00-04:00", 40, ledger_synced="2026-09-02 16:30:00-04:00"
    )
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, volume, "
                "fetched_at) VALUES ('NVDA', '2026-09-01', 'raw', 1, 1, 1, 25, 1, now())"
            )
        )
    a = await assess(engine, "core", period_start=date(2025, 9, 1), hawl_date=date(2026, 9, 1))
    assert a.market_value == 40 * 25
