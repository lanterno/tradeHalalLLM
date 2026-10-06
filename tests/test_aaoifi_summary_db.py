"""compute_aaoifi_summary against the database: each account's buys judged
by the screen in force on the trade's day."""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.halal.aaoifi_summary import compute_aaoifi_summary

NOW = datetime(2026, 10, 6, 15, 0, tzinfo=UTC)


async def _screen(engine: AsyncEngine, symbol: str, as_of: date, verdict: str) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, sic_description, verdict, "
                "reasons, metrics, method, screened_at) VALUES (:a, :s, '', :v, '[]', '{}', "
                "'t', now())"
            ),
            {"a": as_of, "s": symbol, "v": verdict},
        )


async def _trade(engine: AsyncEngine, symbol: str, side: str, when: str, status: str) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO trades (timestamp, symbol, side, quantity, price, status) "
                "VALUES (:t, :s, :side, 1, 100, :st)"
            ),
            {"t": datetime.fromisoformat(when), "s": symbol, "side": side, "st": status},
        )


async def _core_order(engine: AsyncEngine, symbol: str, side: str, when: str, status: str) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO core_orders (submitted_at, symbol, side, qty, est_price, notional, "
                "reason, status) VALUES (:t, :s, :side, 1, 100, 100, 'rebalance', :st)"
            ),
            {"t": datetime.fromisoformat(when), "s": symbol, "side": side, "st": status},
        )


async def test_each_accounts_buys_are_judged_by_the_screen_on_their_day(
    engine: AsyncEngine,
) -> None:
    await _screen(engine, "MSFT", date(2026, 9, 30), "halal")
    await _screen(engine, "NVDA", date(2026, 9, 30), "halal")
    await _screen(engine, "NVDA", date(2026, 10, 1), "not_halal")  # newer: in force from 10-01
    await _screen(engine, "TSM", date(2026, 10, 1), "doubtful")

    # Day-trader: a halal buy, two buys the screen did not hold halal, one buy
    # of a symbol it never screened, a sale (never a violation) and a buy the
    # broker rejected (never a trade).
    await _trade(engine, "MSFT", "buy", "2026-10-01 10:00:00-04:00", "closed")
    await _trade(engine, "NVDA", "buy", "2026-10-02 10:00:00-04:00", "closed")
    await _trade(engine, "TSM", "buy", "2026-10-02 11:00:00-04:00", "filled")
    await _trade(engine, "ZZZZ", "buy", "2026-10-02 12:00:00-04:00", "filled")
    await _trade(engine, "NVDA", "sell", "2026-10-02 15:00:00-04:00", "filled")
    await _trade(engine, "NVDA", "buy", "2026-10-02 15:30:00-04:00", "rejected")
    # The previous quarter's buy, even at 23:00 New York (03:00 UTC on 10-01).
    await _trade(engine, "ZZZZ", "buy", "2026-09-30 23:00:00-04:00", "closed")
    # Core: halal buys; a refused order never reached the market.
    await _core_order(engine, "MSFT", "buy", "2026-10-05 15:40:00-04:00", "submitted")
    await _core_order(engine, "NVDA", "buy", "2026-10-05 15:40:01-04:00", "refused")
    await _core_order(engine, "MSFT", "sell", "2026-10-06 10:40:00-04:00", "submitted")

    s = await compute_aaoifi_summary(engine, now=NOW)

    core, paper = s.accounts
    assert core.account == "core" and paper.account == "paper"
    assert core.trades_this_quarter == 2 and core.buys_this_quarter == 1
    assert core.trades_today == 1  # the 10-06 sale
    assert core.status == "compliant"

    assert paper.trades_this_quarter == 5 and paper.buys_this_quarter == 4
    assert paper.buy_verdicts == {"halal": 1, "doubtful": 1, "not_halal": 1, "unscreened": 1}
    assert paper.status == "violation" and paper.non_halal_buys_quarter == 3
    by_symbol = {b["symbol"]: b for b in paper.non_halal_buys}
    assert by_symbol["NVDA"] == {
        "symbol": "NVDA",
        "day": "2026-10-02",
        "verdict": "not_halal",
        "screen_as_of": "2026-10-01",
    }
    assert by_symbol["ZZZZ"]["screen_as_of"] is None

    assert s.status == "violation" and s.non_halal_fills_quarter == 3
    assert s.trades_this_quarter == 7
    assert s.quarter_start == date(2026, 10, 1)


async def test_a_quarter_with_only_halal_buys_is_compliant(engine: AsyncEngine) -> None:
    await _screen(engine, "MSFT", date(2026, 10, 1), "halal")
    await _core_order(engine, "MSFT", "buy", "2026-10-05 15:40:00-04:00", "submitted")
    s = await compute_aaoifi_summary(engine, now=NOW)
    assert s.status == "compliant" and s.trades_this_quarter == 1
