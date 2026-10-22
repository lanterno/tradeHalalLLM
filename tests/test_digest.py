"""The weekly digest builds from the database alone and reads as plain lines."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.notifications.digest import build


async def test_the_digest_covers_each_section_even_with_no_history(engine: AsyncEngine) -> None:
    settings = SimpleNamespace(
        zakat=SimpleNamespace(hawl_hijri="09-01"),
        llm=SimpleNamespace(monthly_live_usd=25.0, monthly_research_usd=15.0),
        core=SimpleNamespace(paper=True),
    )
    message = await build(engine, settings, today=date(2026, 10, 9))
    assert message.startswith("<b>Halal Trader")
    for section in ("Core account", "Halal:", "Next zakat", "LLM this month"):
        assert section in message
    assert "&#x27;" not in message


async def test_the_digest_says_how_each_exit_did(engine: AsyncEngine) -> None:
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import text

    closed = datetime.now(UTC) - timedelta(days=1)
    async with engine.begin() as conn:
        for symbol, entry, exit_price, minutes, reason in (
            ("AAPL", 100.0, 99.0, 30, "llm_sell"),
            ("MSFT", 100.0, 102.0, 300, "eod_close_all"),
        ):
            await conn.execute(
                text(
                    "INSERT INTO trades (symbol, side, quantity, status, timestamp, filled_price, "
                    "exit_price, exit_reason, closed_at) VALUES (:s, 'buy', 1, 'closed', :t, :e, "
                    ":x, :r, :c)"
                ),
                {
                    "s": symbol,
                    "t": closed - timedelta(minutes=minutes),
                    "e": entry,
                    "x": exit_price,
                    "r": reason,
                    "c": closed,
                },
            )
    settings = SimpleNamespace(
        zakat=SimpleNamespace(hawl_hijri=""),
        llm=SimpleNamespace(monthly_live_usd=25.0, monthly_research_usd=15.0),
        core=SimpleNamespace(paper=True),
    )
    message = await build(engine, settings, today=date(2026, 10, 9))
    assert (
        "Day-trader exits (7 days): eod_close_all 1, avg +2.00% · "
        "llm_sell 1, avg -1.00% (1 in the first hour, -1.00%)"
    ) in message


async def test_the_digest_says_how_the_day_trader_did_against_spy(engine: AsyncEngine) -> None:
    from halal_trader.core.heartbeat import beat
    from halal_trader.trading.decision_quality import BEAT

    await beat(
        engine,
        BEAT,
        {
            "trips": 104,
            "entry_edge": {"mean": -0.0021, "t": -1.43},
            "exit_edge": {"mean": 0.0021, "t": None},
        },
    )
    settings = SimpleNamespace(
        zakat=SimpleNamespace(hawl_hijri=""),
        llm=SimpleNamespace(monthly_live_usd=25.0, monthly_research_usd=15.0),
        core=SimpleNamespace(paper=True),
    )
    message = await build(engine, settings, today=date(2026, 10, 16))
    assert (
        "Day-trader vs SPY (30 days, 104 trips): buys -0.21% (t -1.4) to the close, "
        "sells +0.21% saved"
    ) in message
    assert "Day-trader plans" not in message  # a summary from before the plan split


async def test_the_digest_says_how_many_buys_kept_the_plan(engine: AsyncEngine) -> None:
    from halal_trader.core.heartbeat import beat
    from halal_trader.trading.decision_quality import BEAT

    def stat(plan: str, trips: int, mean: float | None) -> dict[str, object]:
        return {"plan": plan, "trips": trips, "sessions": 3, "mean": mean, "t": None}

    await beat(
        engine,
        BEAT,
        {
            "trips": 20,
            "entry_edge": {"mean": -0.001, "t": None},
            "exit_edge": {"mean": 0.001, "t": None},
            "by_plan": [
                stat("2:1 or better", 5, 0.0012),
                stat("below 2:1", 6, -0.002),
                stat("stop at or above fill", 0, None),
                stat("no target", 1, 0.0),
                stat("no stop", 8, -0.001),
            ],
        },
    )
    settings = SimpleNamespace(
        zakat=SimpleNamespace(hawl_hijri=""),
        llm=SimpleNamespace(monthly_live_usd=25.0, monthly_research_usd=15.0),
        core=SimpleNamespace(paper=True),
    )
    message = await build(engine, settings, today=date(2026, 10, 22))
    assert (
        "Day-trader plans: 5 of 20 buys at 2:1 or better (+0.12% to the close); "
        "below 2:1 6, no target 1, no stop 8"
    ) in message
