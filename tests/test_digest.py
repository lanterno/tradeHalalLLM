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
