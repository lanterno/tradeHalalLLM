"""The news-reversal test's readings: one per company news day, roundups left out."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.events.reversal import news_days
from halal_trader.events.store import EventRecord, EventRecorder

AT = datetime(2018, 3, 1, 15, tzinfo=UTC)


async def test_one_reading_per_company_news_day_timed_at_its_last_headline(
    engine: AsyncEngine,
) -> None:
    def news(sid: str, sym: str, at: datetime, head: str, symbols: list[str]) -> EventRecord:
        return EventRecord(
            "alpaca", sid, "news", sym, at, at, {"headline": head, "symbols": symbols}
        )

    await EventRecorder(engine).record(
        [
            news("a", "MSFT", AT, "Microsoft beats estimates, raises outlook", ["MSFT"]),
            news("b", "MSFT", AT + timedelta(hours=2), "Microsoft shares slump on probe", ["MSFT"]),
            news("c", "AAPL", AT, "Stocks rally broadly", ["AAPL", "MSFT", "NVDA", "AMZN"]),
            news("d", "NVDA", AT.replace(year=2023), "Nvidia soars", ["NVDA"]),
        ]
    )
    obs = await news_days(engine, 2016, 2021)
    assert [(o.symbol, o.published_at) for o in obs] == [("MSFT", AT + timedelta(hours=2))]
    assert await news_days(engine, 2016, 2021, frozenset({"NVDA"})) == []  # 2023 is outside
