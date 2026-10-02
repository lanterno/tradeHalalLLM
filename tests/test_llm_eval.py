"""LLM vs lexicon: one reading per symbol and day, timed at the day's last headline."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.events.llm_eval import day_readings
from halal_trader.events.store import EventRecord, EventRecorder, Score


async def test_headlines_collapse_to_a_day_reading_at_the_last_headline(
    engine: AsyncEngine,
) -> None:
    t1 = datetime(2026, 3, 2, 14, tzinfo=UTC)
    t2 = datetime(2026, 3, 2, 18, tzinfo=UTC)
    rec = EventRecorder(engine)
    await rec.record(
        [
            EventRecord(
                "alpaca",
                "1",
                "news",
                "AAA",
                t1,
                t1,
                {"headline": "AAA beats estimates"},
                Score("llm-batch:m:x", 0.6),
            ),
            EventRecord(
                "alpaca",
                "2",
                "news",
                "AAA",
                t2,
                t2,
                {"headline": "AAA files report"},
                Score("llm-batch:m:x", 0.2),
            ),
            EventRecord(
                "alpaca", "3", "news", "BBB", t1, t1, {"headline": "BBB"}, Score("other", 0.9)
            ),  # another scorer: not read
        ]
    )
    (r,) = await day_readings(engine)
    assert r.symbol == "AAA" and r.last_at == t2
    assert abs(r.llm - 0.4) < 1e-9
