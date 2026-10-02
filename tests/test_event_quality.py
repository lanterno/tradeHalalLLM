"""The event store's quality report: coverage against each year's universe, duplicates."""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.events.quality import coverage, duplicate_share
from halal_trader.events.store import EventRecord, EventRecorder


async def test_coverage_counts_only_that_years_universe_and_duplicates_are_measured(
    engine: AsyncEngine,
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, sic_description, verdict, "
                "reasons, metrics, method, screened_at) "
                "VALUES (:d, :s, '', 'halal', '[]', '{}', 't', now())"
            ),
            [{"d": date(2024, 3, 31), "s": "AAA"}, {"d": date(2024, 3, 31), "s": "BBB"}],
        )
    t = datetime(2024, 5, 1, 14, tzinfo=UTC)
    rec = EventRecorder(engine)
    await rec.record(
        [
            EventRecord("alpaca", "1", "news", "AAA", t, t, {"headline": "x"}),
            EventRecord("alpaca", "2", "news", "AAA", t, t, {"headline": "x"}),  # repeat
            EventRecord("alpaca", "3", "news", "ZZZ", t, t, {"headline": "y"}),  # not a member
        ]
    )
    (row,) = await coverage(engine)
    assert (row.year, row.events, row.companies, row.universe) == (2024, 3, 1, 2)
    assert abs(await duplicate_share(engine) - 1 / 3) < 1e-9
