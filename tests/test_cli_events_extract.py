"""`halal-trader events extract`: adds the current parser's facts, deletes only when asked."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from click.testing import CliRunner
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from halal_trader.cli import cli
from halal_trader.events import earnings_parse as ep

# What a fleet running another parser stored.
DEPLOYED = f"{ep.EXTRACTOR_V4}+{'f' * 12}"


def _run(database_url: str, work: Callable[[AsyncEngine], Awaitable[Any]]) -> Any:
    async def go() -> Any:
        engine = create_async_engine(database_url)
        try:
            return await work(engine)
        finally:
            await engine.dispose()

    return asyncio.run(go())


async def _seed(engine: AsyncEngine) -> None:
    from halal_trader.events.store import EventRecord, EventRecorder

    t = datetime(2026, 10, 1, 20, tzinfo=UTC)
    headline = "Micron Technology Q4 Adj. EPS $33.42 Beats $31.45 Estimate"
    await EventRecorder(engine).record(
        [EventRecord("alpaca", "1", "news", "MU", t, t, {"headline": headline})]
    )
    async with engine.begin() as conn:
        for label in (ep.EXTRACTOR_V3, DEPLOYED):
            await conn.execute(
                text(
                    "INSERT INTO event_facts (event_id, extractor, kind, fields) "
                    "SELECT id, :x, 'result', '{}'::jsonb FROM events"
                ),
                {"x": label},
            )


async def _labels(engine: AsyncEngine) -> dict[str, int]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT extractor, count(*) AS n FROM event_facts GROUP BY 1")
        )
        return {r.extractor: r.n for r in rows}


def test_extract_keeps_every_other_label_unless_told_to_drop(database_url: str) -> None:
    _run(database_url, _seed)

    result = CliRunner().invoke(cli, ["events", "extract"])
    assert result.exit_code == 0, result.output
    assert "1 earnings fact(s) extracted" in result.output
    assert "deleted" not in result.output
    # The other parser's facts, the ones its readers select, are untouched.
    assert _run(database_url, _labels) == {ep.EXTRACTOR_V3: 1, DEPLOYED: 1, ep.EXTRACTOR: 1}

    dropped = CliRunner().invoke(cli, ["events", "extract", "--drop-superseded"])
    assert dropped.exit_code == 0, dropped.output
    assert "0 earnings fact(s) extracted" in dropped.output
    assert "1 superseded fact row(s) deleted" in dropped.output
    assert _run(database_url, _labels) == {ep.EXTRACTOR_V3: 1, ep.EXTRACTOR: 1}
