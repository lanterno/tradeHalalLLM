"""Earnings releases as catalysts: one observation per release, scored as registered."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.events.earnings_parse import EXTRACTOR
from halal_trader.events.earnings_signal import Release, releases, signal_of
from halal_trader.events.store import EventRecord, EventRecorder

AT = datetime(2024, 5, 1, 20, 5, tzinfo=UTC)  # after the close


def test_beat_raise_counts_beats_misses_and_guidance() -> None:
    r = Release("NVDA", AT, 0.04, 0.06, "beat", "beats", "raises")
    assert r.beat_raise == 3 and signal_of(r, "beat-raise") == 3.0
    assert Release("X", AT, -0.1, -0.2, "miss", "misses", "cuts").beat_raise == -3
    assert Release("X", AT, 0.0, 0.0, "vs", "inline", "sees").beat_raise == 0  # no direction
    with pytest.raises(ValueError):
        signal_of(r, "nope")


async def test_one_observation_per_release_from_its_earliest_headline(engine: AsyncEngine) -> None:
    items = [
        (
            "n1",
            AT,
            "result",
            '{"sales_surprise": 0.9, "sales_verdict": "beat", "eps_verdict": "beats"}',
        ),
        ("n2", AT + timedelta(minutes=3), "result", '{"sales_surprise": 0.02}'),  # a repackage
        ("n3", AT + timedelta(minutes=1), "guidance", '{"action": "raises"}'),
    ]
    await EventRecorder(engine).record(
        [
            EventRecord("alpaca", sid, "news", "NVDA", at, at, {"headline": sid})
            for sid, at, _, _ in items
        ]
    )
    async with engine.begin() as conn:
        for sid, _, kind, fields in items:
            await conn.execute(
                text(
                    "INSERT INTO event_facts (event_id, extractor, kind, fields) "
                    "SELECT id, :x, :k, CAST(:f AS JSONB) FROM events WHERE source_id = :s"
                ),
                {"x": EXTRACTOR, "k": kind, "f": fields, "s": sid},
            )
    (r,) = await releases(engine)
    assert r.published_at == AT and r.guidance == "raises" and r.beat_raise == 3
    assert r.sales_surprise == 0.5  # clipped: a 90% surprise is a parse or a tiny base
