"""The "fast in" study: in-session first headlines, minute entries, buckets."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.events.intraday import (
    Headline,
    Outcome,
    entry_and_close,
    first_in_session,
    summarise,
)
from halal_trader.events.store import EventRecord, EventRecorder, Score


async def test_only_the_first_in_session_headline_per_symbol_and_day(engine: AsyncEngine) -> None:
    def rec(i: int, when: datetime, symbol: str = "AAA") -> EventRecord:
        return EventRecord(
            "alpaca",
            str(i),
            "news",
            symbol,
            when,
            when,
            {"headline": "h"},
            Score("llm-batch:m:x", 0.5),
        )

    await EventRecorder(engine).record(
        [
            rec(1, datetime(2026, 3, 2, 13, 0, tzinfo=UTC)),  # 08:00 ET: pre-market
            rec(2, datetime(2026, 3, 2, 15, 0, tzinfo=UTC)),  # 10:00 ET: first in session
            rec(3, datetime(2026, 3, 2, 16, 0, tzinfo=UTC)),  # later the same day
            rec(4, datetime(2026, 3, 2, 20, 45, tzinfo=UTC)),  # 15:45 ET: too late to trade
            rec(5, datetime(2026, 3, 7, 15, 0, tzinfo=UTC)),  # Saturday
        ]
    )
    (h,) = await first_in_session(engine)
    assert h.published_at == datetime(2026, 3, 2, 15, 0, tzinfo=UTC)


async def test_scored_before_pins_the_set_to_the_scores_that_existed(engine: AsyncEngine) -> None:
    from sqlalchemy import text

    when = datetime(2026, 3, 2, 15, 0, tzinfo=UTC)
    await EventRecorder(engine).record(
        [
            EventRecord(
                "alpaca",
                str(i),
                "news",
                sym,
                when,
                when,
                {"headline": "h"},
                Score("llm-batch:m", 0.5),
            )
            for i, sym in ((1, "AAA"), (2, "BBB"))
        ]
    )
    async with engine.begin() as conn:  # AAA scored on 10-01, BBB after the pin
        for symbol, at in (
            ("AAA", datetime(2026, 10, 1, tzinfo=UTC)),
            ("BBB", datetime(2026, 10, 11, tzinfo=UTC)),
        ):
            await conn.execute(
                text(
                    "UPDATE event_scores SET scored_at = :at WHERE event_id = "
                    "(SELECT id FROM events WHERE symbol = :s)"
                ),
                {"at": at, "s": symbol},
            )
    assert len(await first_in_session(engine)) == 2
    pinned = await first_in_session(engine, scored_before=datetime(2026, 10, 10, tzinfo=UTC))
    assert [h.symbol for h in pinned] == ["AAA"]


def test_entry_is_the_first_bar_starting_after_the_latency() -> None:
    t = datetime(2026, 3, 2, 15, 0, tzinfo=UTC)
    rows = [
        SimpleNamespace(ts=t, vwap=10.0, open=10.0, close=10.0),
        SimpleNamespace(ts=t.replace(minute=1), vwap=10.5, open=10.4, close=10.6),
        SimpleNamespace(ts=t.replace(hour=20, minute=59), vwap=11.0, open=11.0, close=11.2),
    ]
    assert entry_and_close(rows, t.replace(second=30)) == (10.5, 11.2)
    assert entry_and_close(rows, t.replace(hour=21)) is None


def test_buckets_split_by_score() -> None:
    t = datetime(2026, 3, 2, 15, 0, tzinfo=UTC)
    outs = [Outcome(Headline("A", t, 0.6), 0.01 * k, None, None) for k in (1, 2, 3)]
    outs += [Outcome(Headline("B", t, 0.0), 0.0, 0.0, 0.0) for _ in range(3)]
    rows = {(b.label, b.horizon): b for b in summarise(outs)}
    assert rows[("score >= 0.4", "same_day")].n == 3
    assert abs(rows[("score >= 0.4", "same_day")].mean - 0.02) < 1e-12
    assert ("score >= 0.4", "next_day") not in rows
