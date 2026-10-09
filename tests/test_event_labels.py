"""Event labels start at the first close after publication; the report counts a catalyst once."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.events.labels import first_close_session, label_events, report
from halal_trader.events.store import EventRecord, EventRecorder, Score

SESSIONS = [
    date(2026, 9, 28) + timedelta(days=i)
    for i in range(40)
    if (date(2026, 9, 28) + timedelta(days=i)).weekday() < 5
]


def _utc(y: int, m: int, d: int, h: int, mi: int = 0) -> datetime:
    return datetime(y, m, d, h, mi, tzinfo=UTC)


def test_the_first_close_is_today_during_the_session_and_the_next_one_after_it() -> None:
    sessions = [date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 5)]
    assert first_close_session(_utc(2026, 10, 1, 14), sessions) == 0  # 10:00 ET
    assert first_close_session(_utc(2026, 10, 1, 21), sessions) == 1  # 17:00 ET
    assert first_close_session(_utc(2026, 10, 3, 15), sessions) == 2  # Saturday
    assert first_close_session(_utc(2026, 10, 6, 21), sessions) is None  # not yet closed


async def _bars(engine: AsyncEngine, symbol: str, closes: list[float]) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, "
                "volume, fetched_at) VALUES (:s, :d, 'all', :c, :c, :c, :c, 1, now())"
            ),
            [{"s": symbol, "d": d, "c": c} for d, c in zip(SESSIONS, closes, strict=False)],
        )


async def test_labels_are_abnormal_returns_from_the_first_close_and_written_once(
    engine: AsyncEngine,
) -> None:
    n = len(SESSIONS)
    await _bars(engine, "SPUS", [100.0] * n)
    await _bars(engine, "AAPL", [100.0 + i for i in range(n)])
    rec = EventRecorder(engine)
    t = _utc(2026, 9, 28, 15)  # Monday 11:00 ET: d0 = Monday's close
    await rec.record(
        [
            EventRecord("alpaca", "1", "news", "AAPL", t, t, {}, Score("s", 0.9)),
            EventRecord(
                "alpaca", "2", "news", "AAPL", t + timedelta(minutes=5), t, {}, Score("s", 0.2)
            ),
        ]
    )

    assert await label_events(engine) == 6  # 2 events x horizons 1, 5, 20
    assert await label_events(engine) == 0
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT ret, abn_ret FROM event_labels WHERE horizon = 5 LIMIT 1")
            )
        ).one()
    assert abs(row.ret - 0.05) < 1e-9 and abs(row.abn_ret - 0.05) < 1e-9

    rows = await report(engine)
    # Two headlines, one catalyst: counted once, at its highest score.
    assert {(r.horizon, r.events, r.above) for r in rows} == {(1, 1, 1), (5, 1, 1), (20, 1, 1)}


async def test_labels_are_written_a_batch_of_symbols_at_a_time(
    engine: AsyncEngine, monkeypatch
) -> None:
    """One symbol per pass labels every symbol, and a half-labelled event gets only
    the horizons it lacks."""
    from halal_trader.events import labels

    monkeypatch.setattr(labels, "LABEL_BATCH_SYMBOLS", 1)
    n = len(SESSIONS)
    await _bars(engine, "SPUS", [100.0] * n)
    await _bars(engine, "AAPL", [100.0 + i for i in range(n)])
    await _bars(engine, "MSFT", [200.0 - i for i in range(n)])
    t = _utc(2026, 9, 28, 15)
    await EventRecorder(engine).record(
        [
            EventRecord("alpaca", "1", "news", "AAPL", t, t, {}, Score("s", 0.9)),
            EventRecord("alpaca", "2", "news", "MSFT", t, t, {}, Score("s", 0.1)),
        ]
    )
    async with engine.begin() as conn:  # MSFT's 1-day label already there
        await conn.execute(
            text(
                "INSERT INTO event_labels (event_id, horizon, ret, abn_ret, labeled_at) "
                "SELECT id, 1, 0, 0, now() FROM events WHERE symbol = 'MSFT'"
            )
        )
    assert await label_events(engine) == 5  # AAPL's three, MSFT's missing two
    async with engine.connect() as conn:
        by_symbol = dict(
            (
                await conn.execute(
                    text(
                        "SELECT e.symbol, count(*) FROM event_labels l "
                        "JOIN events e ON e.id = l.event_id GROUP BY 1"
                    )
                )
            ).all()
        )
    assert by_symbol == {"AAPL": 3, "MSFT": 3}
