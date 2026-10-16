"""The day-trader's round trips scored against SPY (trading/decision_quality.py)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.trading import decision_quality as dq

DAY = date(2026, 10, 14)
OPEN = datetime(2026, 10, 14, 13, 30, tzinfo=UTC)  # 09:30 New York (EDT)


def _trip(
    symbol: str = "AAA",
    *,
    at: timedelta = timedelta(minutes=30),
    held: timedelta = timedelta(hours=1),
    fill: float = 100.0,
    exit_price: float = 101.0,
    reason: str = "llm_sell",
) -> dq.Trip:
    return dq.Trip(symbol, OPEN + at, OPEN + at + held, fill, exit_price, 10, reason, "cycle")


def _spy(*points: tuple[timedelta, float]) -> list[tuple[datetime, float]]:
    return [(OPEN + offset, price) for offset, price in points]


def test_spy_price_is_the_last_minute_bar_ended_by_then() -> None:
    bars = _spy(
        (timedelta(minutes=0), 1.0), (timedelta(minutes=1), 2.0), (timedelta(minutes=2), 3.0)
    )
    assert dq.price_at(bars, OPEN) is None  # the first bar has not ended
    assert dq.price_at(bars, OPEN + timedelta(minutes=1)) == 1.0
    assert dq.price_at(bars, OPEN + timedelta(minutes=2, seconds=30)) == 2.0
    assert dq.price_at(bars, OPEN + timedelta(hours=1)) == 3.0


def test_a_trip_is_scored_on_entry_exit_and_the_whole_trip() -> None:
    # Bought at 100 at 10:00 with SPY at 500; sold at 101 at 11:00 with SPY at 502.5;
    # at the close the name is 99 and SPY 505.
    trip = _trip()
    spy = _spy((timedelta(minutes=29), 500.0), (timedelta(minutes=89), 502.5))
    closes = {("AAA", DAY): 99.0, ("SPY", DAY): 505.0}
    (s,) = dq.score([trip], closes, spy)
    assert s.entry_edge == pytest.approx((99 / 100 - 1) - (505 / 500 - 1))  # -2%
    assert s.exit_edge == pytest.approx((505 / 502.5 - 1) - (99 / 101 - 1))  # sold well
    assert s.excess == pytest.approx((101 / 100 - 1) - (502.5 / 500 - 1))


def test_a_trip_with_a_missing_price_is_left_unscored() -> None:
    spy = _spy((timedelta(minutes=29), 500.0), (timedelta(minutes=89), 502.5))
    assert dq.score([_trip()], {("AAA", DAY): 99.0}, spy) == []  # no SPY close
    assert dq.score([_trip()], {("AAA", DAY): 99.0, ("SPY", DAY): 505.0}, []) == []


def test_each_session_is_one_draw() -> None:
    d1, d2 = date(2026, 10, 13), date(2026, 10, 14)
    s = dq.clustered([(d1, 0.01), (d1, 0.03), (d2, -0.01)])
    assert (s.trips, s.sessions) == (3, 2)
    assert s.mean == pytest.approx((0.02 - 0.01) / 2)
    assert s.t == pytest.approx(0.005 / (0.0212132 / 2**0.5), rel=1e-4)
    assert dq.clustered([]).mean is None
    assert dq.clustered([(d1, 0.01)]).t is None


def test_the_summary_splits_by_hour_and_exit_and_leaves_closing_exits_out_of_sells() -> None:
    spy = _spy(*((timedelta(minutes=m), 500.0) for m in range(391)))
    closes = {("AAA", DAY): 100.0, ("BBB", DAY): 102.0, ("SPY", DAY): 500.0}
    scores = dq.score(
        [
            _trip("AAA", at=timedelta(minutes=5), exit_price=99.0, reason="stop_loss"),
            _trip(
                "BBB",
                at=timedelta(hours=4),
                held=timedelta(hours=2, minutes=29),
                exit_price=102.0,
                reason="eod_close_all",
            ),
        ],
        closes,
        spy,
    )
    out = dq.Report(DAY - timedelta(days=29), DAY, scores).summary()
    assert out["trips"] == 2
    assert out["pnl_usd"] == pytest.approx(-10.0 + 20.0)
    by_hour: Any = out["by_hour"]
    assert by_hour["09:30-10:30"]["trips"] == 1 and by_hour["13:00-16:00"]["trips"] == 1
    assert by_hour["10:30-13:00"]["trips"] == 0
    assert list(out["by_exit"]) == ["stop_loss"]  # type: ignore[call-overload]
    exit_edge: Any = out["exit_edge"]
    assert exit_edge["trips"] == 1
    assert exit_edge["mean"] == pytest.approx(-(100 / 99 - 1), abs=1e-6)  # sold too early
    days: Any = out["days"]
    assert days == [{"day": "2026-10-14", "trips": 2, "entry_edge": 0.01, "pnl_usd": 10.0}]


async def _insert_trip(engine: AsyncEngine, trip: dq.Trip, *, closed: bool = True) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO trades (symbol, side, quantity, status, timestamp, filled_at, "
                "filled_price, filled_quantity, exit_price, exit_reason, closed_at) VALUES "
                "(:s, 'buy', :q, 'closed', :t, :t, :f, :q, :x, :r, :c)"
            ),
            {
                "s": trip.symbol,
                "q": trip.quantity,
                "t": trip.filled_at,
                "f": trip.fill,
                "x": trip.exit,
                "r": trip.exit_reason,
                "c": trip.closed_at if closed else None,
            },
        )


async def _insert_close(engine: AsyncEngine, symbol: str, close: float) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, volume, "
                "fetched_at) VALUES (:s, :d, 'raw', :c, :c, :c, :c, 1, now())"
            ),
            {"s": symbol, "d": DAY, "c": close},
        )


class _Minutes:
    def __init__(self) -> None:
        self.asked: list[tuple[str, datetime, datetime]] = []

    async def minute_bars(
        self, symbol: str, *, start: datetime, end: datetime
    ) -> list[dict[str, Any]]:
        self.asked.append((symbol, start, end))
        return [
            {"t": (OPEN + timedelta(minutes=m)).isoformat().replace("+00:00", "Z"), "c": 500.0}
            for m in range(390)
        ]


async def test_the_report_reads_closed_trips_and_stored_closes(engine: AsyncEngine) -> None:
    await _insert_trip(engine, _trip("AAA"))
    await _insert_trip(engine, _trip("BBB"), closed=False)  # still open: not scored
    await _insert_trip(engine, _trip("OLD", at=timedelta(days=-40)))  # outside 30 days
    await _insert_close(engine, "AAA", 102.0)
    await _insert_close(engine, "SPY", 500.0)
    market = _Minutes()

    r = await dq.report(engine, market, DAY)

    assert [s.trip.symbol for s in r.scores] == ["AAA"]
    assert r.scores[0].entry_edge == pytest.approx(0.02)
    assert r.unscored == 0
    assert market.asked[0][0] == "SPY"


async def test_no_trips_ask_nothing_of_the_market(engine: AsyncEngine) -> None:
    market = _Minutes()
    r = await dq.report(engine, market, DAY)
    assert r.scores == [] and market.asked == []
    assert r.summary()["trips"] == 0


async def test_the_evening_step_keeps_the_summary_in_a_heartbeat(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    from halal_trader.core.heartbeat import read_beat
    from halal_trader.research import daily

    await _insert_trip(engine, _trip("AAA"))
    await _insert_close(engine, "AAA", 102.0)
    await _insert_close(engine, "SPY", 500.0)

    class Market(_Minutes):
        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(daily.AlpacaMarketData, "from_settings", classmethod(lambda c, s: Market()))
    await daily._decision_quality(engine, object(), DAY)  # type: ignore[arg-type]

    b = await read_beat(engine, dq.BEAT)
    assert b is not None and b.detail is not None
    assert b.detail["trips"] == 1
