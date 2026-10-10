"""The simulator against W1c's real point-in-time context (``events.context.PitContext``).

The simulator never imports ``events/context``; this test does, to hold the
two to one contract: ``PitContext`` is a ``ContextView``, its ``DailyPoint``
a ``DailyPointLike`` (``open``/``close``), and a multi-session trade's daily
marks and its close fallback read them. Synthetic data only.
"""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.playbooks.interfaces import ContextView, DailyPointLike
from halabot.playbooks.sim import simulate_symbol
from halabot.playbooks.types import SimConfig
from halal_trader.events.context import PitContext
from halal_trader.market_hours import is_trading_day
from tests.halabot.playbooks._support import (
    MON,
    TUE,
    WED,
    Toy,
    downgrade,
    et,
    path,
    session_bars,
    spy_data,
    story,
)

EPS = 1e-12
SPLIT = TUE  # AAA splits 2:1 effective TUE: A(d) = 0.5 before, 1 from TUE


def _a(symbol: str, day: date) -> float:
    return 0.5 if symbol == "AAA" and day < SPLIT else 1.0


RAW_CLOSE = {"AAA": {MON: 100.4, TUE: 50.3, WED: 50.6}, "SPY": {MON: 200.5, TUE: 201.0, WED: 202.0}}


async def _world(engine: AsyncEngine) -> PitContext:
    rows = []
    d = date(2016, 1, 4)
    while d <= date(2016, 3, 31):
        if is_trading_day(d):
            for symbol, base in (("AAA", 100.0 if d < SPLIT else 50.0), ("SPY", 200.0)):
                c = RAW_CLOSE[symbol].get(d, base)
                o = base
                for adjustment, f in (("raw", 1.0), ("all", _a(symbol, d))):
                    rows.append(
                        {
                            "s": symbol,
                            "d": d,
                            "a": adjustment,
                            "o": o * f,
                            "h": max(o, c) * f,
                            "l": min(o, c) * f,
                            "c": c * f,
                        }
                    )
        d += timedelta(days=1)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, volume, "
                "fetched_at) VALUES (:s, :d, :a, :o, :h, :l, :c, 1e6, now())"
            ),
            rows,
        )
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, cik, sic_description, verdict, "
                "reasons, metrics, method, screened_at) VALUES ('2016-03-01', 'AAA', 1, "
                "'SERVICES-PREPACKAGED SOFTWARE', 'halal', '[]', '{}', 'v12', now())"
            )
        )
    return await PitContext.load(engine, symbols=["AAA"], start=MON, end=WED)


async def test_the_simulator_runs_on_a_real_pit_context(engine: AsyncEngine) -> None:
    pit = await _world(engine)
    ctx: ContextView = pit  # the Protocol, satisfied as is
    point = pit.daily("AAA", MON)
    assert point is not None
    as_like: DailyPointLike = point
    assert (as_like.open, as_like.close) == (100.0, 100.4)
    assert pit.adj("AAA", MON) == 0.5 and pit.adj("AAA", TUE) == 1.0
    assert pit.screen_verdict("AAA", MON) == "halal"

    st = story("AAA", MON, downgrade(MON))
    days = [MON, TUE, WED]
    bars = [
        session_bars(MON, price=100.0),
        session_bars(TUE, price=50.0),
        session_bars(WED, price=50.0, last=(15, 50)),  # no print after 15:50: the close fallback
    ]
    (out,) = simulate_symbol(
        "AAA",
        [st],
        lambda s: Toy(s, sessions=3, target=1.0),
        {st.story_id: path(st.story_id, "AAA", days, bars)},
        spy_data(days),
        ctx,
        SimConfig(),
    )
    t = out.trade
    assert t is not None
    assert t.entry_bar_ts == et(MON, 9, 51) and t.entry_px == 100.0
    assert t.flags == ("close_fallback",) and t.exit_px == 50.6 and t.spy_exit_px == 202.0
    assert (t.adj_entry, t.adj_exit) == (0.5, 1.0)
    assert abs(t.r_gross - (50.6 * 1.0 / (100.0 * 0.5) - 1)) < EPS  # the split is not a loss
    assert abs(t.r_spy - (202.0 / 200.0 - 1)) < EPS
    mon, tue, wed = out.legs
    # Daily marks are the official closes in S units: close * A(d) / A(S).
    assert abs(mon.z - 100.4) < EPS and abs(tue.a - 100.4) < EPS
    assert abs(tue.z - 50.3 / 0.5) < EPS and abs(wed.z - 50.6 / 0.5) < EPS
    assert (mon.spy_z, tue.spy_z, wed.spy_z) == (200.5, 201.0, 202.0)
