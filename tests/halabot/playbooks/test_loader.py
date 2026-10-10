"""The loader: read_windows, the window guard, the calendar check and every path skip."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.playbooks.loader import (
    CalendarMismatch,
    MinuteBarLoader,
    PathRequest,
    Window,
    WindowGuard,
    WindowLocked,
    WindowUnlock,
    batches_of,
    check_calendar,
    sane,
    unit_set_sha,
)
from halabot.playbooks.types import PathData, PathSkip
from halal_trader.db.repos.quant_trials import QuantTrialRepoImpl, config_hash
from halal_trader.market_hours import is_trading_day
from tests.halabot.playbooks._seed import mark_done, seed_bars, seed_calendar, seed_daily
from tests.halabot.playbooks._support import (
    FRI,
    MON,
    THU,
    TUE,
    WED,
    Context,
    daily,
    epoch,
    session_bars,
)

END = date(2016, 3, 31)

# ── the guard ──


async def test_the_guard_locks_train_and_validation_behind_a_preregistration(
    engine: AsyncEngine,
) -> None:
    repo = QuantTrialRepoImpl(engine)
    cfg = {"hypothesis": "h"}
    pid = await repo.record_trial(name="research.news.h1", kind="preregistration", config=cfg)
    other = await repo.record_trial(name="research.x", kind="backtest", config=cfg)

    plain = WindowGuard(window=Window.TRAIN, window_end=date(2021, 12, 31), unlock=WindowUnlock())
    await plain.verify(engine)
    assert plain.allows("AAA", date(2016, 9, 30))  # before the lock
    assert not plain.allows("AAA", date(2016, 10, 3))
    with pytest.raises(WindowLocked, match="preregistration"):
        plain.check("AAA", date(2016, 10, 3))

    unlocked = WindowGuard(
        window=Window.TRAIN,
        window_end=date(2021, 12, 31),
        unlock=WindowUnlock(prereg_id=pid, config_hash=config_hash(cfg)),
    )
    await unlocked.verify(engine)
    assert unlocked.allows("AAA", date(2016, 10, 3)) and unlocked.allows("SPY", date(2021, 12, 31))
    assert not unlocked.allows("AAA", date(2022, 1, 3))  # past window_end

    for bad in (
        WindowUnlock(prereg_id=pid, config_hash="0" * 12),
        WindowUnlock(prereg_id=other, config_hash=config_hash(cfg)),
        WindowUnlock(prereg_id=10**6, config_hash=config_hash(cfg)),
        WindowUnlock(prereg_id=pid),
    ):
        guard = WindowGuard(window=Window.TRAIN, window_end=date(2021, 12, 31), unlock=bad)
        with pytest.raises(WindowLocked):
            await guard.verify(engine)
    with pytest.raises(WindowLocked):  # a train run cannot reach into validation
        WindowGuard(window=Window.TRAIN, window_end=date(2022, 6, 30), unlock=WindowUnlock())
    with pytest.raises(WindowLocked):
        WindowGuard(window=Window.VALIDATION, window_end=date(2025, 1, 2), unlock=WindowUnlock())


async def test_the_holdout_opens_only_after_a_passing_verdict(engine: AsyncEngine) -> None:
    repo = QuantTrialRepoImpl(engine)
    cfg = {"hypothesis": "h"}
    h = config_hash(cfg)
    pid = await repo.record_trial(name="research.news.h1", kind="preregistration", config=cfg)
    unlock = WindowUnlock(prereg_id=pid, config_hash=h, holdout=True)
    guard = WindowGuard(window=Window.HOLDOUT, window_end=date(2025, 11, 28), unlock=unlock)
    with pytest.raises(WindowLocked, match="passing verdict"):
        await guard.verify(engine)
    await repo.record_trial(name="research.news.h1", kind="verdict", config=cfg, verdict="fail")
    with pytest.raises(WindowLocked):
        await guard.verify(engine)
    await repo.record_trial(
        name="research.news.h1", kind="verdict", config={"other": 1}, verdict="pass"
    )
    with pytest.raises(WindowLocked):
        await guard.verify(engine)
    await repo.record_trial(name="research.news.h1", kind="verdict", config=cfg, verdict="pass")
    await guard.verify(engine)
    assert guard.allows("AAA", date(2025, 2, 3))

    no_holdout = WindowGuard(
        window=Window.HOLDOUT,
        window_end=date(2025, 11, 28),
        unlock=WindowUnlock(prereg_id=pid, config_hash=h),
    )
    await no_holdout.verify(engine)
    assert not no_holdout.allows("AAA", date(2025, 2, 3))
    assert no_holdout.allows("AAA", date(2024, 12, 31))


def test_a_gate_unlock_admits_exactly_its_registered_units() -> None:
    units = frozenset({("AAA", date(2025, 12, 4)), ("BBB", date(2025, 12, 5))})
    unlock = WindowUnlock(gate="reactor", units=units, units_sha=unit_set_sha(units))
    guard = WindowGuard(window=Window.GATE, window_end=date(2026, 10, 9), unlock=unlock)
    assert guard.allows("AAA", date(2025, 12, 4)) and guard.allows("BBB", date(2025, 12, 5))
    assert guard.allows("SPY", date(2025, 12, 4))  # SPY on the days of the set
    assert not guard.allows("BBB", date(2025, 12, 4))
    assert not guard.allows("SPY", date(2025, 12, 8))
    assert not guard.allows("AAA", MON)  # outside the set, even where nothing is locked
    with pytest.raises(WindowLocked, match="sha256"):
        WindowGuard(
            window=Window.GATE,
            window_end=date(2026, 10, 9),
            unlock=WindowUnlock(gate="reactor", units=units, units_sha=unit_set_sha(set())),
        )
    with pytest.raises(WindowLocked):
        WindowGuard(
            window=Window.GATE, window_end=date(2026, 10, 9), unlock=WindowUnlock(gate="sue")
        )
    assert unit_set_sha([("A", MON), ("B", TUE)]) == unit_set_sha([("B", TUE), ("A", MON)])


# ── the calendar ──


async def test_the_calendar_must_match_spys_daily_sessions(engine: AsyncEngine) -> None:
    await seed_calendar(engine, date(2016, 1, 4), END)
    await check_calendar(engine, date(2016, 1, 4), END)
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM daily_bars WHERE symbol = 'SPY' AND day = :d"), {"d": TUE}
        )
    with pytest.raises(CalendarMismatch, match="2016-03-08"):
        await check_calendar(engine, date(2016, 1, 4), END)
    await seed_daily(engine, "SPY", {TUE: 200.0, date(2016, 3, 25): 200.0})  # Good Friday
    with pytest.raises(CalendarMismatch, match="2016-03-25"):
        await check_calendar(engine, date(2016, 1, 4), END)


# ── pure helpers ──


def test_sane_drops_impossible_bars_and_reads_bad_vwaps_as_missing() -> None:
    bars = session_bars(
        MON,
        last=(9, 35),
        rows={
            (9, 31): (10.0, 9.0, 8.0, 9.5, 1.0, 9.5),  # h < max(o, c)
            (9, 32): (10.0, 11.0, 10.5, 10.2, 1.0, 10.1),  # l > min(o, c)
            (9, 33): (0.0, 11.0, 0.0, 10.0, 1.0, 10.0),  # non-positive
            (9, 34): (10.0, 11.0, 9.0, 10.0, 1.0, -1.0),  # bad VWAP: kept, NaN
        },
        price=10.0,
    )
    kept, dropped = sane(bars)
    assert dropped == 3
    assert list(kept.ts) == [epoch(MON, 9, 30), epoch(MON, 9, 34), epoch(MON, 9, 35)]
    assert math.isnan(kept.vw[1])
    clean = session_bars(MON, last=(9, 35))
    assert sane(clean) == (clean, 0)


def test_batches_are_month_local_and_in_session_order() -> None:
    reqs = [
        PathRequest("c", "C", date(2016, 3, 1), 1),
        PathRequest("a", "A", date(2016, 2, 29), 1),
        PathRequest("b", "B", date(2016, 3, 1), 1),
        PathRequest("d", "D", date(2016, 3, 2), 1),
    ]
    got = [[r.story_id for r in b] for b in batches_of(reqs, 2)]
    assert got == [["a"], ["b", "c"], ["d"]]


# ── paths ──


async def _market(engine: AsyncEngine) -> Context:
    """SPY complete MON..THU (FRI not done); symbols for every skip reason."""
    await seed_calendar(engine, date(2016, 1, 4), END)
    for d in (MON, TUE, WED, THU):
        await seed_bars(engine, "SPY", session_bars(d, price=200.0))
    await seed_bars(engine, "SPY", session_bars(FRI, price=200.0, last=(12, 49)))  # 200 bars
    await mark_done(engine, [("SPY", d) for d in (MON, TUE, WED, THU, FRI)])
    await mark_done(engine, [("SPY", date(2016, 3, 14))])
    for d in (MON, TUE, WED, THU):
        await seed_bars(engine, "OK", session_bars(d, price=50.0))
    await mark_done(engine, [("OK", d) for d in (MON, TUE, WED, THU)])
    await seed_bars(engine, "NOTDONE", session_bars(MON))
    await mark_done(engine, [("HALT", MON), ("HOLE", MON)])
    await seed_bars(engine, "NODAILY", session_bars(MON))
    insane = {(9, 30 + i): (10.0, 9.0, 8.0, 9.5, 1.0, 9.5) for i in range(6)}
    await seed_bars(engine, "BAD", session_bars(MON, rows=insane))
    five = {(9, 30 + i): (10.0, 9.0, 8.0, 9.5, 1.0, 9.5) for i in range(5)}
    await seed_bars(engine, "FIVE", session_bars(MON, rows=five))
    await seed_bars(engine, "DEFECT", session_bars(MON))
    await seed_bars(engine, "FRIDAY", session_bars(FRI))
    await seed_bars(engine, "LATE", session_bars(date(2016, 3, 14)))
    await mark_done(
        engine,
        [
            ("NODAILY", MON),
            ("BAD", MON),
            ("FIVE", MON),
            ("DEFECT", MON),
            ("FRIDAY", FRI),
            ("LATE", date(2016, 3, 14)),
        ],
    )
    # DEFECT's adjusted series halves on 2016-02-01 while raw / adjusted holds still:
    # adjusted bars fetched before a split and never re-adjusted (a stale series).
    closes = {
        d: (100.0 if d < date(2016, 2, 1) else 50.0)
        for d in (date(2016, 1, 4) + timedelta(days=i) for i in range(70))
        if is_trading_day(d) and d <= MON
    }
    await seed_daily(engine, "DEFECT", closes, adjusted=closes)
    return Context(
        adj={("NODAILY", MON): None},
        daily={
            ("HOLE", MON): daily(10.0),
            **{("OK", d): daily(50.0) for d in (MON, TUE, WED, THU)},
        },
    )


async def test_paths_and_every_skip_reason(engine: AsyncEngine) -> None:
    ctx = await _market(engine)
    loader = MinuteBarLoader(
        engine, window=Window.GATE, window_end=END, unlock=WindowUnlock(), batch_paths=3
    )
    requests = [
        PathRequest("ok", "OK", MON, 3),
        PathRequest("notdone", "NOTDONE", MON, 1),
        PathRequest("halt", "HALT", MON, 1),
        PathRequest("hole", "HOLE", MON, 1),
        PathRequest("nodaily", "NODAILY", MON, 1),
        PathRequest("bad", "BAD", MON, 1),
        PathRequest("five", "FIVE", MON, 1),
        PathRequest("defect", "DEFECT", MON, 1),
        PathRequest("friday", "FRIDAY", FRI, 1),  # SPY's Friday has 200 bars
        PathRequest("late", "LATE", date(2016, 3, 14), 1),  # SPY marked done with no bars
    ]
    got = {x.story_id: x async for x in loader.paths(requests, ctx)}
    skips = {k: v.reason for k, v in got.items() if isinstance(v, PathSkip)}
    assert skips == {
        "notdone": "units_missing",
        "halt": "halted_all_day",
        "hole": "units_missing",
        "nodaily": "no_daily",
        "bad": "bad_bars",
        "defect": "adjust_defect",
        "friday": "spy_missing",
        "late": "spy_missing",
    }
    ok = got["ok"]
    assert isinstance(ok, PathData)
    assert [s.day for s in ok.sessions] == [MON, TUE, WED]
    assert ok.spare is not None and ok.spare.day == THU and ok.spare_bars is not None
    assert [len(b) for b in ok.bars] == [390, 390, 390] and float(ok.bars[1].c[0]) == 50.0
    five = got["five"]
    assert isinstance(five, PathData) and five.dropped == 5 and len(five.bars[0]) == 385
    spy = await loader.spy()
    assert set(spy.days) >= {MON, TUE, WED, THU}
    assert loader.counts["skip:spy_missing"] == 2 and loader.counts["paths"] == 2


async def test_a_path_past_the_window_end_is_refused(engine: AsyncEngine) -> None:
    ctx = await _market(engine)
    loader = MinuteBarLoader(engine, window=Window.GATE, window_end=TUE, unlock=WindowUnlock())
    with pytest.raises(WindowLocked, match="after the window"):
        async for _ in loader.paths([PathRequest("ok", "OK", MON, 3)], ctx):
            pass
    # A path ending on window_end loads, without a spare beyond it.
    loader = MinuteBarLoader(engine, window=Window.GATE, window_end=WED, unlock=WindowUnlock())
    (item,) = [x async for x in loader.paths([PathRequest("ok", "OK", MON, 3)], ctx)]
    assert isinstance(item, PathData) and item.spare is None


async def test_the_loader_refuses_to_start_on_a_calendar_mismatch(engine: AsyncEngine) -> None:
    await seed_calendar(engine, date(2016, 1, 4), date(2016, 2, 29))  # March missing
    loader = MinuteBarLoader(engine, window=Window.GATE, window_end=END, unlock=WindowUnlock())
    with pytest.raises(CalendarMismatch):
        await loader.spy()


async def test_a_locked_session_is_refused_before_any_read(engine: AsyncEngine) -> None:
    await seed_calendar(engine, date(2016, 10, 3), date(2016, 10, 31))
    loader = MinuteBarLoader(
        engine, window=Window.TRAIN, window_end=date(2016, 10, 31), unlock=WindowUnlock()
    )
    with pytest.raises(WindowLocked, match="preregistration"):
        async for _ in loader.paths([PathRequest("x", "AAA", date(2016, 10, 3), 1)], Context()):
            pass


async def test_a_stale_series_is_found_inside_each_paths_own_span(engine: AsyncEngine) -> None:
    """``corporate_actions`` keeps the first stale date per symbol; a later one is asked for."""
    from halal_trader.market_hours import next_trading_day, trading_days_back

    first = date(2016, 3, 1)
    await seed_calendar(engine, date(2016, 1, 4), END)
    for d in (first, MON):
        await seed_bars(engine, "SPY", session_bars(d, price=200.0))
        await seed_bars(engine, "DEF2", session_bars(d))
    await mark_done(engine, [(s, d) for s in ("SPY", "DEF2") for d in (first, MON)])
    since_first = trading_days_back(first, 63)[0]
    since_mon = trading_days_back(MON, 63)[0]
    stale_old = next_trading_day(since_first)  # inside the 03-01 path's span only
    assert since_first < stale_old <= since_mon
    closes = {}
    d = since_first
    while d <= MON:
        if is_trading_day(d):
            closes[d] = 400.0 if d < stale_old else 200.0 if d < date(2016, 2, 1) else 100.0
        d += timedelta(days=1)
    await seed_daily(engine, "DEF2", closes, adjusted=closes)
    loader = MinuteBarLoader(engine, window=Window.GATE, window_end=END, unlock=WindowUnlock())
    reqs = [PathRequest("old", "DEF2", first, 1), PathRequest("new", "DEF2", MON, 1)]
    got = {x.story_id: x async for x in loader.paths(reqs, Context())}
    assert {k: getattr(v, "reason", None) for k, v in got.items()} == {
        "old": "adjust_defect",
        "new": "adjust_defect",  # the 2016-02-01 jump, found by asking again over its span
    }
