"""The loader: read_windows, the window guard, the calendar check and every path skip."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.playbooks.loader import (
    GATE_RANGES,
    GATE_UNITS_NAME,
    SUE_FIRST,
    SUE_MAX_HORIZON,
    CalendarMismatch,
    MinuteBarLoader,
    PathRequest,
    Window,
    WindowGuard,
    WindowLocked,
    WindowUnlock,
    batches_of,
    check_calendar,
    check_gate_units,
    gate_pins,
    register_gate_units,
    sane,
    sue_last_exit,
    unit_set_sha,
)
from halabot.playbooks.types import DATA_SKIPS, PathData, PathSkip
from halal_trader.db.repos.quant_trials import QuantTrialRepoImpl, config_hash
from halal_trader.events import study
from halal_trader.market_hours import MARKET_TZ, is_trading_day
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


async def test_only_the_latest_verdict_of_the_trial_opens_the_holdout(engine: AsyncEngine) -> None:
    """An old pass opens nothing: an amended fail, or a window or amendment after it, close."""
    repo = QuantTrialRepoImpl(engine)
    cfg = {"hypothesis": "h"}
    h = config_hash(cfg)
    pid = await repo.record_trial(name="research.news.h1", kind="preregistration", config=cfg)

    async def opens() -> bool:
        unlock = WindowUnlock(prereg_id=pid, config_hash=h, holdout=True)
        guard = WindowGuard(window=Window.HOLDOUT, window_end=date(2025, 11, 28), unlock=unlock)
        try:
            await guard.verify(engine)
        except WindowLocked:
            return False
        return guard.allows("AAA", date(2025, 2, 3))

    async def record(kind: str, verdict: str | None = None, name: str = "research.news.h1") -> None:
        await repo.record_trial(name=name, kind=kind, config=cfg, verdict=verdict)

    await record("verdict", "pass", name="research.other")  # another trial's verdict
    assert not await opens()
    await record("verdict", "pass")
    assert await opens()
    await record("amendment")  # train is rerun: the pass no longer stands
    assert not await opens()
    await record("window")
    assert not await opens()
    await record("verdict", "fail")  # the amended verdict
    assert not await opens()
    await record("verdict", "pass")  # passing again, nothing after it
    assert await opens()
    await record("stage-a")
    assert not await opens()


REACTOR_END = date(2026, 10, 9)


def _gate(gate: str, units: frozenset[tuple[str, date]], sha: str | None = None) -> WindowGuard:
    unlock = WindowUnlock(gate=gate, units=units, units_sha=sha or unit_set_sha(units))  # type: ignore[arg-type]
    return WindowGuard(window=Window.GATE, window_end=REACTOR_END, unlock=unlock)


async def test_a_gate_unlock_admits_exactly_its_pinned_units(engine: AsyncEngine) -> None:
    units = frozenset({("AAA", date(2025, 12, 4)), ("BBB", date(2025, 12, 5))})
    sha = await register_gate_units(engine, "reactor", units)
    assert sha == unit_set_sha(units)
    guard = _gate("reactor", units)
    assert not guard.allows("AAA", date(2025, 12, 4))  # nothing opens before the ledger check
    await guard.verify(engine)
    assert guard.allows("AAA", date(2025, 12, 4)) and guard.allows("BBB", date(2025, 12, 5))
    assert guard.allows("SPY", date(2025, 12, 4))  # SPY on the days of the set
    assert not guard.allows("BBB", date(2025, 12, 4))
    assert not guard.allows("SPY", date(2025, 12, 8))
    assert not guard.allows("AAA", MON)  # outside the set, even where nothing is locked
    with pytest.raises(WindowLocked, match="sha256"):
        _gate("reactor", units, unit_set_sha({("AAA", date(2025, 12, 4))}))
    with pytest.raises(WindowLocked):
        WindowGuard(window=Window.GATE, window_end=REACTOR_END, unlock=WindowUnlock(gate="sue"))
    assert unit_set_sha([("A", MON), ("B", TUE)]) == unit_set_sha([("B", TUE), ("A", MON)])


async def test_an_unpinned_gate_set_is_refused(engine: AsyncEngine) -> None:
    """A set whose sha matches itself still opens nothing until the gate code pins it."""
    pinned = frozenset({("AAA", date(2025, 12, 4))})
    await register_gate_units(engine, "reactor", pinned)
    other = frozenset({("AAA", date(2025, 12, 4)), ("MSFT", date(2026, 3, 2))})
    guard = _gate("reactor", other)  # self-consistent, but never registered
    with pytest.raises(WindowLocked, match="not pinned"):
        await guard.verify(engine)
    assert not guard.allows("MSFT", date(2026, 3, 2))
    # The same set pinned for another gate does not count either.
    g1 = frozenset({("AAA", date(2016, 3, 7))})
    await register_gate_units(engine, "g1", g1)
    with pytest.raises(WindowLocked, match="not pinned"):
        await _gate("calib", g1).verify(engine)
    await _gate("g1", g1).verify(engine)


async def test_a_gate_reads_only_inside_its_dates(engine: AsyncEngine) -> None:
    assert GATE_RANGES == {
        "g1": (date(2016, 1, 4), date(2016, 9, 30)),
        "calib": (date(2016, 1, 4), date(2016, 9, 30)),
        "sue": (date(2016, 1, 4), date(2020, 1, 30)),
        "reactor": (date(2025, 12, 1), date(2026, 10, 9)),
    }
    cases = [
        ("g1", ("AAPL", date(2016, 10, 3))),  # train
        ("calib", ("AAPL", date(2015, 12, 31))),
        ("sue", ("MSFT", date(2020, 1, 31))),  # past the last h=20 exit
        ("reactor", ("AAPL", date(2025, 6, 2))),  # the clean holdout
        ("reactor", ("MSFT", date(2019, 5, 1))),
    ]
    for gate, bad in cases:
        units = frozenset({bad, ("SPY", GATE_RANGES[gate][0])})
        with pytest.raises(WindowLocked, match="outside"):
            await register_gate_units(engine, gate, units)  # type: ignore[arg-type]
        with pytest.raises(WindowLocked, match="outside"):
            _gate(gate, units)  # even with a matching sha
    async with engine.connect() as conn:
        rows = await conn.scalar(text("SELECT count(*) FROM quant_trials"))
    assert rows == 0  # nothing out of range was ever pinned
    with pytest.raises(WindowLocked, match="empty"):
        await register_gate_units(engine, "g1", [])
    edges = frozenset({("AAA", date(2016, 1, 4)), ("AAA", date(2016, 9, 30))})
    assert await register_gate_units(engine, "g1", edges) == unit_set_sha(edges)


def test_the_sue_range_ends_at_the_last_h20_exit_of_2019() -> None:
    """An observation published on 2019-12-31 holds its h=20 exit in the gate's set."""
    days = (date(2019, 12, 31) + timedelta(days=i) for i in range(40))
    sessions = [d for d in days if is_trading_day(d)]
    published = (
        datetime(2019, 12, 31, 15, 0, tzinfo=MARKET_TZ),  # a close entry on 12-31
        datetime(2019, 12, 31, 18, 0, tzinfo=MARKET_TZ),  # an open entry on 2020-01-02
        datetime(2019, 12, 31, 23, 59, tzinfo=UTC),  # the year's last instant, read in UTC
    )
    exits = set()
    for t in published:
        point = study.entry_point(t, sessions)
        assert point is not None
        i, at = point
        exits.add(sessions[i + SUE_MAX_HORIZON - (1 if at == "open" else 0)])
    assert exits == {sue_last_exit()} == {date(2020, 1, 30)}  # MLK day 2020-01-20 skipped
    assert GATE_RANGES["sue"] == (SUE_FIRST, sue_last_exit())
    assert check_gate_units("sue", [("MSFT", date(2020, 1, 30)), ("MSFT", date(2016, 1, 4))])
    with pytest.raises(WindowLocked, match="outside"):
        check_gate_units("sue", [("MSFT", date(2020, 1, 31))])


async def test_pinning_is_idempotent_and_never_a_trial(engine: AsyncEngine) -> None:
    units = [("AAA", date(2017, 5, 1)), ("BBB", date(2018, 2, 1))]
    first = await register_gate_units(engine, "sue", units)
    again = await register_gate_units(engine, "sue", list(reversed(units)))
    assert first == again
    async with engine.connect() as conn:
        rows = (
            await conn.execute(text("SELECT name, kind, config, metrics FROM quant_trials"))
        ).all()
    assert len(rows) == 1
    (row,) = rows
    assert (row.name, row.kind) == (GATE_UNITS_NAME, "gate-units")
    assert row.config == {"gate": "sue", "units_sha": first}
    assert "active_sr_period" not in (row.metrics or {})  # never counted for the DSR


async def test_a_gate_keeps_its_first_pin(engine: AsyncEngine) -> None:
    """One pin per gate: a different set is refused, the same set again is a no-op."""
    first = frozenset({("AAA", date(2017, 5, 1)), ("AAA", date(2017, 5, 8))})
    sha = await register_gate_units(engine, "sue", first)
    widened = first | {("MSFT", date(2019, 5, 1))}  # a train-era unit slipped into the set
    with pytest.raises(WindowLocked, match="already pinned"):
        await register_gate_units(engine, "sue", widened)
    with pytest.raises(WindowLocked, match="already pinned"):
        await register_gate_units(engine, "sue", [("AAA", date(2017, 5, 1))])
    assert await register_gate_units(engine, "sue", first) == sha
    assert await gate_pins(engine, "sue") == {sha}
    with pytest.raises(WindowLocked, match="not pinned"):
        await _gate("sue", widened).verify(engine)
    await _gate("sue", first).verify(engine)
    # Another gate pins its own set.
    calib = frozenset({("AAA", date(2016, 3, 7))})
    assert await register_gate_units(engine, "calib", calib) == unit_set_sha(calib)
    async with engine.connect() as conn:
        assert await conn.scalar(text("SELECT count(*) FROM quant_trials")) == 2


async def test_a_pin_must_hash_to_the_expected_sha(engine: AsyncEngine) -> None:
    """The units plan passes its own sha: a set that differs from it is never pinned."""
    plan = frozenset({("AAA", date(2016, 3, 7)), ("BBB", date(2016, 3, 8))})
    want = unit_set_sha(plan)
    with pytest.raises(WindowLocked, match="not the expected"):
        await register_gate_units(
            engine, "g1", plan - {("BBB", date(2016, 3, 8))}, expected_sha=want
        )
    assert await gate_pins(engine, "g1") == set()
    assert await register_gate_units(engine, "g1", plan, expected_sha=want) == want


async def test_two_pins_for_one_gate_open_neither(engine: AsyncEngine) -> None:
    """A race that left two pins (or a row written by hand) keeps the gate shut."""
    a = frozenset({("AAA", date(2025, 12, 4))})
    b = frozenset({("BBB", date(2025, 12, 5))})
    repo = QuantTrialRepoImpl(engine)
    for units in (a, b):  # what two concurrent registrations would leave
        await repo.record_trial(
            name=GATE_UNITS_NAME,
            kind="gate-units",
            config={"gate": "reactor", "units_sha": unit_set_sha(units)},
        )
    assert await gate_pins(engine, "reactor") == {unit_set_sha(a), unit_set_sha(b)}
    for units in (a, b):
        guard = _gate("reactor", units)
        with pytest.raises(WindowLocked, match="2 pinned unit sets"):
            await guard.verify(engine)
        assert not guard.allows(*next(iter(units)))
        with pytest.raises(WindowLocked, match="already pinned"):
            await register_gate_units(engine, "reactor", units)


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
        "friday": "spy_thin",  # bars, but fewer than 300
        "late": "spy_missing",  # none at all
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
    assert loader.counts["skip:spy_missing"] == 1 and loader.counts["skip:spy_thin"] == 1
    assert loader.counts["paths"] == 2
    data = {r for r in skips.values() if r in DATA_SKIPS}  # Stage A's data skips
    assert data == {"units_missing", "bad_bars", "adjust_defect", "spy_thin", "spy_missing"}


async def test_a_spy_session_without_an_a_factor_is_spy_missing(engine: AsyncEngine) -> None:
    """SPY's bars are complete on TUE, but it has no A-factor there: missing, not thin."""
    await _market(engine)
    loader = MinuteBarLoader(engine, window=Window.GATE, window_end=END, unlock=WindowUnlock())
    no_adj = Context(adj={("SPY", TUE): None})
    (item,) = [x async for x in loader.paths([PathRequest("ok", "OK", MON, 3)], no_adj)]
    assert isinstance(item, PathSkip) and item.reason == "spy_missing"


async def test_bars_cut_on_the_spare_session_are_counted_apart(engine: AsyncEngine) -> None:
    """The spare is not a path session: its cut bars never make bad_bars, and count apart."""
    await seed_calendar(engine, date(2016, 1, 4), END)
    for d in (MON, TUE):
        await seed_bars(engine, "SPY", session_bars(d, price=200.0))
    insane = {(9, 30 + i): (10.0, 9.0, 8.0, 9.5, 1.0, 9.5) for i in range(7)}
    await seed_bars(engine, "SPARE", session_bars(MON))
    await seed_bars(engine, "SPARE", session_bars(TUE, rows=insane))  # 7 > MAX_DROPPED
    await mark_done(engine, [("SPY", MON), ("SPY", TUE), ("SPARE", MON), ("SPARE", TUE)])
    loader = MinuteBarLoader(engine, window=Window.GATE, window_end=END, unlock=WindowUnlock())
    (item,) = [x async for x in loader.paths([PathRequest("s", "SPARE", MON, 1)], Context())]
    assert isinstance(item, PathData)
    assert item.spare is not None and item.spare.day == TUE and item.spare_bars is not None
    assert (item.dropped, item.spare_dropped, len(item.spare_bars)) == (0, 7, 383)
    assert loader.counts["spare_dropped_bars"] == 7 and loader.counts["dropped_bars"] == 0


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


def test_check_refuses_every_unreadable_path_day_up_front() -> None:
    """No database: the guard alone, before any batch would load."""
    loader = MinuteBarLoader(None, window=Window.GATE, window_end=TUE, unlock=WindowUnlock())  # type: ignore[arg-type]
    loader.check([PathRequest("a", "OK", MON, 2)])  # MON..TUE
    with pytest.raises(WindowLocked, match="2016-03-09 is after the window's end"):
        loader.check([PathRequest("a", "OK", MON, 2), PathRequest("b", "OK", MON, 3)])
    train = MinuteBarLoader(
        None,  # type: ignore[arg-type]
        window=Window.TRAIN,
        window_end=date(2021, 12, 31),
        unlock=WindowUnlock(),
    )
    with pytest.raises(WindowLocked, match="preregistration"):
        train.check([PathRequest("c", "AAA", date(2016, 9, 30), 2)])  # crosses into 2016-10-03


async def test_a_pinned_gate_set_loads_its_paths_and_nothing_else(engine: AsyncEngine) -> None:
    ctx = await _market(engine)
    units = frozenset({("OK", MON), ("OK", TUE), ("OK", WED)})
    await register_gate_units(engine, "g1", units)
    unlock = WindowUnlock(gate="g1", units=units, units_sha=unit_set_sha(units))
    loader = MinuteBarLoader(engine, window=Window.GATE, window_end=END, unlock=unlock)
    await loader.prepare()
    loader.check([PathRequest("ok", "OK", MON, 3)])
    with pytest.raises(WindowLocked, match="outside gate 'g1'"):
        loader.check([PathRequest("five", "FIVE", MON, 1)])
    (item,) = [x async for x in loader.paths([PathRequest("ok", "OK", MON, 3)], ctx)]
    assert isinstance(item, PathData) and item.spare is None  # THU is not in the set


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
