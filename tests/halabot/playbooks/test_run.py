"""The driver: checks before any write, no state shared between runs, how workers run."""

from __future__ import annotations

import asyncio
import logging
import math
import sys
import threading
from concurrent.futures.process import BrokenProcessPool
from datetime import date, timedelta
from multiprocessing.reduction import ForkingPickler

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.playbooks import sim
from halabot.playbooks.legacy import (
    HarnessStory,
    HoldFactory,
    r1_dropped,
    reactor_config,
    reactor_story,
)
from halabot.playbooks.loader import (
    Window,
    WindowLocked,
    WindowUnlock,
    register_gate_units,
    unit_set_sha,
)
from halabot.playbooks.playbook import Factory, PlaybookFactory
from halabot.playbooks.records import MemorySink, PgOutcomeSink, outcomes_sha256
from halabot.playbooks.sim import pool_method, run
from halabot.playbooks.types import SimConfig
from halal_trader.core import events
from halal_trader.data import minutes
from halal_trader.events import intraday
from halal_trader.market_hours import MARKET_TZ, is_trading_day
from tests.halabot.playbooks._seed import (
    mark_done,
    seed_bars,
    seed_calendar,
    seed_daily,
    seed_market,
)
from tests.halabot.playbooks._support import (
    FACTS,
    MON,
    TUE,
    WED,
    Context,
    Toy,
    ToyFactory,
    downgrade,
    et,
    session_bars,
    story,
)
from tests.halabot.playbooks._synth import WEEK, Market, random_market

END = date(2016, 3, 31)


async def _go(  # type: ignore[no-untyped-def]
    engine: AsyncEngine,
    market: Market,
    *,
    sink=None,
    batch: int = 250,
    window_end: date = END,
    unlock: WindowUnlock | None = None,
    cfg: SimConfig | None = None,
    parallel=False,
):
    sink = sink if sink is not None else MemorySink(run_id="00000000-0000-0000-0000-00000000000a")
    summary = await run(
        engine,
        market.stories,
        ToyFactory(sessions=market.sessions, target=0.005),
        context=market.ctx,
        window=Window.GATE,
        window_end=window_end,
        cfg=cfg or SimConfig(),
        unlock=unlock or WindowUnlock(),
        sink=sink,
        workers=3,
        batch_paths=batch,
        parallel=parallel,
    )
    return sink, summary


async def test_two_runs_sharing_an_event_loop_match_each_run_alone(engine: AsyncEngine) -> None:
    """Paths of 3 and 1 sessions: a run that read the other's state would fail or differ."""
    a = random_market(21, 10, sessions=3, prefix="A")
    b = random_market(22, 10, sessions=1, prefix="B")
    await seed_calendar(engine, date(2016, 1, 4), END)
    await seed_market(engine, a)
    await seed_market(engine, b)
    alone_a, sum_a = await _go(engine, a, batch=2)
    alone_b, sum_b = await _go(engine, b, batch=2)
    (both_a, _), (both_b, _) = await asyncio.gather(
        _go(engine, a, batch=2), _go(engine, b, batch=2)
    )
    assert sum_a.trades > 0 and sum_b.trades > 0
    assert outcomes_sha256(both_a.outcomes) == outcomes_sha256(alone_a.outcomes)
    assert outcomes_sha256(both_b.outcomes) == outcomes_sha256(alone_b.outcomes)
    assert {o.symbol[0] for o in both_a.outcomes} == {"A"}
    assert {o.symbol[0] for o in both_b.outcomes} == {"B"}


async def test_a_path_past_the_window_end_fails_before_anything_is_written(
    halabot_engine: AsyncEngine,
) -> None:
    engine = halabot_engine
    market = random_market(23, 4, sessions=3)
    await seed_calendar(engine, date(2016, 1, 4), END)
    await seed_market(engine, market)
    sink = PgOutcomeSink(engine)
    with pytest.raises(WindowLocked, match="after the window's end"):
        await _go(engine, market, sink=sink, window_end=WEEK[1])  # WEEK[0] paths end WEEK[2]
    memory = MemorySink()
    with pytest.raises(WindowLocked):
        await _go(engine, market, sink=memory, window_end=WEEK[1])
    assert memory.info is None and memory.outcomes == []
    async with engine.connect() as conn:
        assert await conn.scalar(text("SELECT count(*) FROM hb_playbook_run")) == 0
    # The same stories within the window run, and write.
    _, summary = await _go(engine, market, sink=PgOutcomeSink(engine), window_end=WEEK[-1])
    async with engine.connect() as conn:
        assert await conn.scalar(text("SELECT count(*) FROM hb_playbook_run")) == 1
        stories = await conn.scalar(text("SELECT count(*) FROM hb_playbook_story"))
    assert stories == summary.outcomes > 0


async def test_an_unpinned_gate_fails_before_anything_is_written(engine: AsyncEngine) -> None:
    market = random_market(24, 3, sessions=1)
    await seed_calendar(engine, date(2016, 1, 4), END)
    await seed_market(engine, market)
    units = frozenset((p.symbol, s.day) for p in market.paths.values() for s in p.sessions)
    unlock = WindowUnlock(gate="g1", units=units, units_sha=unit_set_sha(units))
    sink = MemorySink()
    with pytest.raises(WindowLocked, match="not pinned"):
        await _go(engine, market, sink=sink, unlock=unlock)
    assert sink.info is None
    await register_gate_units(engine, "g1", units)
    _, summary = await _go(engine, market, sink=sink, unlock=unlock)
    assert sink.info is not None and summary.outcomes > 0


async def test_a_gate_only_fill_model_needs_a_gate_unlock(engine: AsyncEngine) -> None:
    """R1's shape through the driver: a reactor headline, the legacy fills, a pinned set."""
    await seed_calendar(engine, date(2016, 1, 4), END)
    rows = {(10, 16): (50.0, 50.5, 49.5, 50.2, 1_000.0, 50.9)}  # a VWAP above the high: kept
    rows[(15, 59)] = (51.0, 51.5, 50.5, 51.3, 1_000.0, 51.1)
    await seed_bars(engine, "AAA", session_bars(MON, price=50.0, rows=rows))
    spy_rows = {(10, 16): (200.0, 201.0, 199.0, 200.5, 1e5, 200.4)}
    spy_rows[(15, 59)] = (202.0, 202.5, 201.5, 202.2, 1e5, 202.1)
    await seed_bars(engine, "SPY", session_bars(MON, price=200.0, rows=spy_rows))
    await mark_done(engine, [("AAA", MON), ("SPY", MON)])
    st = reactor_story("AAA:r1", "AAA", et(MON, 10, 14, 30))  # decides at 10:15:30
    factory = HoldFactory(1, {st.story_id: FACTS})

    async def go(unlock: WindowUnlock, sink: MemorySink):  # type: ignore[no-untyped-def]
        return await run(
            engine,
            [st],
            factory,
            context=Context(verdicts={("AAA", MON): "not_halal"}),  # not read by the gate model
            window=Window.GATE,
            window_end=END,
            cfg=reactor_config(),
            unlock=unlock,
            sink=sink,
            workers=1,
        )

    sink = MemorySink()
    with pytest.raises(ValueError, match="gate runs only"):
        await go(WindowUnlock(), sink)
    assert sink.info is None
    units = frozenset({("AAA", MON)})
    await register_gate_units(engine, "g1", units)
    summary = await go(WindowUnlock(gate="g1", units=units, units_sha=unit_set_sha(units)), sink)
    assert summary.trades == 1 and sink.info is not None
    assert sink.info.sim["fill"] == "legacy-reactor" and sink.info.playbook == "gate-hold"
    t = sink.outcomes[0].trade
    assert t is not None
    assert t.entry_active_at == et(MON, 10, 15, 30) and t.entry_bar_ts == et(MON, 10, 16)
    assert (t.entry_px, t.spy_entry_px) == (50.9, 200.4)  # VWAPs, unclamped
    assert t.exit_bar_ts == et(MON, 15, 59) and (t.exit_px, t.spy_exit_px) == (51.3, 202.2)
    assert t.flags == ("last_close",) and t.exit_reason == "time_stop"


GOOD_FRIDAY = date(2016, 3, 25)  # a weekday, not a session
HEADS = {  # headline id -> (symbol, published at)
    "h-ok": ("AAA", et(MON, 10, 14, 30)),
    "h-late": ("BBB", et(MON, 15, 20)),  # no stock bar after 15:21
    "h-nospy": ("CCC", et(TUE, 15, 20)),  # no SPY bar after 15:21
    "h-holiday": ("DDD", et(GOOD_FRIDAY, 11, 0)),
    "h-bad": ("EEE", et(MON, 11, 0)),
    "h-thin": ("FFF", et(WED, 11, 0)),
    "h-jump": ("III", et(MON, 11, 0)),
    "h-unbuilt": ("GGG", et(MON, 11, 0)),  # in the set, but no story is built for it
}


async def _headlines(
    engine: AsyncEngine,
) -> tuple[list[HarnessStory], dict[str, tuple[str, date]], WindowUnlock]:
    """Reactor headlines, each meeting one way to have no result, and a pinned unit set."""
    await seed_calendar(engine, date(2016, 1, 4), END)
    insane = (10.0, 9.0, 8.0, 9.5, 1.0, 9.5)
    await seed_bars(engine, "SPY", session_bars(MON, price=200.0))
    await seed_bars(engine, "SPY", session_bars(TUE, price=200.0, last=(15, 0)))  # 331 bars
    await seed_bars(engine, "SPY", session_bars(WED, price=200.0, last=(12, 49)))  # 200: thin
    await seed_bars(engine, "AAA", session_bars(MON, price=50.0))
    await seed_bars(engine, "BBB", session_bars(MON, last=(15, 0)))
    await seed_bars(engine, "CCC", session_bars(TUE))
    await seed_bars(engine, "EEE", session_bars(MON, rows={(9, 30 + i): insane for i in range(6)}))
    await seed_bars(engine, "FFF", session_bars(WED))
    jump = {(15, 59): (120.0, 121.0, 119.0, 120.0, 1_000.0, 120.0)}  # 50 -> 120: implausible
    await seed_bars(engine, "III", session_bars(MON, price=50.0, rows=jump))
    stories = [reactor_story(i, s, t) for i, (s, t) in HEADS.items() if i != "h-unbuilt"]
    expected = {i: (s, t.astimezone(MARKET_TZ).date()) for i, (s, t) in HEADS.items()}
    done = [u for u in expected.values() if u[1] != GOOD_FRIDAY]
    await mark_done(engine, [*done, ("SPY", MON), ("SPY", TUE), ("SPY", WED)])
    units = frozenset(expected.values())
    await register_gate_units(engine, "g1", units)
    return stories, expected, WindowUnlock(gate="g1", units=units, units_sha=unit_set_sha(units))


async def _reactor_run(  # type: ignore[no-untyped-def]
    engine: AsyncEngine,
    stories: list[HarnessStory],
    expected: dict[str, tuple[str, date]],
    unlock: WindowUnlock,
    sink: MemorySink,
):
    return await run(
        engine,
        stories,
        HoldFactory(1, {s.story_id: FACTS for s in stories}),
        context=Context(),
        window=Window.GATE,
        window_end=END,
        cfg=reactor_config(),
        unlock=unlock,
        sink=sink,
        workers=2,
        expected=expected,
    )


async def test_an_explicit_set_accounts_for_every_id(engine: AsyncEngine) -> None:
    """R1's run: each headline without a return has its reason, simulated or not."""
    stories, expected, unlock = await _headlines(engine)
    sink = MemorySink()
    summary = await _reactor_run(engine, stories, expected, unlock, sink)
    assert summary.expected == 8 and summary.stories == 7 and summary.started == 6
    assert summary.dropped == {
        "h-bad": "bad_bars",
        "h-holiday": "no_session",
        "h-late": "entry_unfilled",
        "h-nospy": "no_spy",
        "h-thin": "spy_thin",
        "h-unbuilt": "no_story",
    }
    trades = {o.story_id: o.trade for o in sink.outcomes if o.trade is not None}
    assert set(trades) == {"h-ok", "h-jump", "h-nospy"}  # the study's filter is R1's, not run's
    nospy = trades["h-nospy"]
    assert nospy is not None and "no_spy" in nospy.flags and math.isnan(nospy.r_net_abn)
    assert sink.summary is not None and sink.summary["dropped"] == summary.dropped
    assert sink.summary["expected"] == 8
    assert summary.expected_sha == sim.id_set_sha(HEADS) == sink.summary["expected_sha"]


async def test_an_explicit_set_refuses_other_stories_before_writing(engine: AsyncEngine) -> None:
    stories, expected, unlock = await _headlines(engine)
    sink = MemorySink()
    stray = reactor_story("h-stray", "AAA", et(TUE, 10, 0))
    with pytest.raises(ValueError, match="h-stray is not in the expected set"):
        await _reactor_run(engine, [*stories, stray], expected, unlock, sink)
    moved = reactor_story("h-ok", "AAA", et(TUE, 10, 0))  # the set says MON
    with pytest.raises(ValueError, match="h-ok is AAA 2016-03-08; the expected set says"):
        await _reactor_run(engine, [moved, *stories[1:]], expected, unlock, sink)
    with pytest.raises(ValueError, match="h-ok is given twice"):
        await _reactor_run(engine, [*stories, stories[0]], expected, unlock, sink)
    assert sink.info is None and sink.outcomes == []


async def test_r1_in_miniature_compares_the_run_with_the_study(engine: AsyncEngine) -> None:
    """The study's drops (``intraday.entry_and_close`` on the stored rows) against the run's."""
    stories, expected, unlock = await _headlines(engine)
    await seed_bars(engine, "GGG", session_bars(MON))  # the study computes it: no story did
    sink = MemorySink()
    summary = await _reactor_run(engine, stories, expected, unlock, sink)
    lo, hi = intraday._PLAUSIBLE
    study_dropped = set()
    for hid, (symbol, published) in HEADS.items():
        day = published.astimezone(MARKET_TZ).date()
        at = published + intraday.LATENCY
        if not is_trading_day(day):
            study_dropped.add(hid)
            continue
        s = intraday.entry_and_close(await minutes.read(engine, symbol, day), at)
        q = intraday.entry_and_close(await minutes.read(engine, "SPY", day), at)
        if s is None or q is None or not lo < s[1] / s[0] < hi:
            study_dropped.add(hid)
    assert study_dropped == {"h-holiday", "h-late", "h-nospy", "h-jump"}
    trades = [o.trade for o in sink.outcomes if o.trade is not None]
    check = r1_dropped(HEADS, summary, trades, study_dropped)
    assert check.set_aside == {"h-bad": ("bad_bars",), "h-thin": ("spy_thin",)}
    assert check.sim_only == {"h-unbuilt": "no_story"} and check.study_only == ()
    assert check.kept == ("h-ok",) and check.compared == 6
    assert check.share == 0.25 and not check.within_cap and not check.passed
    with pytest.raises(ValueError, match="not H's 7"):
        r1_dropped([h for h in HEADS if h != "h-ok"], summary, trades, study_dropped)
    # The same set run again: its summary does not take the first run's trades.
    again = await _reactor_run(engine, stories, expected, unlock, MemorySink())
    with pytest.raises(ValueError, match=f"is from run {summary.run_id}, not {again.run_id}"):
        r1_dropped(HEADS, again, trades, study_dropped)
    with pytest.raises(ValueError, match="h-ok"):
        r1_dropped(HEADS, summary, [t for t in trades if t.story_id != "h-ok"], study_dropped)


async def test_a_run_logs_its_start_batches_and_end(
    engine: AsyncEngine, caplog: pytest.LogCaptureFixture
) -> None:
    """The structured events come from core/events.py, under one run_id."""
    market = random_market(25, 6, sessions=1)
    await seed_calendar(engine, date(2016, 1, 4), END)
    await seed_market(engine, market)
    with caplog.at_level(logging.INFO, logger=sim.__name__):
        sink, summary = await _go(engine, market, batch=2)
    records = [r for r in caplog.records if str(getattr(r, "event", "")).startswith("playbooks.")]
    names = [r.event for r in records]  # type: ignore[attr-defined]
    assert names[0] == events.SIM_RUN_START and names[-1] == events.SIM_RUN_DONE
    assert names.count(events.SIM_RUN_BATCH) == len(names) - 2 >= 2
    assert {r.run_id for r in records} == {summary.run_id}  # type: ignore[attr-defined]
    assert records[-1].trades == summary.trades  # type: ignore[attr-defined]


async def test_a_run_lists_the_stories_its_data_rules_touched(engine: AsyncEngine) -> None:
    """Skips by reason; paths whose own bars, SPY's or the spare session's were cut, apart."""
    await seed_calendar(engine, date(2016, 1, 4), END)
    insane = (10.0, 9.0, 8.0, 9.5, 1.0, 9.5)  # h < max(o, c)
    await seed_bars(engine, "SPY", session_bars(MON, price=200.0))
    await seed_bars(engine, "SPY", session_bars(TUE, price=200.0, rows={(11, 0): insane}))
    await seed_bars(engine, "OK", session_bars(MON))
    await seed_bars(engine, "BAD", session_bars(MON, rows={(9, 30 + i): insane for i in range(6)}))
    await seed_bars(engine, "TWO", session_bars(MON, rows={(9, 31): insane, (9, 32): insane}))
    await seed_bars(engine, "SPYCUT", session_bars(TUE))  # its own bars clean, SPY's cut
    await seed_bars(engine, "DEFECT", session_bars(MON))
    await seed_bars(engine, "SPARE", session_bars(MON))
    await seed_bars(engine, "SPARE", session_bars(TUE, rows={(9, 40): insane}))  # its spare
    days = [(s, MON) for s in ("OK", "BAD", "TWO", "DEFECT", "SPARE")] + [("SPYCUT", TUE)]
    await mark_done(engine, [*days, ("SPARE", TUE), ("SPY", MON), ("SPY", TUE)])
    # DEFECT's adjusted series halves on 2016-02-01 while raw / adjusted holds still (stale).
    closes = {
        d: (100.0 if d < date(2016, 2, 1) else 50.0)
        for d in (date(2016, 1, 4) + timedelta(days=i) for i in range(70))
        if is_trading_day(d) and d <= MON
    }
    await seed_daily(engine, "DEFECT", closes, adjusted=closes)
    stories = [story(s, d, downgrade(d)) for s, d in days]
    sink = MemorySink()
    summary = await run(
        engine,
        stories,
        ToyFactory(sessions=1, target=0.005),
        context=Context(),
        window=Window.GATE,
        window_end=END,
        cfg=SimConfig(),
        unlock=WindowUnlock(),
        sink=sink,
        workers=2,
    )
    assert summary.skip_ids == {
        "bad_bars": ("BAD:2016-03-07",),
        "adjust_defect": ("DEFECT:2016-03-07",),
    }
    assert summary.bar_drop_ids == ("TWO:2016-03-07",)  # the stock's own bars
    assert summary.spy_drop_ids == ("SPYCUT:2016-03-08",)  # SPY's, on a path session
    assert summary.spy_drop_days == {"2016-03-08": 1}
    assert summary.spare_drop_ids == ("SPARE:2016-03-07",)  # the spare (TUE), not the path
    assert summary.skips == {"bad_bars": 1, "adjust_defect": 1}
    assert summary.loader["dropped_bars"] == 2 and summary.loader["spy_dropped_bars"] == 1
    assert summary.loader["spare_dropped_bars"] == 1
    assert summary.expected == 0 and summary.dropped == {}  # no explicit set
    assert sink.summary is not None
    assert sink.summary["skip_ids"] == {
        "adjust_defect": ["DEFECT:2016-03-07"],
        "bad_bars": ["BAD:2016-03-07"],
    }
    assert sink.summary["bar_drop_ids"] == ["TWO:2016-03-07"]
    assert sink.summary["spy_drop_ids"] == ["SPYCUT:2016-03-08"]
    assert sink.summary["spy_drop_days"] == {"2016-03-08": 1}
    assert sink.summary["spare_drop_ids"] == ["SPARE:2016-03-07"]
    assert {o.story_id for o in sink.outcomes if o.skip is None} == {
        "OK:2016-03-07",
        "TWO:2016-03-07",
        "SPYCUT:2016-03-08",
        "SPARE:2016-03-07",
    }


def _fail_to_load() -> Context:
    raise RuntimeError("this context does not load in a worker")


class _Unloadable(Context):
    """Pickles here; unpickling it, as a spawn worker does, raises."""

    def __reduce__(self) -> tuple[object, tuple[()]]:
        return (_fail_to_load, ())


async def test_a_spawn_run_refuses_a_state_it_cannot_ship_before_writing(
    halabot_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A factory or context a worker cannot get fails the run with no run row.

    Its parts are pickled one by one only after the pool failed to start: a run
    whose state ships pays for no extra pickle.
    """
    engine = halabot_engine
    market = random_market(26, 4, sessions=1)
    await seed_calendar(engine, date(2016, 1, 4), END)
    await seed_market(engine, market)
    toy = ToyFactory(sessions=1, target=0.005)

    async def go(factory: PlaybookFactory, context: Context) -> sim.RunSummary:
        return await run(
            engine,
            market.stories,
            factory,
            context=context,
            window=Window.GATE,
            window_end=END,
            cfg=SimConfig(),
            unlock=WindowUnlock(),
            sink=PgOutcomeSink(engine),
            workers=2,
            parallel="spawn",
        )

    closure = Factory(lambda s: Toy(s, target=0.005), "toy", "1", 1)
    with pytest.raises(ValueError, match=r"and make_playbook cannot be pickled"):
        await go(closure, market.ctx)
    locked = Context()
    locked.lock = threading.Lock()  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match=r"and context cannot be pickled"):
        await go(toy, locked)
    with pytest.raises(BrokenProcessPool):  # it pickles, but no worker can load it
        await go(toy, _Unloadable())
    async with engine.connect() as conn:
        assert await conn.scalar(text("SELECT count(*) FROM hb_playbook_run")) == 0

    def no_diagnosis(shared: object) -> list[str]:
        raise AssertionError("a state that ships is never pickled part by part")

    real_dumps = ForkingPickler.dumps
    whole: list[object] = []

    def dumps(cls: type, obj: object, protocol: int | None = None) -> object:
        if isinstance(obj, sim._Shared):
            whole.append(obj)  # a check pickling the run state before the pool does
        return real_dumps(obj, protocol)

    monkeypatch.setattr(sim, "_unpicklable", no_diagnosis)
    monkeypatch.setattr(ForkingPickler, "dumps", classmethod(dumps))
    summary = await go(toy, market.ctx)  # the same run with a state that ships
    assert whole == []
    assert summary.outcomes > 0
    async with engine.connect() as conn:
        assert await conn.scalar(text("SELECT count(*) FROM hb_playbook_run")) == 1


def test_pool_method_defaults_to_serial_on_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    assert pool_method(None, 1) is None and pool_method(True, 1) is None
    assert pool_method(False, 6) is None
    monkeypatch.setattr(sys, "platform", "darwin")
    assert pool_method(None, 6) is None  # no fork pool where the fleet runs
    assert pool_method(True, 6) == "spawn"
    assert pool_method("spawn", 6) == "spawn"
    with pytest.raises(ValueError, match="unsafe on macOS"):
        pool_method("fork", 6)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sim.multiprocessing, "get_all_start_methods", lambda: ["fork", "spawn"])
    assert pool_method(None, 6) == "fork" and pool_method(True, 6) == "fork"
    assert pool_method("spawn", 6) == "spawn"
    monkeypatch.setattr(sim.multiprocessing, "get_all_start_methods", lambda: ["spawn"])
    assert pool_method(None, 6) is None and pool_method(True, 6) == "spawn"
    with pytest.raises(ValueError, match="not available"):
        pool_method("fork", 6)


def test_the_parent_never_holds_a_run_state() -> None:
    """Only pool workers set the per-process state, by their initializer."""
    assert sim._WORKER_STATE is None
    assert not hasattr(sim, "_SHARED")
