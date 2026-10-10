"""The driver: checks before any write, no state shared between runs, how workers run."""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from concurrent.futures.process import BrokenProcessPool
from datetime import date, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.playbooks import sim
from halabot.playbooks.legacy import HoldFactory, reactor_config, reactor_story
from halabot.playbooks.loader import (
    Window,
    WindowLocked,
    WindowUnlock,
    register_gate_units,
    unit_set_sha,
)
from halabot.playbooks.playbook import Factory, PlaybookFactory
from halabot.playbooks.records import MemorySink, PgOutcomeSink, outcomes_sha256
from halabot.playbooks.sim import DATA_SKIPS, pool_method, run
from halabot.playbooks.types import SimConfig
from halal_trader.core import events
from halal_trader.market_hours import is_trading_day
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


async def test_a_run_lists_the_stories_its_data_rules_set_apart(engine: AsyncEngine) -> None:
    """bad_bars and adjust_defect skips, and paths whose bars (or SPY's) the sanity rule cut."""
    await seed_calendar(engine, date(2016, 1, 4), END)
    insane = (10.0, 9.0, 8.0, 9.5, 1.0, 9.5)  # h < max(o, c)
    await seed_bars(engine, "SPY", session_bars(MON, price=200.0))
    await seed_bars(engine, "SPY", session_bars(TUE, price=200.0, rows={(11, 0): insane}))
    await seed_bars(engine, "OK", session_bars(MON))
    await seed_bars(engine, "BAD", session_bars(MON, rows={(9, 30 + i): insane for i in range(6)}))
    await seed_bars(engine, "TWO", session_bars(MON, rows={(9, 31): insane, (9, 32): insane}))
    await seed_bars(engine, "SPYCUT", session_bars(TUE))  # its own bars clean, SPY's cut
    await seed_bars(engine, "DEFECT", session_bars(MON))
    days = [(s, MON) for s in ("OK", "BAD", "TWO", "DEFECT")] + [("SPYCUT", TUE)]
    await mark_done(engine, [*days, ("SPY", MON), ("SPY", TUE)])
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
    assert summary.bar_drop_ids == ("SPYCUT:2016-03-08", "TWO:2016-03-07")
    assert summary.data_filtered() == {
        "BAD:2016-03-07",
        "DEFECT:2016-03-07",
        "SPYCUT:2016-03-08",
        "TWO:2016-03-07",
    }
    assert summary.skips == {"bad_bars": 1, "adjust_defect": 1}
    assert summary.loader["dropped_bars"] == 2 and summary.loader["spy_dropped_bars"] == 1
    assert sink.summary is not None
    assert sink.summary["skip_ids"] == {
        "adjust_defect": ["DEFECT:2016-03-07"],
        "bad_bars": ["BAD:2016-03-07"],
    }
    assert sink.summary["bar_drop_ids"] == ["SPYCUT:2016-03-08", "TWO:2016-03-07"]
    assert {o.story_id for o in sink.outcomes if o.skip is None} == {
        "OK:2016-03-07",
        "TWO:2016-03-07",
        "SPYCUT:2016-03-08",
    }
    assert DATA_SKIPS == ("adjust_defect", "bad_bars")


def _fail_to_load() -> Context:
    raise RuntimeError("this context does not load in a worker")


class _Unloadable(Context):
    """Pickles here; unpickling it, as a spawn worker does, raises."""

    def __reduce__(self) -> tuple[object, tuple[()]]:
        return (_fail_to_load, ())


async def test_a_spawn_run_refuses_a_state_it_cannot_ship_before_writing(
    halabot_engine: AsyncEngine,
) -> None:
    """A factory or context a worker cannot get fails the run with no run row."""
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
    summary = await go(toy, market.ctx)  # the same run with a state that ships
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
