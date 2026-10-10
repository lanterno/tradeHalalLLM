"""The driver: checks before any write, no state shared between runs, how workers run."""

from __future__ import annotations

import asyncio
import sys
from datetime import date

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
from halabot.playbooks.records import MemorySink, PgOutcomeSink, outcomes_sha256
from halabot.playbooks.sim import pool_method, run
from halabot.playbooks.types import SimConfig
from tests.halabot.playbooks._seed import mark_done, seed_bars, seed_calendar, seed_market
from tests.halabot.playbooks._support import (
    FACTS,
    MON,
    Context,
    ToyFactory,
    et,
    session_bars,
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
