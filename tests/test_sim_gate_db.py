"""The Phase 0 gates on a database (events/sim_gate.py): rows, pins, refusals, the runs.

Three small synthetic worlds, every price made up:

* **G1**: Acme (AAA) and Bolt (BBB) downgraded before the open in 2016; the
  stories are built from stored events (``events stories build``), the real
  ``units.g1_stories`` selects them, and the paths load behind the gate's pin.
* **G2**: twelve headlines on three sessions of March 2026 (the selection is
  monkeypatched, its count too); minute bars from random walks.
* **G3**: SUE observations on four names in 2016 (the observations and the
  calibration pairs are monkeypatched; Σ_c and Σ_s are the units module's);
  minute bars agree with the daily bars, so minute mode equals daily mode.

Each gate refuses (no row) while one of its units is not done.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.playbooks.bounce import OverreactionBounce
from halabot.playbooks.loader import GATE_UNITS_NAME, WindowLocked, gate_pin, gate_pins
from halabot.playbooks.types import Submit
from halal_trader.data import minutes
from halal_trader.data.minutes import BarArrays
from halal_trader.db.repos.quant_trials import QuantTrialRepoImpl, config_hash
from halal_trader.events import h1, intraday, sim_gate, units
from halal_trader.events.intraday import Headline
from halal_trader.events.sim_gate import (
    CRITERIA,
    GATE_NAME,
    G1Factory,
    GateRefused,
    GateResult,
    GateRun,
    LookaheadWorld,
    pin_units,
    read_gate_bars,
    record_gate,
    require_done,
    run_determinism,
    run_gates,
    run_lookahead,
    run_reactor,
    run_sue,
)
from halal_trader.events.stories import build_range
from halal_trader.events.study import Observation
from halal_trader.market_hours import MARKET_TZ, is_trading_day, next_trading_day
from tests._stories import add_aliases, mark_built, news_row, store
from tests.halabot.playbooks._seed import mark_done, seed_bars, seed_calendar, seed_daily
from tests.halabot.playbooks._support import epoch, session_bars
from tests.halabot.playbooks._synth import random_session


def ny(day: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm), MARKET_TZ).astimezone(UTC)


def sessions(lo: date, hi: date) -> list[date]:
    out, d = [], lo
    while d <= hi:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


async def gate_rows(engine: AsyncEngine) -> list[Any]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                'SELECT id, config, config_hash, metrics, verdict, criterion, "window" '
                "FROM quant_trials WHERE name = :n AND kind = 'gate' ORDER BY id"
            ),
            {"n": GATE_NAME},
        )
        return list(rows.all())


async def daily(engine: AsyncEngine, symbol: str, prices: dict[date, float]) -> None:
    """Raw and all-adjusted daily bars at the same prices (A = 1)."""
    await seed_daily(engine, symbol, prices, adjusted=prices)


async def liquidity(engine: AsyncEngine, ranked: Sequence[str], first: date, last: date) -> None:
    rows = []
    m = first.replace(day=1)
    while m <= last:
        rows += [{"s": s, "m": m, "v": 1e9 / (i + 1)} for i, s in enumerate(ranked)]
        m = date(m.year + m.month // 12, m.month % 12 + 1, 1)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO monthly_bars (symbol, month, close, volume, vwap) "
                "VALUES (:s, :m, 50, :v, 50)"
            ),
            rows,
        )


async def undone(engine: AsyncEngine, unit: tuple[str, date]) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM backfill_progress WHERE task = :t AND unit = :u"),
            {"t": minutes.TASK, "u": minutes.unit(*unit)},
        )


# ── the ledger, pins and reads ──


async def test_a_gate_row_carries_the_constants_the_rule_and_the_numbers(
    engine: AsyncEngine,
) -> None:
    row_id = await record_gate(
        engine, GateResult("r1", True, {"x": math.nan, "d": date(2026, 3, 2)}, {"units_sha": "ab"})
    )
    (row,) = await gate_rows(engine)
    assert row.id == row_id and row.verdict == "pass" and row.criterion == CRITERIA["r1"]
    assert row.config == {**sim_gate.gate_config("r1"), "units_sha": "ab"}
    assert row.config_hash == config_hash(row.config)
    assert row.metrics == {"x": None, "d": "2026-03-02"}
    assert await h1.gate_rows(engine) == [("r1", row_id, "pass")]  # what the H1 runner reads
    await record_gate(engine, GateResult("r1", False, {}))
    assert [r.verdict for r in await gate_rows(engine)] == ["pass", "fail"]


CALIB_DAY = date(2016, 2, 8)


async def test_units_are_pinned_checked_done_and_read_behind_the_guard(
    engine: AsyncEngine,
) -> None:
    day, later = CALIB_DAY, date(2016, 2, 12)
    unit_set = frozenset({("AAA", day), ("AAA", later)})
    await seed_calendar(engine, date(2016, 1, 4), date(2016, 9, 30))
    bad = session_bars(day, price=50.0, rows={(10, 0): (50.0, 49.0, 48.0, 50.0, 1e3, 50.0)})
    await seed_bars(engine, "AAA", bad)  # one bar with h < max(o, c): cut by the sanity rule
    await seed_bars(engine, "AAA", session_bars(later, price=51.0))
    for d in (day, later):
        await seed_bars(engine, "SPY", session_bars(d, price=200.0))
    await mark_done(engine, sorted(unit_set))
    with pytest.raises(GateRefused, match="2 of 4 units are not done"):
        await require_done(engine, ["s1-calib"], unit_set)
    await mark_done(engine, [("SPY", day), ("SPY", later)])
    await require_done(engine, ["s1-calib"], unit_set)

    plan = units.UnitPlan({"gate_calib": unit_set})  # plan H's gate_calib
    planned = plan.sha("gate_calib")
    with pytest.raises(WindowLocked, match="not pinned"):
        await read_gate_bars(engine, "calib", unit_set, planned, date(2016, 9, 30))
    with pytest.raises(WindowLocked, match="does not match"):
        await read_gate_bars(engine, "calib", unit_set, "0" * 64, date(2016, 9, 30))
    # A set that is not plan H's is refused, naming both hashes, and pins nothing.
    stray = {("AAA", day)}
    stray_sha = units.UnitPlan({"gate_calib": frozenset(stray)}).sha("gate_calib")
    with pytest.raises(GateRefused, match=f"hashes to {stray_sha}, but plan H's to {planned}"):
        await pin_units(engine, "calib", "gate_calib", stray, plan)
    assert await gate_pins(engine, "calib") == set()
    sha = await pin_units(engine, "calib", "gate_calib", unit_set, plan)
    assert sha == planned
    assert await pin_units(engine, "calib", "gate_calib", unit_set, plan) == sha  # a no-op
    assert await gate_pins(engine, "calib") == {sha}
    moved = units.UnitPlan({"gate_calib": frozenset({("BBB", day)})})  # plan H changed since
    other = moved.sha("gate_calib")
    with pytest.raises(GateRefused, match=f"pinned to {sha}, but .* hashes to {other}"):
        await pin_units(engine, "calib", "gate_calib", {("BBB", day)}, moved)
    assert await gate_pins(engine, "calib") == {sha}
    bars, cut = await read_gate_bars(engine, "calib", unit_set, sha, date(2016, 9, 30))
    assert cut == 1 and len(bars[("AAA", day)]) == 389 and len(bars[("SPY", later)]) == 390
    async with engine.connect() as conn:
        pins = await conn.scalar(
            text("SELECT count(*) FROM quant_trials WHERE name = :n"), {"n": GATE_UNITS_NAME}
        )
    assert pins == 1 and await gate_rows(engine) == []  # a pin is not a gate row


async def test_a_changed_selection_repins_only_with_a_reason(engine: AsyncEngine) -> None:
    """After an upstream data fix the gate's set changed: a reason re-pins it, logged."""
    day = CALIB_DAY
    old = frozenset({("AAA", day)})
    new = frozenset({("AAA", day), ("BBB", day)})
    before, after = units.UnitPlan({"gate_calib": old}), units.UnitPlan({"gate_calib": new})
    sha = await pin_units(engine, "calib", "gate_calib", old, before)
    other = after.sha("gate_calib")
    with pytest.raises(GateRefused, match="--repin <reason>"):
        await pin_units(engine, "calib", "gate_calib", new, after)
    assert await gate_pin(engine, "calib") == sha
    assert await pin_units(engine, "calib", "gate_calib", new, after, repin="aliases") == other
    assert await gate_pin(engine, "calib") == other and await gate_pins(engine, "calib") == {
        sha,
        other,
    }
    async with engine.connect() as conn:
        config = await conn.scalar(
            text("SELECT config FROM quant_trials WHERE name = :n ORDER BY id DESC LIMIT 1"),
            {"n": GATE_UNITS_NAME},
        )
    assert config == {"gate": "calib", "units_sha": other, "supersedes": sha, "reason": "aliases"}
    # The new set is the pin: the same set again is a no-op, the old one is refused.
    assert await pin_units(engine, "calib", "gate_calib", new, after) == other
    with pytest.raises(GateRefused, match=f"pinned to {other}, but .* hashes to {sha}"):
        await pin_units(engine, "calib", "gate_calib", old, before)
    with pytest.raises(WindowLocked, match="superseded"):
        await read_gate_bars(engine, "calib", old, sha, date(2016, 9, 30))


async def test_conflicting_pins_refuse_the_gate_until_a_repin(engine: AsyncEngine) -> None:
    """Two pins nobody sanctioned (a race) refuse the run, never crash it."""
    a, b = frozenset({("AAA", CALIB_DAY)}), frozenset({("BBB", CALIB_DAY)})
    repo = QuantTrialRepoImpl(engine)
    for unit_set in (a, b):
        sha = units.UnitPlan({"gate_calib": unit_set}).sha("gate_calib")
        await repo.record_trial(
            name=GATE_UNITS_NAME, kind="gate-units", config={"gate": "calib", "units_sha": sha}
        )
    plan = units.UnitPlan({"gate_calib": b})
    with pytest.raises(GateRefused, match="without superseding"):
        await pin_units(engine, "calib", "gate_calib", b, plan)
    sha_b = await pin_units(engine, "calib", "gate_calib", b, plan, repin="settle the race")
    assert await gate_pin(engine, "calib") == sha_b


async def test_a_repin_needs_a_reason_before_any_gate_runs(engine: AsyncEngine) -> None:
    for blank in ("", "  "):
        with pytest.raises(ValueError, match="a re-pin needs a reason"):
            await run_gates(engine, "all", repin=blank)
    assert await gate_rows(engine) == []


# ── G2: the reactor ──

R_DAYS = (date(2026, 3, 2), date(2026, 3, 3), date(2026, 3, 4))
R_SYMBOLS = ("AAA", "BBB", "CCC", "DDD")
R_SCORES = (0.9, 0.5, 0.1, -0.6)


def reactor_headlines() -> list[Headline]:
    out = [
        Headline(sym, ny(day, 10 + j, 7 * i + 3), score)
        for i, day in enumerate(R_DAYS)
        for j, (sym, score) in enumerate(zip(R_SYMBOLS, R_SCORES))
    ]
    out.append(Headline("EEE", ny(date(2026, 4, 3), 11), 0.8))  # Good Friday: no session
    return out


@pytest.fixture
async def reactor_world(engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch) -> AsyncEngine:
    headlines = reactor_headlines()

    async def chosen(engine: AsyncEngine) -> list[Headline]:
        return list(headlines)

    async def first_in_session(engine: AsyncEngine, **_: Any) -> list[Headline]:
        return list(headlines) * 3

    monkeypatch.setattr(units, "reactor_headlines", chosen)
    monkeypatch.setattr(intraday, "first_in_session", first_in_session)
    await seed_calendar(engine, date(2016, 1, 4), date(2026, 10, 9))
    days = sessions(date(2025, 1, 2), date(2026, 4, 30))
    await seed_daily(engine, "SPY", {}, adjusted=dict.fromkeys(days, 200.0))
    for k, sym in enumerate(R_SYMBOLS):
        await daily(engine, sym, {d: 40.0 + 10 * k + (i % 3) for i, d in enumerate(days)})
    rng = np.random.default_rng(42)
    done: set[tuple[str, date]] = set()
    for day in R_DAYS:
        await seed_bars(engine, "SPY", random_session(rng, day, 200.0, holes=False))
        done.add(("SPY", day))
        for k, sym in enumerate(R_SYMBOLS):
            await seed_bars(engine, sym, random_session(rng, day, 40.0 + 10 * k))
            done.add((sym, day))
    await mark_done(engine, sorted(done))
    return engine


async def test_the_reactor_gates_run_on_the_pinned_set_and_write_three_rows(
    reactor_world: AsyncEngine,
) -> None:
    engine = reactor_world
    out = await run_reactor(engine)
    assert not out.refused
    r0, r1, r2 = out.results
    assert [r.gate for r in out.results] == ["r0", "r1", "r2"]
    # R0 runs the legacy study with a market that cannot fetch: the numbers are this
    # world's, not the recorded ones, so it fails (and says what it found).
    checks = r0.metrics["checks"]
    assert not r0.passed and checks["headlines"]["got"] == 39 and checks["strong_n"]["got"] == 6
    assert checks["strong_headlines"] == {"got": 7, "want": 7_931, "ok": False}
    # R1: the simulator reproduces every headline the study kept, and drops the same.
    assert r1.passed, r1.metrics
    assert r1.metrics["compared"] == 12 and r1.metrics["max_abs_diff"] <= 1e-10
    assert r1.metrics["dropped"]["sim_only"] == {} and r1.metrics["dropped"]["study_only"] == []
    assert r1.metrics["run"]["expected"] == 13
    # R2: realistic fills, inside the legacy interval, with each headline's decomposition.
    assert r2.passed, r2.metrics
    assert r2.metrics["realistic"]["n"] == 6 and r2.metrics["legacy"]["clusters"] == 3
    parts = r2.metrics["decomposition"]["per_headline"]
    assert len(parts) == 6 and all(len(v) == 2 for v in parts.values())
    rows = await gate_rows(engine)
    assert [r.config["gate"] for r in rows] == ["r0", "r1", "r2"]
    assert [r.verdict for r in rows] == ["fail", "pass", "pass"]
    sha = rows[0].config["units_sha"]
    assert all(r.config["units_sha"] == sha for r in rows) and await gate_pins(
        engine, "reactor"
    ) == {sha}
    assert sha == (await units.h1_plan(engine, parts=["gate_reactor"])).sha("gate_reactor")
    assert rows[1].config["run_sim"]["fill"] == "legacy-reactor"
    assert rows[2].config["run_sim"]["fill"] == "gate-market"
    assert rows[0].window == "2026-03-02..2026-03-04"
    # Idempotent: a rerun writes the same numbers again.
    again = await run_reactor(engine)
    assert [r.metrics for r in again.results] == [r.metrics for r in out.results]
    assert len(await gate_rows(engine)) == 6


async def test_the_reactor_gates_refuse_while_a_unit_is_not_done(
    reactor_world: AsyncEngine,
) -> None:
    await undone(reactor_world, ("SPY", R_DAYS[1]))
    out = await run_reactor(reactor_world)
    assert out.results == [] and set(out.refused) == {"r0", "r1", "r2"}
    assert "1 of 15 units are not done" in out.refused["r1"]
    assert await gate_rows(reactor_world) == []
    assert await gate_pins(reactor_world, "reactor") == set()  # a refused run pins nothing


async def test_the_reactor_gates_refuse_a_set_other_than_the_pinned_one(
    reactor_world: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The selection changes after the pin: a refusal naming both hashes, not a crash."""
    first = await run_reactor(reactor_world, write=False)
    assert not first.refused
    (pinned,) = await gate_pins(reactor_world, "reactor")
    fewer = [h for h in reactor_headlines() if h.symbol != "DDD"]

    async def chosen(engine: AsyncEngine) -> list[Headline]:
        return list(fewer)

    monkeypatch.setattr(units, "reactor_headlines", chosen)
    changed = units.UnitPlan({"gate_reactor": units.reactor_units(fewer)}).sha("gate_reactor")
    out = await run_reactor(reactor_world)
    assert out.results == [] and set(out.refused) == {"r0", "r1", "r2"}
    both = f"pinned to {pinned}, but this run's gate_reactor set hashes to {changed}"
    assert both in out.refused["r1"]
    assert await gate_pins(reactor_world, "reactor") == {pinned}
    assert await gate_rows(reactor_world) == []
    assert "--repin <reason>" in out.refused["r1"]


async def test_the_reactor_gates_repin_a_changed_set_given_a_reason(
    reactor_world: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The selection changed after a data fix: the reason re-pins it and every gate reruns."""
    first = await run_reactor(reactor_world)
    assert not first.refused
    pinned = await gate_pin(reactor_world, "reactor")
    old_units = units.reactor_units(reactor_headlines())
    fewer = [h for h in reactor_headlines() if h.symbol != "DDD"]

    async def chosen(engine: AsyncEngine) -> list[Headline]:
        return list(fewer)

    monkeypatch.setattr(units, "reactor_headlines", chosen)
    changed = units.UnitPlan({"gate_reactor": units.reactor_units(fewer)}).sha("gate_reactor")
    out = await run_gates(reactor_world, "reactor", repin="aliases rebuilt (2026-10-11)")
    assert not out.refused and [r.gate for r in out.results] == ["r0", "r1", "r2"]
    assert all(r.config["units_sha"] == changed for r in out.results)
    rows = await gate_rows(reactor_world)
    assert [r.config["units_sha"] for r in rows] == [pinned] * 3 + [changed] * 3
    assert await gate_pin(reactor_world, "reactor") == changed
    assert await gate_pins(reactor_world, "reactor") == {pinned, changed}
    with pytest.raises(WindowLocked, match="superseded"):
        await read_gate_bars(reactor_world, "reactor", old_units, str(pinned), date(2026, 10, 9))
    # The same set with the reason again: no new pin, the gates rerun on it.
    again = await run_reactor(reactor_world, repin="aliases rebuilt (2026-10-11)")
    assert not again.refused
    async with reactor_world.connect() as conn:
        pins = await conn.scalar(
            text("SELECT count(*) FROM quant_trials WHERE name = :n"), {"n": GATE_UNITS_NAME}
        )
    assert pins == 2


async def test_the_reactor_gates_refuse_a_set_plan_h_did_not_select(
    reactor_world: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate's headlines against plan H's: another set is refused before any pin."""
    real = units.h1_plan
    elsewhere = frozenset({("AAA", R_DAYS[0])})

    async def h1_plan(engine: AsyncEngine, **kw: Any) -> units.UnitPlan:
        plan = await real(engine, **kw)
        return units.UnitPlan({**plan.parts, "gate_reactor": elsewhere})

    monkeypatch.setattr(units, "h1_plan", h1_plan)
    gate = units.UnitPlan({"gate_reactor": units.reactor_units(reactor_headlines())})
    planned = units.UnitPlan({"gate_reactor": elsewhere})
    out = await run_reactor(reactor_world)
    assert out.results == [] and set(out.refused) == {"r0", "r1", "r2"}
    want = f"hashes to {gate.sha('gate_reactor')}, but plan H's to {planned.sha('gate_reactor')}"
    assert want in out.refused["r0"]
    assert await gate_pins(reactor_world, "reactor") == set()


async def test_r0_fails_when_the_study_would_have_to_fetch(
    reactor_world: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with reactor_world.begin() as conn:
        await conn.execute(text("DELETE FROM minute_bars WHERE symbol = 'CCC'"))
        await conn.execute(
            text("DELETE FROM backfill_progress WHERE unit LIKE 'CCC:%' AND task = :t"),
            {"t": minutes.TASK},
        )

    async def nothing(*_: Any, **__: Any) -> None:
        return None

    # With the done check bypassed, the legacy study asks the market for CCC's bars.
    monkeypatch.setattr(sim_gate, "require_done", nothing)
    out = await run_reactor(reactor_world, write=False)
    r0 = out.results[0]
    assert r0.gate == "r0" and not r0.passed and "asked the market for CCC" in r0.metrics["error"]


# ── G3: SUE ──

S_SYMBOLS = ("AAA", "BBB", "CCC", "DDD")
S_RANGE = (date(2015, 1, 2), date(2016, 12, 30))
S_DAYS = sessions(*S_RANGE)


def s_price(k: int, i: int, *, close: bool) -> float:
    base = 30.0 + 10 * k + 3 * math.sin(i / 7 + k)
    return base * (1.004 if close else 1.0)


def s_observations() -> list[Observation]:
    rng = np.random.default_rng(7)
    obs = []
    first = sessions(date(2016, 2, 1), date(2016, 5, 31))
    for n in range(28):
        day = first[int(rng.integers(0, len(first)))]
        at = ny(day, 8) if n % 3 else ny(day, 11, 30)  # an open entry, else a close entry
        obs.append(Observation(S_SYMBOLS[n % 4], at, float(rng.normal())))
    return sorted(obs, key=lambda o: o.published_at)


CALIB_PAIRS = [
    (S_SYMBOLS[i % 4], d) for i, d in enumerate(sessions(date(2016, 2, 1), date(2016, 2, 19)))
]


async def sigma_c(engine: AsyncEngine, obs: Sequence[Observation]) -> list[units.SueEvent]:
    """Σ_c as the gate builds it: less the events meeting H1's train and validation units."""
    w = await units.h1_windows(engine)
    return await units.sue_complement(
        engine, h1_units=w["train"] | w["validation"], observations=obs
    )


def minute_day(day: date, open_px: float, close_px: float) -> BarArrays:
    """Bars at 09:30 (the daily open) and every hour, 15:56 and 15:59 at the daily close."""
    keep = [(9, 30), (10, 30), (11, 45), (12, 30), (13, 30), (14, 30), (15, 56), (15, 59)]
    ts = np.asarray([epoch(day, h, m) for h, m in keep], dtype=np.int64)
    p = np.asarray([open_px] + [close_px] * (len(keep) - 1))
    return BarArrays(ts, p, p.copy(), p.copy(), p.copy(), np.full(len(keep), 1e3), p.copy())


@pytest.fixture
async def sue_world(engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch) -> AsyncEngine:
    obs = s_observations()

    async def load(engine: AsyncEngine) -> list[Observation]:
        return list(obs)

    async def pairs(engine: AsyncEngine, **_: Any) -> list[tuple[str, date]]:
        return list(CALIB_PAIRS)

    monkeypatch.setattr(units, "load_sue_observations", load)
    monkeypatch.setattr(units, "calib_pairs", pairs)
    spy = {d: 200.0 + (i % 2) for i, d in enumerate(S_DAYS)}
    await daily(engine, "SPY", spy)
    await seed_calendar(engine, date(2016, 12, 31), date(2020, 1, 30))
    for k, sym in enumerate(S_SYMBOLS):
        rows = [
            {
                "s": sym,
                "d": d,
                "a": a,
                "o": s_price(k, i, close=False),
                "c": s_price(k, i, close=True),
            }
            for i, d in enumerate(S_DAYS)
            for a in ("raw", "all")
        ]
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, "
                    "volume, fetched_at) VALUES (:s, :d, :a, :o, :c, :o, :c, 1e6, now())"
                ),
                rows,
            )
    await liquidity(engine, S_SYMBOLS, date(2015, 1, 1), date(2016, 12, 1))
    await mark_built(engine, units.TRAIN[0], units.VALIDATION[1])
    # Minute bars agreeing with the daily bars on every unit the gates read.
    complement = await sigma_c(engine, obs)
    assert len(complement) == len(obs)  # no story, so no H1 unit: every event is in Σ_c
    wanted = units.sue_units(units.sue_sample(complement)) | units.calib_units(CALIB_PAIRS)
    index = {d: i for i, d in enumerate(S_DAYS)}
    done: set[tuple[str, date]] = set()
    for sym, d in sorted(wanted):
        k, i = S_SYMBOLS.index(sym), index[d]
        await seed_bars(
            engine, sym, minute_day(d, s_price(k, i, close=False), s_price(k, i, close=True))
        )
        done.add((sym, d))
        if ("SPY", d) not in done:
            await seed_bars(engine, "SPY", minute_day(d, spy[d], spy[d]))
            done.add(("SPY", d))
    await mark_done(engine, sorted(done))
    return engine


async def test_the_sue_gates_run_and_minute_mode_matches_daily_mode(sue_world: AsyncEngine) -> None:
    engine = sue_world
    out = await run_sue(engine, b=200)
    assert not out.refused
    by_id = {r.gate: r for r in out.results}
    assert list(by_id) == ["s0", "s1", "s1-calib", "s2", "s3"]
    s0 = by_id["s0"]
    # Four names ranked 0..3: one bucket, too few events for decile rows, so S0 records
    # the table it has and fails on the missing mid-cap row.
    assert not s0.passed and set(s0.metrics["table"]) == {"large (<300)"}
    assert s0.metrics["sigma_c"]["events"] == 28 and set(s0.metrics["sigma_c_daily"]) == {"5", "20"}
    s1 = by_id["s1"]
    assert s1.passed, s1.metrics
    assert s1.metrics["outcomes_study"] == s1.metrics["outcomes_sim"] > 28 * 3
    assert s1.metrics["max_abs_diff"] <= 1e-10
    cal = by_id["s1-calib"]
    assert cal.passed and cal.metrics["n"] == len(CALIB_PAIRS) and cal.metrics["p99_cal"] == 0.0
    s2 = by_id["s2"]
    assert s2.passed, s2.metrics
    assert (
        s2.metrics["horizons"]["5"]["median_abs_d"] == 0.0
        and s2.metrics["horizons"]["5"]["n"] == 28
    )
    s3 = by_id["s3"]
    assert s3.passed, s3.metrics
    five = s3.metrics["horizons"]["5"]
    assert five["realistic"]["d10_d1"] == pytest.approx(five["daily"]["d10_d1"])
    rows = await gate_rows(engine)
    assert [r.config["gate"] for r in rows] == ["s0", "s1", "s1-calib", "s2", "s3"]
    assert rows[2].config["units_sha"] == next(iter(await gate_pins(engine, "calib")))
    assert rows[3].config["units_sha"] == next(iter(await gate_pins(engine, "sue")))
    plan = await units.h1_plan(engine, parts=["gate_calib", "gate_sue"])
    assert rows[2].config["units_sha"] == plan.sha("gate_calib")
    assert rows[3].config["units_sha"] == plan.sha("gate_sue")


async def test_sigma_c_leaves_out_the_events_meeting_h1s_units(
    sue_world: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An event whose exit session is a train unit leaves Σ_c, and Σ_s with it."""
    events = await sigma_c(sue_world, s_observations())
    taken = next(
        u
        for e in events
        for u in sorted(e.units())
        if sum(u in other.units() for other in events) == 1
    )
    asked: list[Any] = []

    async def h1_windows(engine: AsyncEngine, **kw: Any) -> dict[str, frozenset[Any]]:
        asked.append(kw.get("windows"))
        out = {"train": frozenset({taken}), "validation": frozenset()}
        wanted = kw.get("windows")
        return {k: v for k, v in out.items() if wanted is None or k in wanted}

    monkeypatch.setattr(units, "h1_windows", h1_windows)
    out = await run_sue(sue_world, b=50)
    s0 = next(r for r in out.results if r.gate == "s0")
    assert asked and s0.metrics["sigma_c"]["events"] == 27
    assert s0.metrics["sigma_c"]["counts"]["h1_overlap"] == 1
    s2 = next(r for r in out.results if r.gate == "s2")
    assert s2.metrics["events"] == 27 and s2.config["events"] == 27


async def test_the_sue_gates_refuse_the_minute_gates_while_units_are_not_done(
    sue_world: AsyncEngine,
) -> None:
    complement = await sigma_c(sue_world, s_observations())
    sample = units.sue_sample(complement)
    await undone(sue_world, (sample[0].symbol, sample[0].exits[-1]))
    await undone(sue_world, CALIB_PAIRS[0])
    out = await run_sue(sue_world, b=50)
    assert [r.gate for r in out.results] == ["s0", "s1"]
    assert set(out.refused) == {"s1-calib", "s2", "s3"}
    assert [r.config["gate"] for r in await gate_rows(sue_world)] == ["s0", "s1"]
    assert await gate_pins(sue_world, "calib") == set() == await gate_pins(sue_world, "sue")


async def test_s2_is_refused_when_the_calibration_fails(
    sue_world: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sim_gate, "MIN_COVERAGE", 1.01)  # nothing can cover 101%
    out = await run_sue(sue_world, b=50)
    gates = {r.gate: r for r in out.results}
    assert not gates["s1-calib"].passed and "s2" not in gates and "s3" in gates
    assert "s1-calib to pass" in out.refused["s2"]


# ── G1: look-ahead on stored stories ──

G1_DAYS = (date(2016, 3, 8), date(2016, 5, 10))


def bounce_rows(exit_px: float) -> dict[tuple[int, int], tuple[float, ...]]:
    """A fall from 100 to a 95 low, quiet, a reclaim at 10:21, then ``exit_px``."""
    rows: dict[tuple[int, int], tuple[float, ...]] = {}
    for k in range(30):
        p = 100.0 - 0.15 * k
        rows[(9, 30 + k)] = (p, p, p, p, 100.0, p)
    rows[(10, 0)] = (95.65, 95.65, 95.0, 95.2, 100.0, 95.2)
    for m in range(1, 21):
        rows[(10, m)] = (95.3, 95.3, 95.3, 95.3, 10_000.0, 95.3)
    rows[(10, 21)] = (95.3, 95.8, 95.3, 95.8, 10_000.0, 95.6)
    for m in range(22, 60):
        rows[(10, m)] = (96.0, 96.0, 96.0, 96.0, 10_000.0, 96.0)
    return rows


@pytest.fixture
async def g1_world(engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch) -> AsyncEngine:
    monkeypatch.setattr(sim_gate, "G1_MIN_STORIES", 3)  # the world's three stories, not 500
    monkeypatch.setattr(sim_gate, "DETERMINISM_BATCH", 1)  # one path a batch: state carries
    days = sessions(date(2015, 1, 2), date(2016, 10, 31))
    eves = {d for d in days if next_trading_day(d) in G1_DAYS}
    for sym, base in (("SPY", 200.0), ("AAA", 100.0), ("BBB", 60.0)):
        prices = {
            d: (base if d in eves else base + (i % 2) * (2.0 if sym == "SPY" else 1.0))
            for i, d in enumerate(days)
        }
        await daily(engine, sym, prices)
    await liquidity(engine, ["AAA", "BBB"], date(2015, 1, 1), date(2016, 10, 1))
    rows = [
        news_row(n, "AAA", ny(d, 8), "Morgan Stanley Downgrades Acme to Equal-Weight")
        for n, d in enumerate(G1_DAYS, 1)
    ]
    rows.append(news_row(9, "BBB", ny(G1_DAYS[0], 8), "Barclays Downgrades Bolt to Underweight"))
    await store(engine, rows)
    await add_aliases(
        engine,
        [
            ("AAA", "Acme", "name"),
            ("AAA", "AAA", "ticker"),
            ("BBB", "Bolt", "name"),
            ("BBB", "BBB", "ticker"),
        ],
    )
    await build_range(engine, start=date(2016, 1, 4), end=date(2016, 9, 30), force=True)
    await mark_built(engine, *units.G1_RANGE)
    done: set[tuple[str, date]] = set()
    for s in G1_DAYS:
        for k, d in enumerate(units.path(s, units.PATH_SESSIONS, units.G1_RANGE[1])):
            if ("AAA", d) not in done:
                rows_ = bounce_rows(97.6) if k == 0 else None
                await seed_bars(
                    engine, "AAA", session_bars(d, price=97.6, rows=rows_, volume=10_000.0)
                )
                done.add(("AAA", d))
            if s == G1_DAYS[0] and ("BBB", d) not in done:
                await seed_bars(engine, "BBB", session_bars(d, price=59.0, volume=10_000.0))
                done.add(("BBB", d))
            if ("SPY", d) not in done:
                await seed_bars(engine, "SPY", session_bars(d, price=200.0, volume=1e5))
                done.add(("SPY", d))
    await mark_done(engine, sorted(done))
    return engine


async def test_g1_runs_on_the_stored_stories_and_writes_three_rows(g1_world: AsyncEngine) -> None:
    engine = g1_world
    chosen = await units.g1_stories(engine)
    assert {g.story_id for g in chosen} == {"AAA:2016-03-08", "AAA:2016-05-10", "BBB:2016-03-08"}
    out = await run_lookahead(engine, workers=2, synthetic_paths=40)
    assert not out.refused
    by_id = {r.gate: r for r in out.results}
    assert list(by_id) == ["g1-synthetic", "g1-lookahead", "g1-determinism"]
    assert all(r.passed for r in out.results), [r.metrics for r in out.results]
    look = by_id["g1-lookahead"].metrics
    assert look["selection"] == {"stories": 3, "nsn": 3}
    for cell in ("hold1", "hold3"):
        c = look["cells"][cell]
        assert c["stories"] == 3 and c["started"] == 3 and c["probes"] == 15
        assert c["trades"] >= 2 and c["mismatches"] == 0
        assert c["entries"] >= c["trades"] and c["exits"] >= 1
        assert c["dismissed_share"] == c["dismissed"] / c["outcomes"]
    rows = await gate_rows(engine)
    assert [r.config["gate"] for r in rows] == ["g1-synthetic", "g1-lookahead", "g1-determinism"]
    assert rows[1].config["units_sha"] == next(iter(await gate_pins(engine, "g1")))
    assert rows[1].config["units_sha"] == (await units.h1_plan(engine, parts=["gate_g1"])).sha(
        "gate_g1"
    )
    assert rows[2].config["workers"] == 2 and rows[1].window == "2016-01-04..2016-09-30"
    # g1-determinism ran sim.run itself: serial, then a pool of two in one-path batches.
    det = by_id["g1-determinism"].metrics["g1"]
    assert det["pool"] == ("spawn" if sys.platform == "darwin" else "fork")
    assert det["workers"] == 2 and det["batch_paths"] == 1
    for cell in ("hold1", "hold3"):
        shas = det["sha256"][cell]
        assert shas["workers_1"] == shas["workers_2"]
        c = det["cells"][cell]
        assert c["passed"] and c["outcomes"] == c["started"] == 3 and c["left_out"] == 0
        assert c["trades"] >= 2


async def test_g1_s_md3_determinism_runs_the_stories_whose_path_fits(
    g1_world: AsyncEngine,
) -> None:
    """A story on 2016-09-29 has two sessions left: the look-ahead runs it on them, and
    sim.run (which would ask for three, past the window) only carries its news."""
    engine = g1_world
    late = date(2016, 9, 29)
    await store(engine, [news_row(10, "AAA", ny(late, 8), "Morgan Stanley Downgrades Acme")])
    await build_range(engine, start=date(2016, 1, 4), end=date(2016, 9, 30), force=True)
    await mark_built(engine, *units.G1_RANGE)
    done = []
    for d in units.path(late, units.PATH_SESSIONS, units.G1_RANGE[1]):
        await seed_bars(engine, "AAA", session_bars(d, price=97.6, volume=10_000.0))
        await seed_bars(engine, "SPY", session_bars(d, price=200.0, volume=1e5))
        done += [("AAA", d), ("SPY", d)]
    await mark_done(engine, done)
    assert len(await units.g1_stories(engine)) == 4
    out = await run_lookahead(engine, workers=2, synthetic_paths=20)
    assert not out.refused and all(r.passed for r in out.results), out.results
    by_id = {r.gate: r for r in out.results}
    look = by_id["g1-lookahead"].metrics["cells"]
    assert look["hold1"]["stories"] == look["hold3"]["stories"] == 4
    det = by_id["g1-determinism"].metrics["g1"]["cells"]
    assert det["hold1"]["left_out"] == 0 and det["hold1"]["outcomes"] == 4
    assert det["hold3"]["left_out"] == 1 and det["hold3"]["outcomes"] == 3


class _PidTagged(OverreactionBounce):
    """The bounce, each buy's variant the process it was decided in: records that depend
    on where they are simulated."""

    def start(self, ctx):  # type: ignore[no-untyped-def]
        return self._tag(super().start(ctx))

    def on(self, ev, ctx):  # type: ignore[no-untyped-def]
        return self._tag(super().on(ev, ctx))

    @staticmethod
    def _tag(out):  # type: ignore[no-untyped-def]
        return [
            replace(i, facts=replace(i.facts, variant=f"pid{os.getpid()}"))
            if isinstance(i, Submit) and i.facts is not None
            else i
            for i in out
        ]


class _PidFactory(G1Factory):
    def __call__(self, story):  # type: ignore[no-untyped-def]
        pre, elig = self.context[story.story_id]
        return _PidTagged(story, pre, elig, self.params[story.story_id])


class _PidWorld(LookaheadWorld):
    def factory(self, hold: int) -> G1Factory:
        f = super().factory(hold)
        return _PidFactory(f.context, f.params, f.hold)


async def test_g1_determinism_fails_records_that_depend_on_the_process(
    g1_world: AsyncEngine,
) -> None:
    engine = g1_world
    chosen = await units.g1_stories(engine)
    unit_set = units.g1_units(chosen)
    plan = await sim_gate.plan_h(engine, "gate_g1")
    sha = await pin_units(engine, "g1", "gate_g1", unit_set, plan)
    world = await sim_gate.g1_world(engine, chosen, unit_set, sha)
    ok, good = await run_determinism(engine, world, unit_set, sha, workers=2, batch_paths=1)
    assert ok and good["sha256"]["hold1"]["workers_1"] == good["sha256"]["hold1"]["workers_2"]
    tagged = _PidWorld(world.stories, world.paths, world.spy, world.ctx, world.context, world.holds)
    bad_ok, bad = await run_determinism(engine, tagged, unit_set, sha, workers=2, batch_paths=1)
    assert not bad_ok
    for cell in ("hold1", "hold3"):
        assert bad["sha256"][cell]["workers_1"] != bad["sha256"][cell]["workers_2"]
        assert not bad["cells"][cell]["passed"]
    assert "DIFFER" in sim_gate.describe(
        GateResult("g1-determinism", bad_ok, {"g1": bad, "synthetic": {}})
    )


async def test_g1_refuses_its_real_gates_below_500_stories(
    g1_world: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sim_gate, "G1_MIN_STORIES", 500)
    out = await run_lookahead(g1_world, workers=2, synthetic_paths=20)
    assert [r.gate for r in out.results] == ["g1-synthetic"]
    assert set(out.refused) == {"g1-lookahead", "g1-determinism"}
    assert "holds 3 stories, fewer than the 500" in out.refused["g1-lookahead"]
    assert await gate_pins(g1_world, "g1") == set()


async def test_g1_refuses_its_real_gates_while_a_unit_is_not_done(g1_world: AsyncEngine) -> None:
    await undone(g1_world, ("AAA", G1_DAYS[1]))
    out = await run_lookahead(g1_world, workers=2, synthetic_paths=20)
    assert [r.gate for r in out.results] == ["g1-synthetic"]
    assert set(out.refused) == {"g1-lookahead", "g1-determinism"}
    assert [r.config["gate"] for r in await gate_rows(g1_world)] == ["g1-synthetic"]
    assert await gate_pins(g1_world, "g1") == set()


async def test_g1_refuses_stories_that_no_longer_rebuild_as_selected(
    g1_world: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = await units.g1_stories(g1_world)
    stale = [units.GateStory(g.story_id, g.symbol, g.session, False) for g in real]

    async def chosen(engine: AsyncEngine, **_: Any) -> list[units.GateStory]:
        return stale

    monkeypatch.setattr(units, "g1_stories", chosen)
    out = await run_lookahead(g1_world, workers=2, synthetic_paths=20)
    assert "differ when rebuilt" in out.refused["g1-lookahead"]
    assert await gate_pins(g1_world, "g1") == set()  # refused before the pin


async def test_all_runs_every_group_even_when_one_is_refused(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def lookahead(engine: Any, **_: Any) -> GateRun:
        return GateRun([GateResult("g1-synthetic", True, {})], {"g1-lookahead": "units"})

    async def reactor(engine: Any, **_: Any) -> GateRun:
        return GateRun(refused=dict.fromkeys(("r0", "r1", "r2"), "units"))

    async def sue(engine: Any, **_: Any) -> GateRun:
        return GateRun([GateResult("s0", True, {})])

    monkeypatch.setattr(sim_gate, "run_lookahead", lookahead)
    monkeypatch.setattr(sim_gate, "run_reactor", reactor)
    monkeypatch.setattr(sim_gate, "run_sue", sue)
    out = await sim_gate.run_gates(engine, "all")
    assert [r.gate for r in out.results] == ["g1-synthetic", "s0"]
    assert set(out.refused) == {"g1-lookahead", "r0", "r1", "r2"} and not out.passed
    only = await sim_gate.run_gates(engine, "sue")
    assert [r.gate for r in only.results] == ["s0"] and only.passed
