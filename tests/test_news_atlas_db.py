"""The path atlas against the database: the H1 lock, a seeded week end to end, the command.

Every price is synthetic (``tests/_atlas.py``); no stored market data is read.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from halal_trader.cli import cli
from halal_trader.db.repos.quant_trials import QuantTrialRepoImpl
from halal_trader.events import atlas
from halal_trader.events.atlas import (
    OUTPUT_NAME,
    AtlasLocked,
    h1_closed,
    run_atlas,
    write_atlas,
)
from halal_trader.events.h1 import NAME as H1_NAME
from halal_trader.events.h1 import STAGE_A_FAIL
from tests._atlas import CONFIG, END, S1, S2, START, daily_closes, register_h1, seed_world

COST = 7.0 / 1e4  # study.cost_bps for a rank below 300, one way


# ── the lock ──────────────────────────────────────────────────


async def test_the_atlas_waits_for_h1s_verdict(engine: AsyncEngine) -> None:
    with pytest.raises(AtlasLocked, match="no research.news.h1 preregistration"):
        await h1_closed(engine)
    reg = await register_h1(engine, closing=None)
    repo = QuantTrialRepoImpl(engine)
    with pytest.raises(AtlasLocked, match=f"registration {reg} has no verdict"):
        await h1_closed(engine)
    # A Stage-A row that is no failure for lack of events does not open it.
    await repo.record_trial(name=H1_NAME, kind="stage-a", config=CONFIG, verdict=None)
    # Nor a verdict under another configuration.
    await repo.record_trial(name=H1_NAME, kind="verdict", config={"other": 1}, verdict="fail")
    with pytest.raises(AtlasLocked):
        await h1_closed(engine)
    closing = await repo.record_trial(
        name=H1_NAME, kind="stage-a", config=CONFIG, verdict=STAGE_A_FAIL
    )
    found = await h1_closed(engine)
    assert (found.id, found.closed_by) == (reg, closing)
    assert found.closing == f"stage-a: {STAGE_A_FAIL}"


async def test_a_verdict_opens_it_and_a_newer_registration_closes_it_again(
    engine: AsyncEngine,
) -> None:
    reg = await register_h1(engine, closing="verdict")
    assert (await h1_closed(engine)).id == reg
    await QuantTrialRepoImpl(engine).record_trial(
        name=H1_NAME, kind="preregistration", config={**CONFIG, "version": 2}
    )
    with pytest.raises(AtlasLocked, match="has no verdict"):
        await h1_closed(engine)


async def test_a_locked_atlas_reads_nothing_else(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("read stories before the lock was checked")

    monkeypatch.setattr(atlas, "candidates", never)
    with pytest.raises(AtlasLocked):
        await run_atlas(engine, start=START, end=END)


async def test_a_range_past_the_train_window_is_refused_before_the_database(
    engine: AsyncEngine,
) -> None:
    await register_h1(engine)
    with pytest.raises(ValueError, match="ends by 2021-12-23"):
        await run_atlas(engine, start=START, end=date(2021, 12, 27))
    with pytest.raises(ValueError, match="starts on 2016-10-03"):
        await run_atlas(engine, start=date(2016, 9, 30), end=END)


# ── a seeded week ─────────────────────────────────────────────


@pytest.fixture
async def world(engine: AsyncEngine, small_map: None) -> AsyncEngine:
    await seed_world(engine)
    return engine


async def test_the_units_are_the_primary_stories_with_a_substantive_item(
    world: AsyncEngine,
) -> None:
    result = await run_atlas(world, start=START, end=END)
    assert [r.story_id for r in result.rows] == [
        "ALFA:2017-03-07",
        "CHRL:2017-03-07",
        "BRVO:2017-03-08",
        "ALFA:2017-03-09",
    ]  # DLTA is not halal; ECHO's story is an earnings preview (noise) only
    counts = result.meta["counts"]
    assert counts["candidates"] == 4 and counts["rebuilt"] == 4
    assert counts["eligibility:not_halal"] == 1
    assert counts["measured"] == 4
    assert counts.get("persisted_mismatch", 0) == 0 and counts["missing_rebuilt"] == 0
    by_id = {r.story_id: r for r in result.rows}
    assert by_id["ALFA:2017-03-07"].type == "analyst_downgrade"
    assert by_id["ALFA:2017-03-09"].type == "fraud_probe"
    assert by_id["BRVO:2017-03-08"].type == "dilution"
    assert by_id["CHRL:2017-03-07"].type == "product"
    assert result.meta["registration"]["config_hash"]
    assert result.meta["pins"]["builder_version"] == "stories-v1"


async def test_an_nsn_story_is_measured_from_its_previous_close(world: AsyncEngine) -> None:
    result = await run_atlas(world, start=START, end=END)
    row = next(r for r in result.rows if r.story_id == "ALFA:2017-03-07")
    assert (row.nsn, row.family_ever, row.start_case, row.timing) == (
        True,
        "NSN_CORE",
        "out",
        "pre_open",
    )
    assert row.rank == 0 and row.tech and row.cost_bps == 7.0
    m = row.measures
    assert m is not None and row.sigma is not None and row.skip is None
    sigma = row.sigma
    assert m.p0 == 100.0 and m.spy0 == 200.0
    assert m.gap_sigma == pytest.approx((97.0 / 100.0 - 1.0) / sigma, abs=1e-12)
    assert m.low_sigma == pytest.approx((94.5 / 100.0 - 1.0) / sigma, abs=1e-12)
    assert (m.low, m.t_low) == (94.0, 10.0)
    assert m.retrace_close == pytest.approx(0.2, abs=1e-12)
    assert m.retrace_max_s == pytest.approx(0.2, abs=1e-12)
    assert m.retrace_max_s2 == pytest.approx(0.6, abs=1e-12)  # 97.6 on S+1
    assert m.fade is True  # S+2 trades below 94
    closes = daily_closes("ALFA")
    days = {1: S2, 3: date(2017, 3, 10), 5: date(2017, 3, 14)}
    for h, value in zip((1, 3, 5), row.cont):
        assert value == pytest.approx(closes[days[h]] / closes[S1] - 1.0, abs=1e-12)


async def test_the_machine_runs_without_the_family_check(world: AsyncEngine) -> None:
    result = await run_atlas(world, start=START, end=END)
    by_id = {r.story_id: r for r in result.rows}
    alfa = by_id["ALFA:2017-03-07"]
    assert alfa.machine and alfa.id_run is not None and alfa.md3_run is not None
    # It reclaims its VWAP on the 10:06 bar and fills at 95.2; ID holds to the flatten.
    assert (alfa.id_run.triggered, alfa.id_run.armed, alfa.id_run.entered) == (True, True, True)
    assert alfa.id_run.exit_reason == "time_stop"
    assert alfa.id_run.r == pytest.approx(-2 * COST, abs=1e-12)
    # MD3 reaches the 50% target (97) on S+1's noon rally.
    assert alfa.md3_run.exit_reason == "target"
    assert alfa.md3_run.r == pytest.approx(97.6 / 95.2 - 1.0 - 2 * COST, abs=1e-12)
    # A structural story is a negative one: it runs too, and never triggers on a flat path.
    probe = by_id["ALFA:2017-03-09"]
    assert probe.machine and not probe.nsn and probe.start_case == "in"
    assert probe.id_run is not None and probe.id_run.ran and not probe.id_run.triggered
    assert probe.id_run.reason == "cutoff"
    offering = by_id["BRVO:2017-03-08"]
    assert offering.machine and offering.id_run is not None and offering.id_run.triggered
    # A positive story is described, never run.
    launch = by_id["CHRL:2017-03-07"]
    assert not launch.machine and launch.id_run is None and launch.md3_run is None
    assert launch.measures is not None and launch.measures.retrace_close is None  # no drop
    machine = result.meta["machine"]
    assert machine["ID"]["started"] == 3 and machine["MD3"]["started"] == 3


async def test_an_in_session_story_is_anchored_at_its_news(world: AsyncEngine) -> None:
    result = await run_atlas(world, start=START, end=END)
    row = next(r for r in result.rows if r.story_id == "BRVO:2017-03-08")
    m = row.measures
    assert m is not None and row.sigma is not None
    assert (row.start_case, row.timing) == ("in", "in_session")
    assert m.p0 == 50.0  # the 10:59 bar's close
    assert m.gap_sigma == pytest.approx((48.5 / 50.0 - 1.0) / row.sigma, abs=1e-12)
    assert m.low == 47.0 and m.t_low == 5.0
    assert m.retrace_close == pytest.approx((47.3 - 47.0) / 3.0, abs=1e-12)


async def test_the_cells_and_the_file(world: AsyncEngine, tmp_path: Path) -> None:
    result = await run_atlas(world, start=START, end=END)
    assert {c.table for c in result.cells} == set(atlas.TABLES)
    assert all(c.stats is None for c in result.cells)  # every cell is under 30 stories
    type_cells = {c.key: c for c in result.cells if c.table == "type"}
    assert type_cells[("analyst_downgrade",)].n == 1
    coverage = {c.key: c.n for c in result.cells if c.table == "coverage"}
    assert coverage == {
        ("analyst_downgrade", "measured"): 1,
        ("dilution", "measured"): 1,
        ("fraud_probe", "measured"): 1,
        ("product", "measured"): 1,
    }
    path = write_atlas(result, tmp_path / "research" / OUTPUT_NAME)
    data = json.loads(path.read_text())
    assert len(data["rows"]) == 4 and data["meta"]["end"] == "2017-03-10"
    first = data["rows"][0]
    assert first["story_id"] == "ALFA:2017-03-07"
    assert set(first["cont"]) == {"cont_1", "cont_3", "cont_5"}
    assert first["md3_run"]["exit_reason"] == "target"


async def test_the_same_week_gives_the_same_atlas(world: AsyncEngine) -> None:
    one = await run_atlas(world, start=START, end=END)
    two = await run_atlas(world, start=START, end=END, batch_symbols=1)
    assert json.dumps(atlas.to_json(one)["rows"]) == json.dumps(atlas.to_json(two)["rows"])
    assert [c.key for c in one.cells] == [c.key for c in two.cells]
    assert not math.isnan(one.meta["spy_vol_edges"][0])


# ── the command ───────────────────────────────────────────────


def _run(database_url: str, work: Callable[[AsyncEngine], Awaitable[Any]]) -> Any:
    async def go() -> Any:
        engine = create_async_engine(database_url)
        try:
            return await work(engine)
        finally:
            await engine.dispose()

    return asyncio.run(go())


ARGS = ["events", "atlas", "--start", "2017-03-06", "--end", "2017-03-10"]


def test_the_command_refuses_before_h1s_verdict(database_url: str) -> None:
    result = CliRunner().invoke(cli, ARGS)
    assert result.exit_code == 1
    assert "no research.news.h1 preregistration" in result.output
    assert "Traceback" not in result.output


def test_the_command_refuses_a_range_past_train() -> None:
    result = CliRunner().invoke(cli, ["events", "atlas", "--end", "2022-03-01"])
    assert result.exit_code == 1 and "ends by 2021-12-23" in result.output


@pytest.mark.usefixtures("small_map")
def test_the_command_prints_the_tables_and_writes_the_file(
    database_url: str, tmp_path: Path
) -> None:
    _run(database_url, seed_world)
    result = CliRunner().invoke(cli, ARGS)
    assert result.exit_code == 0, result.output
    assert "== type x low_sigma bucket ==" in result.output
    assert "counts only" in result.output
    assert "4 stories, 4 measured; 0 differ from their news_stories row" in result.output
    written = tmp_path / "data" / "research" / OUTPUT_NAME  # DATA_DIR is the test's tmp tree
    assert written.exists() and "written to" in result.output
    assert json.loads(written.read_text())["meta"]["registration"]["closing"] == "verdict: fail"
