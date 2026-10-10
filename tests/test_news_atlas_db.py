"""The path atlas against the database: the H1 lock, a seeded week end to end, the command.

Every price is synthetic (``tests/_atlas.py``); no stored market data is read.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections import Counter
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
    candidates,
    h1_closed,
    run_atlas,
    write_atlas,
)
from halal_trader.events.h1 import NAME as H1_NAME
from halal_trader.events.h1 import STAGE_A_FAIL
from halal_trader.events.stories import StoriesNotReady, build_range, pins
from tests._atlas import CONFIG, END, S1, S2, START, daily_closes, register_h1, seed_world
from tests._stories import add_aliases, news_row, ny, store

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


# ── the stories it may describe ───────────────────────────────


async def test_the_atlas_refuses_story_pins_h1_did_not_register(
    engine: AsyncEngine, small_map: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    reg = await seed_world(engine)
    assert (await h1_closed(engine)).pins == await pins(engine)
    # A new alias changes alias_sha: these are not the stories H1 ran on.
    await add_aliases(engine, [("ALFA", "Alfa Holdings", "name")])

    async def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("read stories before the pins were checked")

    monkeypatch.setattr(atlas, "candidates", never)
    with pytest.raises(AtlasLocked, match=f"pins changed since .* registration {reg}: alias_sha"):
        await run_atlas(engine, start=START, end=END)


async def test_a_registration_without_pins_is_refused(engine: AsyncEngine, small_map: None) -> None:
    await seed_world(engine)
    await register_h1(engine)  # newer, closed, and without pins
    with pytest.raises(AtlasLocked, match="alias_sha, builder_version"):
        await run_atlas(engine, start=START, end=END)


def test_changed_pins_names_every_difference() -> None:
    assert atlas.changed_pins({"a": "1", "b": "2"}, {"a": "1", "b": "3", "c": "4"}) == ["b", "c"]
    assert atlas.changed_pins({"a": "1"}, {"a": "1"}) == []


async def test_a_range_without_stories_is_refused(engine: AsyncEngine, small_map: None) -> None:
    await seed_world(engine)
    with pytest.raises(StoriesNotReady, match="no stories-v1 stories with S in 2017-04-03"):
        await run_atlas(engine, start=date(2017, 4, 3), end=date(2017, 4, 3))


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


async def test_candidates_read_the_unit_items_from_postgres(world: AsyncEngine) -> None:
    counts: Counter[str] = Counter()
    chosen = await candidates(world, START, END, counts)
    # Six stories; ECHO's preview is noise only; DLTA is not halal.
    assert (counts["stories"], counts["substantive"], counts["candidates"]) == (6, 5, 4)
    assert sorted(chosen) == [
        "ALFA:2017-03-07",
        "ALFA:2017-03-09",
        "BRVO:2017-03-08",
        "CHRL:2017-03-07",
    ]


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
    # Five sessions: one story a week is 50.4 a year, in the type cell and in its regimes.
    assert result.meta["years"] == pytest.approx(5 / 252)
    assert type_cells[("analyst_downgrade",)].per_year == pytest.approx(252 / 5)
    screen = next(c for c in result.cells if c.table == "screen_regime" and c.n == 1)
    assert screen.key[1] == "before" and screen.per_year == pytest.approx(252 / 5)
    assert result.meta["state_years"]["screen_regime:before"] == pytest.approx(5 / 252)
    assert "screen_regime:from" not in result.meta["state_years"]
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


async def test_spy_regimes_do_not_move_with_the_range(world: AsyncEngine) -> None:
    closes = daily_closes("SPY")
    days = sorted(closes)
    expected = atlas.spy_regimes(
        days,
        [closes[d] for d in days],
        edges_from=date(2016, 10, 3),
        edges_to=date(2021, 12, 31),
    )
    full = await run_atlas(world, start=START, end=END)
    short = await run_atlas(world, start=S2, end=S2)
    assert full.meta["spy_vol_edges"] == short.meta["spy_vol_edges"] == list(expected.edges)
    state = await atlas.spy_state(world)
    assert state.edges == expected.edges
    # Known from the first session the atlas covers; the SMA from 2016-10-18 only.
    assert state.vol_tercile(date(2016, 10, 3)) != "n/a"
    assert state.trend(date(2016, 10, 17)) == "n/a" and state.trend(date(2016, 10, 18)) != "n/a"
    by_id = {r.story_id: r for r in full.rows}
    alfa = by_id["ALFA:2017-03-07"]
    assert alfa.spy_vol == expected.vol_tercile(S1) and alfa.spy_trend == expected.trend(S1)


async def _add_headline(engine: AsyncEngine, n: int, day: date, headline: str) -> None:
    """One more ALFA headline at 08:00 on ``day``, and March's stories built again."""
    await store(engine, [news_row(n, "ALFA", ny(day, 8), headline)])
    await build_range(engine, start=date(2017, 3, 1), end=date(2017, 3, 31))


async def test_a_story_is_blocked_behind_a_live_story_of_its_own_lane(world: AsyncEngine) -> None:
    await _add_headline(world, 7, S2, "Goldman Sachs Downgrades Alfa to Sell")
    result = await run_atlas(world, start=START, end=END)
    by_id = {r.story_id: r for r in result.rows}
    first, second = by_id["ALFA:2017-03-07"], by_id["ALFA:2017-03-08"]
    assert (first.lane, second.lane) == ("NSN_CORE", "NSN_CORE") and second.nsn
    # MD3: the first holds from S1 10:06 to its target at S2's noon, so the second
    # starts at S2's open behind it. ID: the first is flat by S1's close.
    assert first.md3_run is not None and first.md3_run.exit_reason == "target"
    assert second.md3_run is not None and second.md3_run.blocked and not second.md3_run.ran
    assert second.id_run is not None and not second.id_run.blocked and second.id_run.ran
    stats = atlas.cell_stats([r for r in result.rows if r.type == "analyst_downgrade"])
    assert (stats["md3_starters"], stats["md3_blocked"], stats["md3_runs"]) == (2, 1, 1)
    assert stats["md3_blocked_share"] == 0.5
    assert (stats["id_starters"], stats["id_blocked"], stats["id_runs"]) == (2, 0, 2)
    md3 = result.meta["machine_lanes"]["MD3"]
    assert md3["NSN_CORE"]["terminal:DISMISSED/blocked_open"] == 1
    assert result.meta["machine"]["MD3"]["terminal:DISMISSED/blocked_open"] == 1


async def test_another_lane_never_blocks_a_story(
    world: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _add_headline(
        world, 7, S2, "Morgan Stanley Maintains Equal-Weight on Alfa, Lowers Price Target to $90"
    )
    result = await run_atlas(world, start=START, end=END)
    by_id = {r.story_id: r for r in result.rows}
    cut = by_id["ALFA:2017-03-08"]
    assert (cut.type, cut.nsn, cut.lane) == ("analyst_pt_cut", False, "analyst_pt_cut")
    # Its own lane: nothing of it is live when it starts, though ALFA's NSN story holds.
    assert cut.md3_run is not None and cut.md3_run.ran and not cut.md3_run.blocked
    assert by_id["ALFA:2017-03-07"].md3_run is not None
    assert by_id["ALFA:2017-03-07"].md3_run.exit_reason == "target"
    assert set(result.meta["machine_lanes"]["MD3"]) == {
        "NSN_CORE",
        "analyst_pt_cut",
        "dilution",
        "fraud_probe",
    }
    # One run for every lane (the defect) would have blocked it behind the NSN position.
    monkeypatch.setattr(
        atlas,
        "lanes_of",
        lambda units: {"all": {u.story.story_id: u for u in units if u.lane is not None}},
    )
    merged = await run_atlas(world, start=START, end=END)
    cut_merged = next(r for r in merged.rows if r.story_id == "ALFA:2017-03-08")
    assert cut_merged.md3_run is not None and cut_merged.md3_run.blocked


async def test_workers_do_not_change_the_atlas(world: AsyncEngine) -> None:
    one = await run_atlas(world, start=START, end=END, workers=1)
    three = await run_atlas(world, start=START, end=END, workers=3)
    a, b = atlas.to_json(one), atlas.to_json(three)
    assert json.dumps(a["rows"]) == json.dumps(b["rows"])
    assert json.dumps(a["cells"]) == json.dumps(b["cells"])
    assert a["meta"]["counts"] == b["meta"]["counts"]


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
def test_the_command_refuses_a_range_without_stories(database_url: str) -> None:
    _run(database_url, seed_world)
    result = CliRunner().invoke(
        cli, ["events", "atlas", "--start", "2017-04-03", "--end", "2017-04-03"]
    )
    assert result.exit_code == 1
    assert "no stories-v1 stories" in result.output and "Traceback" not in result.output


def test_the_command_refuses_an_end_that_is_not_a_session() -> None:
    result = CliRunner().invoke(cli, ["events", "atlas", "--end", "2019-06-30"])
    assert result.exit_code == 1
    assert "2019-06-30 is not a trading session" in result.output
    assert "Traceback" not in result.output


async def test_a_weekend_end_is_refused_before_the_database(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("read the ledger before the range was checked")

    monkeypatch.setattr(atlas, "h1_closed", never)
    with pytest.raises(ValueError, match="not a trading session"):
        await run_atlas(engine, start=START, end=date(2017, 3, 11))


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
