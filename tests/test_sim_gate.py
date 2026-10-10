"""The Phase 0 gate runner's pure parts (events/sim_gate.py), on synthetic data.

Each gate's verdict is computed from inputs built here, with a passing and a
failing case for every gate id: G1's look-ahead (a playbook that peeks
fails), its determinism (a factory whose output depends on the processing
order fails), R0's reproduction, R1's exactness, R2's interval, S0's table,
S1's exactness, the calibration, S2's equivalence and S3's interval. The
database runs are in ``test_sim_gate_db.py``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from itertools import count
from typing import Any, cast

import numpy as np
import pytest

from halabot.playbooks import legacy
from halabot.playbooks.bounce import OverreactionBounce
from halabot.playbooks.clock import SIP_RT
from halabot.playbooks.records import TradeRecord
from halabot.playbooks.sim import RunSummary, id_set_sha, simulate_symbol
from halabot.playbooks.types import (
    BarIn,
    Session,
    SimConfig,
    SpyData,
    Submit,
    TradeFacts,
    Transition,
)
from halal_trader.data.minutes import BarArrays
from halal_trader.events import h1, sim_gate, study
from halal_trader.events.context import PreEvent
from halal_trader.events.intraday import Headline, Outcome
from halal_trader.events.sim_gate import (
    CRITERIA,
    GATE_IDS,
    GROUPS,
    FlattenHoldFactory,
    G1Factory,
    G1Story,
    GateResult,
    GateRun,
    LiquidityEligibility,
    LiquidityOnly,
    LookaheadWorld,
    MinuteReturns,
    Paired,
    Spread,
    calib_result,
    calibration,
    cell_stories,
    daily_mode,
    decile_spread,
    decomposition,
    describe,
    determinism,
    explicit_set,
    gate_config,
    headline_id,
    liquidity_eligibility,
    lookahead,
    r0_result,
    r1_result,
    r2_result,
    realistic_config,
    realistic_return,
    reference_deltas,
    reference_table,
    s0_result,
    s1_result,
    s2_result,
    s3_result,
    spread,
    study_clock_return,
    synthetic_world,
    to_json,
    with_spy,
)
from halal_trader.events.stories import NEWS_LAG, RawItem, Story, StoryItem
from halal_trader.events.study import Observation
from halal_trader.market_hours import MARKET_TZ, is_trading_day
from tests.halabot.playbooks._support import (
    MON,
    TUE,
    WED,
    Context,
    Daily,
    epoch,
    et,
    path,
    session_bars,
)
from tests.halabot.playbooks._synth import random_session

EPS = 1e-12


def ny(day: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm), MARKET_TZ).astimezone(UTC)


# ── the ledger row ──


def test_gate_ids_groups_and_criteria_are_exactly_the_h1_runners() -> None:
    assert GATE_IDS == h1.REQUIRED_GATES  # the ids the H1 runner requires, in order
    assert GATE_IDS == (
        "g1-lookahead",
        "g1-synthetic",
        "g1-determinism",
        "r0",
        "r1",
        "r2",
        "s0",
        "s1",
        "s1-calib",
        "s2",
        "s3",
    )
    assert sorted(g for ids in GROUPS.values() for g in ids) == sorted(GATE_IDS)
    assert set(CRITERIA) == set(GATE_IDS)


def test_the_config_pins_the_simulator_constants_the_gate_and_the_seed() -> None:
    config = gate_config("r1", {"units_sha": "ab"})
    assert config["sim"] == SimConfig().as_config() and config["gate"] == "r1"
    assert config["seed"] == 20261010 and config["units_sha"] == "ab"
    with pytest.raises(ValueError, match="unknown gate"):
        gate_config("g2")


def test_to_json_makes_plain_json() -> None:
    out = to_json(
        {
            "nan": math.nan,
            "inf": -math.inf,
            "np": np.float64(0.5),
            "i": np.int64(3),
            "day": date(2016, 3, 7),
            "t": (1, 2),
            "s": frozenset({"b", "a"}),
            "nested": {1: [math.nan]},
            "flag": True,
        }
    )
    assert out == {
        "nan": None,
        "inf": None,
        "np": 0.5,
        "i": 3,
        "day": "2016-03-07",
        "t": [1, 2],
        "s": ["a", "b"],
        "nested": {"1": [None]},
        "flag": True,
    }


def test_a_gate_run_passes_only_when_every_gate_ran_and_passed() -> None:
    run = GateRun([GateResult("r0", True, {})])
    assert run.passed
    run.refuse(["s2", "s3"], "units")
    assert not run.passed and run.refused == {"s2": "units", "s3": "units"}
    other = GateRun([GateResult("r1", False, {})])
    run.extend(other)
    assert [r.gate for r in run.results] == ["r0", "r1"] and not run.passed
    assert GateResult("r1", False, {}).verdict == "fail"


def test_with_spy_adds_spys_unit_on_every_day() -> None:
    assert with_spy({("A", MON), ("B", TUE)}) == {
        ("A", MON),
        ("B", TUE),
        ("SPY", MON),
        ("SPY", TUE),
    }


# ── G1: stories, eligibility and the context ──


def _item(n: int, at: datetime, itype: str) -> StoryItem:
    raw = RawItem(n, f"t{n}", "news", "AAA", at, at, "", 1)
    return StoryItem(raw, at, at + NEWS_LAG, itype, frozenset())


def _story(day: date, *items: tuple[datetime, str]) -> Story:
    made = [_item(n, at, kind) for n, (at, kind) in enumerate(sorted(items), 1)]
    return Story(f"AAA:{day.isoformat()}", "AAA", day, made)


def test_g1_detects_nsn_stories_at_their_own_nsn_item() -> None:
    st = _story(MON, (ny(MON, 7), "noise"), (ny(MON, 8), "analyst_downgrade"))
    g = G1Story(st, "nsn")
    cutoff = Session.of(MON).entry_cutoff
    assert g.nsn_at(cutoff) == ny(MON, 8) + NEWS_LAG and g.at_news() == ny(MON, 8)
    assert g.start_case() == "out" and g.nsn_at(ny(MON, 8)) is None  # not yet by then
    assert g.card_at(cutoff).family == "NSN_CORE"
    assert list(g.news_times()) == [ny(MON, 7) + NEWS_LAG, ny(MON, 8) + NEWS_LAG]


def test_g1_detects_other_negative_stories_at_their_first_substantive_item() -> None:
    st = _story(MON, (ny(MON, 9, 40), "noise"), (ny(MON, 10, 5), "analyst_pt_cut"))
    g = G1Story(st, "other")
    assert g.nsn_at(Session.of(MON).entry_cutoff) == ny(MON, 10, 15)
    assert g.at_news() == ny(MON, 10, 5) and g.start_case() == "in"
    late = G1Story(_story(MON, (ny(MON, 15, 0), "analyst_pt_cut")), "other")
    assert late.nsn_at(Session.of(MON).entry_cutoff) is None  # after the cutoff: no start
    carrier = G1Story(st, None)
    assert carrier.nsn_at(ny(TUE, 0)) is None and carrier.at_news() is None
    assert carrier.start_case() == "out"


def test_g1_stories_forget_the_items_after_t() -> None:
    st = _story(MON, (ny(MON, 8), "analyst_downgrade"), (ny(MON, 11), "dilution"))
    g = G1Story(st, "nsn")
    early = g.before(ny(MON, 9))
    assert early.mode == "nsn" and len(early.story.items) == 1 and len(g.story.items) == 2
    assert early.card_at(ny(MON, 12)).family == "NSN_CORE"
    assert g.card_at(ny(MON, 12)).family is None  # the dilution vetoes it
    assert g.before(ny(MON, 7)).nsn_at(ny(TUE, 0)) is None


@dataclass(frozen=True)
class _Elig:
    liquidity_rank: int | None


class _Pit:
    def __init__(self, rank: int | None, pre: PreEvent | None, *, raises: bool = False) -> None:
        self.rank, self.pre, self.raises = rank, pre, raises

    def eligibility(
        self, symbol: str, session: date, *, at_news: datetime, universe: str = "primary"
    ) -> _Elig:
        if self.raises:
            raise ValueError("too early")
        assert universe == "broad"
        return _Elig(self.rank)

    def pre_event(self, symbol: str, session: date, at_news: datetime) -> PreEvent | None:
        return self.pre


def _pre(prev: float = 50.0) -> PreEvent:
    nan = math.nan
    return PreEvent(MON, MON, prev, 200.0, 0.01, 60, 1.0, 0.02, 1e7, 0, 0, nan, nan, nan, nan)


@pytest.mark.parametrize(
    ("pit", "reason", "cost"),
    [
        (_Pit(10, _pre()), "ok", 7.0),
        (_Pit(500, _pre()), "ok", 15.0),
        (_Pit(1000, _pre()), "rank", 30.0),
        (_Pit(None, _pre()), "rank", 30.0),
        (_Pit(10, None), "no_pre_event", 7.0),
        (_Pit(10, _pre(4.99)), "price", 7.0),
        (_Pit(10, _pre(), raises=True), "no_pre_event", 30.0),
    ],
)
def test_the_liquidity_only_universe(pit: _Pit, reason: str, cost: float) -> None:
    pre, elig = liquidity_eligibility(pit, "AAA", MON, ny(MON, 8))
    assert elig.reason == reason and elig.eligible == (reason == "ok")
    assert elig.cost_bps == cost and not elig.tech
    assert (pre is None) == (reason == "no_pre_event")


def test_the_liquidity_only_context_admits_every_screen() -> None:
    base = Context(adj={("A", MON): 0.5}, verdicts={("A", MON): "no_screen"}, sessions=[MON])
    view = LiquidityOnly(base)
    assert view.screen_verdict("A", MON) == "halal" and base.screen_verdict("A", MON) == "no_screen"
    assert (
        view.adj("A", MON) == 0.5 and list(view.sessions) == [MON] and view.daily("A", MON) is None
    )


# ── G1: the look-ahead gate on synthetic markets ──


def test_the_synthetic_world_is_seeded_and_holds_both_kinds_of_story() -> None:
    a, b = synthetic_world(n_paths=60), synthetic_world(n_paths=60)
    assert [s.story_id for s in a.stories] == [s.story_id for s in b.stories]
    assert len(a.gate_stories()) == 60
    modes = Counter(s.mode for s in a.stories)
    assert modes["nsn"] > 10 and modes["other"] > 5
    sid = a.stories[0].story_id
    for hold in (1, 3):
        p, q = a.paths[hold][sid], b.paths[hold][sid]
        assert not isinstance(p, legacy.PathSkip) and not isinstance(q, legacy.PathSkip)
        assert len(p.sessions) == hold == a.holds[hold][sid]
        assert np.array_equal(p.bars[0].c, q.bars[0].c)
    other = synthetic_world(seed=1, n_paths=60)
    assert [s.story_id for s in other.stories] != [s.story_id for s in a.stories]
    assert all(is_trading_day(d) for d in a.spy.days)


def test_g1_passes_the_bounce_on_synthetic_paths() -> None:
    world = synthetic_world(n_paths=120)
    ok, metrics = lookahead(world)
    assert ok, metrics
    for hold in ("hold1", "hold3"):
        cell = metrics["cells"][hold]
        assert cell["mismatches"] == 0 and cell["fill_bar_violations"] == []
        assert cell["probes"] == 5 * 120 and cell["checked"] >= cell["probes"] // 2
        assert cell["trades"] > 0 and cell["stories"] == 120
        assert cell["entries"] >= cell["trades"] and cell["exits"] > 0
        states = {k.split("/")[0] for k in cell["terminal"]}
        assert {"EXITED", "EXPIRED", "DISMISSED"} <= states
        dismissed = sum(n for k, n in cell["terminal"].items() if k.startswith("DISMISSED/"))
        assert cell["outcomes"] == sum(cell["terminal"].values()) == cell["started"]
        assert cell["dismissed"] == dismissed and 0 < cell["dismissed_share"] < 1
        assert cell["dismissed_share"] == dismissed / cell["outcomes"]
    again_ok, again = lookahead(world)
    assert again_ok and again == metrics  # seeded: a rerun is identical


class _Peeking(OverreactionBounce):
    """The bounce, but it buys whenever a bar 30 minutes ahead closes higher (look-ahead)."""

    def on(self, ev, ctx):  # type: ignore[no-untyped-def]
        if (
            self.state() == "WATCHING"
            and isinstance(ev, BarIn)
            and ev.symbol == ctx.story.symbol
            and ctx.session.entry_start <= ev.at <= ctx.session.entry_cutoff
        ):
            full = ctx.market._bars
            n = len(ctx.market.bars(ev.symbol))
            if n + 30 < len(full) and full.c[n + 30] > full.c[n - 1]:
                facts = TradeFacts("x", "x", "ID", cost_bps=15.0, rank=1, tech=False)
                self._state = type(self._state)("ENTERING")
                return [Transition("ENTERING", "peek"), Submit("buy", facts=facts)]
        return super().on(ev, ctx)


class _PeekingFactory(G1Factory):
    def __call__(self, story):  # type: ignore[no-untyped-def]
        pre, elig = self.context[story.story_id]
        return _Peeking(story, pre, elig, self.params[story.story_id])


class _PeekingWorld(LookaheadWorld):
    def factory(self, hold: int) -> G1Factory:
        f = super().factory(hold)
        return _PeekingFactory(f.context, f.params, f.hold)


def test_g1_fails_a_playbook_that_peeks() -> None:
    w = synthetic_world(n_paths=60, cells=(1,))
    peeking = _PeekingWorld(w.stories, w.paths, w.spy, w.ctx, w.context, w.holds)
    ok, metrics = lookahead(peeking)
    cell = metrics["cells"]["hold1"]
    assert not ok and not cell["passed"] and cell["mismatches"] > 0
    assert cell["mismatches_shown"][0][2] in ("intents", "transitions", "started")
    assert "1 checked" not in describe(GateResult("g1-synthetic", ok, metrics))


def test_g1_fails_a_cell_that_never_enters() -> None:
    """Every story dismissed for want of σ: invariant, but no order was ever exercised."""
    w = synthetic_world(n_paths=20, cells=(1, 3))
    no_sigma = LiquidityEligibility(False, "no_pre_event", None, study.cost_bps(None))
    blind = dict.fromkeys(w.context, (None, no_sigma))
    dull = LookaheadWorld(w.stories, w.paths, w.spy, w.ctx, blind, w.holds)  # type: ignore[arg-type]
    ok, metrics = lookahead(dull)
    assert not ok
    for hold in ("hold1", "hold3"):
        cell = metrics["cells"][hold]
        assert not cell["passed"] and cell["mismatches"] == 0 and cell["checked"] > 0
        assert cell["entries"] == cell["trades"] == cell["exits"] == 0
        assert cell["dismissed"] == cell["outcomes"] > 0 and cell["dismissed_share"] == 1.0
        assert set(cell["terminal"]) == {"DISMISSED/no_pre_event"}
    assert "0 entries, 0 trades, 0 exits, 100% dismissed" in describe(
        GateResult("g1-lookahead", ok, metrics)
    )


def test_determinism_agrees_across_worker_counts() -> None:
    w = synthetic_world(n_paths=60)
    ok, metrics = determinism(w, workers=6)
    assert ok
    for cell in metrics["sha256"].values():
        assert cell["workers_1"] == cell["workers_6"]


class _Numbered(OverreactionBounce):
    """The bounce, each buy's variant numbered in the order playbooks are built: records
    that depend on how the symbols are partitioned."""

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self._n = next(_COUNT)

    def _tag(self, out):  # type: ignore[no-untyped-def]
        return [
            replace(i, facts=replace(i.facts, variant=f"v{self._n}"))
            if isinstance(i, Submit) and i.facts is not None
            else i
            for i in out
        ]

    def start(self, ctx):  # type: ignore[no-untyped-def]
        return self._tag(super().start(ctx))

    def on(self, ev, ctx):  # type: ignore[no-untyped-def]
        return self._tag(super().on(ev, ctx))


class _Counter(G1Factory):
    def __call__(self, story):  # type: ignore[no-untyped-def]
        pre, elig = self.context[story.story_id]
        return _Numbered(story, pre, elig, self.params[story.story_id])


_COUNT = count()


class _CountingWorld(LookaheadWorld):
    def factory(self, hold: int) -> G1Factory:
        f = super().factory(hold)
        return _Counter(f.context, f.params, f.hold)


def test_determinism_fails_an_order_dependent_factory() -> None:
    w = synthetic_world(n_paths=80, cells=(1,))
    bad = _CountingWorld(w.stories, w.paths, w.spy, w.ctx, w.context, w.holds)
    ok, metrics = determinism(bad, workers=6)
    assert not ok
    cell = metrics["sha256"]["hold1"]
    assert cell["workers_1"] != cell["workers_6"]
    assert "DIFFER" in describe(
        GateResult("g1-determinism", ok, {"g1": metrics, "synthetic": metrics})
    )


def test_a_story_whose_md3_path_is_cut_only_carries_its_news_in_sim_runs_cell() -> None:
    """sim.run asks for the hold's sessions of every started story: a shorter path stays out."""
    w = synthetic_world(n_paths=20)
    gate = w.gate_stories()
    short = gate[0].story_id
    holds = {h: dict(v) for h, v in w.holds.items()}
    holds[3][short] = 2  # S on 2016-09-29: two sessions left in the window
    cut = LookaheadWorld(w.stories, w.paths, w.spy, w.ctx, w.context, holds)
    md3, left = cell_stories(cut, 3)
    assert left == 1 and [s.story_id for s in md3] == [s.story_id for s in w.stories]
    by_id = {s.story_id: s for s in md3}
    assert by_id[short].mode is None and by_id[short].story is gate[0].story
    assert by_id[short].nsn_at(Session.of(gate[0].session).entry_cutoff) is None  # never starts
    assert all(by_id[s.story_id] == s for s in gate[1:])
    assert cell_stories(cut, 1) == (list(w.stories), 0)


# ── G2: the reactor ──


def _headlines() -> list[Headline]:
    out = []
    for i, day in enumerate((MON, TUE, WED)):
        for j, sym in enumerate(("AAA", "BBB", "CCC", "DDD")):
            score = (0.9, 0.5, 0.1, -0.6)[j]
            out.append(Headline(sym, ny(day, 10 + j, 5 * i), score))
    return out


def test_headline_ids_are_symbol_and_new_york_day() -> None:
    h = Headline("AAA", datetime(2016, 3, 8, 3, 0, tzinfo=UTC), 0.5)  # 22:00 NY on the 7th
    assert headline_id(h) == "AAA:2016-03-07"
    assert explicit_set([h]) == {"AAA:2016-03-07": ("AAA", MON)}
    with pytest.raises(ValueError, match="share"):
        explicit_set([h, replace(h, published_at=h.published_at + timedelta(hours=1))])


def _outcomes(headlines: list[Headline], value: float) -> list[Outcome]:
    rng = np.random.default_rng(0)
    return [Outcome(h, value + float(rng.normal(0, 0.001)), None, None) for h in headlines]


def test_r0_reproduces_the_recorded_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    hs = _headlines()
    outcomes = _outcomes(hs, -0.003)
    strong = [o.same_day for o in outcomes if o.headline.score >= 0.4]
    control = [o.same_day for o in outcomes if abs(o.headline.score) < 0.4]
    neg = [o.same_day for o in outcomes if o.headline.score <= -0.4]
    sd = float(np.std(strong, ddof=1))
    expect = {
        "headlines": 40,
        "strong_headlines": 6,
        "strong_n": 6,
        "strong_mean_pct": round(float(np.mean(strong)) * 100, 2),
        "strong_t": round(float(np.mean(strong)) / (sd / math.sqrt(6)), 1),
        "control_mean_pct": round(float(np.mean(control)) * 100, 2),
        "negative_mean_pct": round(float(np.mean(neg)) * 100, 2),
    }
    monkeypatch.setattr(sim_gate, "R0_EXPECTED", expect)
    r = r0_result(40, hs, outcomes)
    assert r.passed and all(c["ok"] for c in r.metrics["checks"].values())
    assert "strong_n 6 (want 6)" in describe(r)
    # One headline fewer in the study: n and the mean move, and R0 fails.
    worse = r0_result(40, hs, [o for o in outcomes if o.headline is not hs[0]])
    assert not worse.passed and not worse.metrics["checks"]["strong_n"]["ok"]
    assert not r0_result(41, hs, outcomes).passed  # another pinned set
    assert not r0_result(40, hs, []).passed


def test_r0_s_real_expectations_are_the_recorded_ones() -> None:
    assert sim_gate.R0_EXPECTED == {
        "headlines": 30_614,
        "strong_headlines": 7_932,
        "strong_n": 7_922,
        "strong_mean_pct": -0.30,
        "strong_t": -12.1,
        "control_mean_pct": -0.28,
        "negative_mean_pct": -0.40,
    }


@dataclass(frozen=True)
class _Trade:
    """The fields of a TradeRecord the reactor gates read."""

    story_id: str
    entry_px: float
    exit_px: float
    spy_entry_px: float
    spy_exit_px: float
    cost_bps: float
    r_net_abn: float
    run_id: str = "r"  # _summary's run


def _trade(sid: str, e: float, x: float, se: float, sx: float, c: float = 7.0) -> TradeRecord:
    r = (x / e - 1) - (sx / se - 1) - 2 * c / 1e4
    return cast(TradeRecord, _Trade(sid, e, x, se, sx, c, r))


def _summary(ids: Sequence[str], **kw: Any) -> RunSummary:
    """Run ``r``'s summary over the explicit set ``ids``."""
    n = len(ids)
    base: dict[str, Any] = dict(
        run_id="r",
        stories=n,
        started=n,
        outcomes=n,
        entries=n,
        trades=n,
        terminal={},
        skips={},
        loader={},
        skip_ids={},
        bar_drop_ids=(),
        expected=n,
        expected_sha=id_set_sha(ids),
    )
    return RunSummary(**{**base, **kw})


def test_r1_passes_when_every_kept_headline_matches_and_fails_otherwise() -> None:
    hs = _headlines()
    ids = list(explicit_set(hs))
    trades = [_trade(sid, 50.0, 50.5 + 0.1 * k, 200.0, 200.2) for k, sid in enumerate(ids[:-1])]
    study_r = {t.story_id: t.r_net_abn for t in trades}
    summary = _summary(ids, dropped={ids[-1]: "entry_unfilled"})
    ok = r1_result(hs, summary, trades, study_r)
    assert ok.passed and ok.metrics["compared"] == len(ids) - 1
    assert ok.metrics["max_abs_diff"] == 0.0 and ok.metrics["over_tolerance"] == 0
    assert not {"skip_ids", "spy_drop_from_start_ids", "dropped"} & set(ok.metrics["run"])
    off = dict(study_r)
    off[ids[0]] += 1e-9
    bad = r1_result(hs, summary, trades, off)
    assert not bad.passed and bad.metrics["over_shown"] == {ids[0]: pytest.approx(1e-9)}
    tiny = dict(study_r)
    tiny[ids[0]] += 5e-11
    assert r1_result(hs, summary, trades, tiny).passed
    # The study drops a headline the simulator kept: the dropped sets differ.
    less = {k: v for k, v in study_r.items() if k != ids[1]}
    differ = r1_result(hs, summary, trades, less)
    assert not differ.passed and differ.metrics["dropped"]["study_only"] == [ids[1]]
    assert "study only 1" in describe(differ)


def test_the_r2_decomposition_splits_the_difference_between_entry_and_exit() -> None:
    old = _trade("a", 50.0, 51.0, 200.0, 201.0)
    new = _trade("a", 50.2, 50.8, 200.1, 200.9)
    entry_d, exit_d = decomposition(old, new)
    assert abs(entry_d + exit_d - (new.r_net_abn - old.r_net_abn)) <= EPS
    mixed = (51.0 / 50.2 - 1) - (201.0 / 200.1 - 1) - 2 * 7.0 / 1e4
    assert abs(entry_d - (mixed - old.r_net_abn)) <= EPS
    assert decomposition(old, old) == (0.0, 0.0)


def test_r2_judges_the_strong_groups_mean_against_the_legacy_clustered_interval() -> None:
    hs = _headlines()
    ids = list(explicit_set(hs))
    rng = np.random.default_rng(1)
    r1_trades = [_trade(s, 50.0, 50.0 * (1 + rng.normal(0, 0.01)), 200.0, 200.0) for s in ids]
    study_r = {t.story_id: t.r_net_abn for t in r1_trades}
    close = [_trade(t.story_id, t.entry_px * 1.0002, t.exit_px, 200.0, 200.0) for t in r1_trades]
    r2 = r2_result(hs, study_r, r1_trades, (), _summary(ids), close)
    assert r2.passed, r2.metrics
    lo, hi = r2.metrics["legacy_ci95"]
    assert lo <= r2.metrics["realistic"]["mean"] <= hi
    assert r2.metrics["legacy"]["clusters"] == 3 and r2.metrics["legacy"]["n"] == 6
    d = r2.metrics["decomposition"]
    assert d["n"] == 6 and len(d["per_headline"]) == 6 and d["exit_mean"] == pytest.approx(0.0)
    assert d["entry_mean"] < 0  # a dearer entry
    assert "clustered t" in describe(r2)
    # Realistic fills 5% worse: far outside the interval.
    far = [_trade(t.story_id, t.entry_px * 1.05, t.exit_px, 200.0, 200.0) for t in r1_trades]
    bad = r2_result(hs, study_r, r1_trades, (), _summary(ids), far)
    assert not bad.passed and bad.metrics["realistic"]["mean"] < lo
    # Set-aside and implausible headlines leave the comparison.
    odd = [*close[1:], _trade(close[0].story_id, 50.0, 150.0, 200.0, 200.0)]
    aside = r2_result(hs, study_r, r1_trades, [ids[4]], _summary(ids), odd)
    assert aside.metrics["realistic"]["n"] == 4
    assert aside.metrics["excluded"] == {
        "set_aside": 1,
        "no_trade": 0,
        "implausible_or_no_spy": 1,
    }
    none = r2_result(hs, {}, [], (), _summary(ids), [])
    assert not none.passed and describe(none) == "not computable"


def test_flatten_hold_buys_at_once_and_sells_at_the_flatten_as_the_d5_rule_fills() -> None:
    day = MON
    published = et(day, 9, 31)  # before the entry window: the gate model still admits it
    st = legacy.reactor_story("AAA:x", "AAA", published)
    bars = session_bars(day, price=50.0)
    facts = TradeFacts("reactor", "gate", "legacy", cost_bps=7.0, rank=10, tech=False)
    ctx = Context(verdicts={("AAA", day): "not_halal"})  # admitted all the same (gate only)
    (out,) = simulate_symbol(
        "AAA",
        [st],
        FlattenHoldFactory({st.story_id: facts}),
        {st.story_id: path(st.story_id, "AAA", [day], [bars])},
        SpyData({day: session_bars(day, price=200.0)}),
        ctx,
        realistic_config(),
    )
    t = out.trade
    assert t is not None and out.terminal_state == "EXITED" and t.exit_reason == "time_stop"
    assert t.entry_active_at == published + timedelta(seconds=63)  # 60 s news lag + ORDER_LAG
    assert t.entry_bar_ts == et(day, 9, 33)
    assert t.exit_decided_at == et(day, 15, 55) and t.exit_bar_ts == et(day, 15, 56)
    assert abs(t.r_net_abn - (-14e-4)) <= EPS
    assert realistic_config().as_config()["fill"] == "gate-market"
    assert realistic_config().feed == SIP_RT and sim_gate.GATE_MARKET_FILL.gate_only


def test_flatten_hold_without_a_fill_by_the_flatten_expires() -> None:
    day = MON
    st = legacy.reactor_story("AAA:y", "AAA", et(day, 15, 0))
    bars = session_bars(day, price=50.0, last=(14, 0))  # no print after 14:00
    facts = TradeFacts("reactor", "gate", "legacy", cost_bps=7.0, rank=10, tech=False)
    (out,) = simulate_symbol(
        "AAA",
        [st],
        FlattenHoldFactory({st.story_id: facts}),
        {st.story_id: path(st.story_id, "AAA", [day], [bars])},
        SpyData({day: session_bars(day, price=200.0)}),
        Context(),
        realistic_config(),
    )
    assert out.trade is None and out.terminal_state == "EXPIRED"
    assert out.reason == "entry_unfilled"


@pytest.mark.parametrize(
    ("published", "entry_bar"),
    [((15, 51, 30), (15, 53)), ((15, 52, 30), (15, 54))],
)
def test_flatten_hold_sells_a_buy_filled_as_the_flatten_passes(
    published: tuple[int, int, int], entry_bar: tuple[int, int]
) -> None:
    """A buy on the 15:54 bar is filled when the 15:55 flatten arrives, its notice a second
    later: it is sold at once on the 15:56 bar, never left to the official close."""
    day = MON
    st = legacy.reactor_story("AAA:z", "AAA", et(day, *published))
    px = {m: 50.0 + 0.1 * m for m in range(50, 60)}
    rising = {(15, m): (p, p + 0.1, p - 0.1, p, 1e3, p) for m, p in px.items()}
    bars = session_bars(day, price=50.0, rows=rising)
    facts = TradeFacts("reactor", "gate", "legacy", cost_bps=7.0, rank=10, tech=False)
    official = Context(daily={("AAA", day): Daily(50.0, 60.0, 49.0, 55.0, 1e6)})  # not a bar
    (out,) = simulate_symbol(
        "AAA",
        [st],
        FlattenHoldFactory({st.story_id: facts}),
        {st.story_id: path(st.story_id, "AAA", [day], [bars])},
        SpyData({day: session_bars(day, price=200.0)}),
        official,
        realistic_config(),
        keep_transitions=True,
    )
    t = out.trade
    assert t is not None and out.terminal_state == "EXITED" and t.exit_reason == "time_stop"
    assert t.entry_bar_ts == et(day, *entry_bar)
    assert t.exit_bar_ts == et(day, 15, 56) and "close_fallback" not in t.flags
    assert t.exit_px != 55.0
    assert [to for _, _, to, _ in out.transitions] == ["ENTERING", "ENTERED", "EXITING", "EXITED"]


# ── G3: SUE ──


def test_decile_spread_follows_the_studys_deciles() -> None:
    signal = np.arange(100, dtype=np.float64)
    rets = np.where(signal >= 90, 0.02, np.where(signal < 10, -0.01, 0.0))
    assert decile_spread(signal, rets) == pytest.approx(0.03)
    assert math.isnan(decile_spread(signal[:9], rets[:9]))
    obs = [(Observation("A", ny(MON, 8), float(s)), 5, float(r)) for s, r in zip(signal, rets)]
    rows = {r.decile: r.mean for r in study.summarise(obs).rows}
    assert rows[10] - rows[1] == pytest.approx(decile_spread(signal, rets))


def test_spread_gives_seeded_cluster_bootstrap_intervals() -> None:
    rng = np.random.default_rng(2)
    signal = rng.normal(size=300)
    rets = 0.01 * signal + rng.normal(0, 0.02, 300)
    clusters = [i // 3 for i in range(300)]
    s = spread(signal, rets, clusters, b=200)
    assert s.d10_d1_ci[0] <= s.d10_d1 <= s.d10_d1_ci[1] and s.d10_d1 > 0
    assert s.ic_ci[0] <= s.ic <= s.ic_ci[1] and s.ic > 0
    assert s.clusters == 100 and s.n == 300 and s.dropped == 0
    assert spread(signal, rets, clusters, b=200) == s
    with pytest.raises(ValueError):
        spread(signal[:5], rets[:5], [0] * 5, b=10)  # one cluster


def _study_result() -> study.StudyResult:
    rows = []
    ic = {}
    for g, scale in (("large (<300)", 1.0), ("mid (300-1000)", 2.0), ("small (1000+)", 0.5)):
        for h in (1, 5, 20, 60):
            ic[(g, h)] = 0.05 * scale if h in (5, 20) else 0.01
            for d in range(1, 11):
                rows.append(study.DecileRow(g, h, d, 100, 0.001 * d * scale, 1.0))
    return study.StudyResult(
        rows, ic, {"large (<300)": 1000, "mid (300-1000)": 900, "small (1000+)": 800}
    )


def test_s0_records_the_table_and_its_deltas_against_the_plan() -> None:
    table = reference_table(_study_result())
    mid = table["mid (300-1000)"]
    assert mid["ic"] == {"1": 0.01, "5": 0.1, "20": 0.1, "60": 0.01}
    assert mid["d10_d1"]["20"] == pytest.approx(0.018)
    deltas = reference_deltas(table)
    assert deltas["mid_20d_d10_d1"]["delta"] == pytest.approx(0.018 - 0.0198)
    assert deltas["mid_20d_d10_d1"]["same_sign"]
    assert deltas["ic"]["large (<300) 5d"]["in_range"]
    assert not deltas["ic"]["mid (300-1000) 20d"]["in_range"]
    assert deltas["ic"]["mid (300-1000) 20d"]["outside_by"] == pytest.approx(0.01)
    sp = Spread(100, 50, 0.01, (0.0, 0.02), 0.05, (0.01, 0.09), 0)
    ok = s0_result(table, {5: sp, 20: sp})
    assert ok.passed and ok.metrics["sigma_c_daily"]["5"]["d10_d1_ci95"] == [0.0, 0.02]
    assert "mid 20d D10-D1" in describe(ok)
    assert not s0_result({}, {5: sp, 20: sp}).passed  # no table
    assert not s0_result(table, None, "undefined on 30 of 2000 resamples").passed
    no_mid = {k: v for k, v in table.items() if k != "mid (300-1000)"}
    assert not s0_result(no_mid, {5: sp, 20: sp}).passed
    nan = replace(sp, ic=math.nan)
    assert not s0_result(table, {5: sp, 20: nan}).passed


def _bars(rng: np.random.Generator, sessions: list[date], symbols: list[str]) -> study.Bars:
    opens: dict[str, dict[date, float]] = {}
    closes: dict[str, dict[date, float]] = {}
    for s in symbols:
        p = float(rng.uniform(20, 200))
        for d in sessions:
            o = p * float(np.exp(rng.normal(0, 0.01)))
            p = o * float(np.exp(rng.normal(0, 0.015)))
            opens.setdefault(s, {})[d] = o
            closes.setdefault(s, {})[d] = p
    return study.Bars(sessions, opens, closes)


def _calendar(lo: date, hi: date) -> list[date]:
    return [
        lo + timedelta(days=i)
        for i in range((hi - lo).days + 1)
        if is_trading_day(lo + timedelta(days=i))
    ]


def _evaluate(
    bars: study.Bars, obs: list[Observation], rank: int | None
) -> list[tuple[Observation, int, float]]:
    """``study.evaluate``'s loop on in-memory bars (one rank for every observation)."""
    out = []
    for o in obs:
        entry = study.entry_point(o.published_at, bars.sessions)
        if entry is None:
            continue
        tagged = Observation(
            o.symbol,
            o.published_at,
            o.signal,
            {"year": o.published_at.year, "bucket": study.bucket_of(rank)},
        )
        for h in study.HORIZONS:
            r = study.outcome(bars, o.symbol, entry, h, study.cost_bps(rank))
            if r is not None:
                out.append((tagged, h, r))
    return out


def test_s1_passes_daily_bars_through_the_simulator_and_fails_on_any_difference() -> None:
    rng = np.random.default_rng(5)
    sessions = _calendar(date(2016, 1, 4), date(2016, 8, 31))
    bars = _bars(rng, sessions, ["AAA", "BBB", "SPY"])
    bars.close["AAA"][date(2016, 3, 11)] = bars.open["AAA"][date(2016, 3, 11)] * 20  # implausible
    obs = [
        Observation(
            ("AAA", "BBB")[n % 2],
            datetime(2016, 1, 5, 12, tzinfo=UTC)
            + timedelta(seconds=float(rng.uniform(0, 200 * 86400))),
            float(n),
        )
        for n in range(80)
    ]
    obs.append(Observation("AAA", ny(date(2016, 3, 11), 8), 99.0))  # its 1-day exit is implausible
    src = legacy.DailyBarSource(bars)
    rank_of = {(o.symbol, o.published_at): 400 for o in obs}
    got, look, skips = daily_mode(src, obs, rank_of)
    want = _evaluate(bars, obs, 400)
    assert look == 0 and skips["implausible"] >= 1 and skips["no_daily"] > 0
    ok = s1_result(want, got, lookahead=look, early_close=3, skips=skips)
    assert ok.passed, ok.metrics
    assert ok.metrics["max_abs_diff"] <= 1e-12 and ok.metrics["early_close_excluded"] == 3
    assert (
        ok.metrics["deciles"]["all"]["identical"] and ok.metrics["deciles"]["bucket"]["identical"]
    )
    off = [(o, h, r + (1e-9 if i == 7 else 0.0)) for i, (o, h, r) in enumerate(got)]
    bad = s1_result(want, off, lookahead=0, early_close=0)
    assert not bad.passed and bad.metrics["over_tolerance"] == 1
    missing = s1_result(want, got[1:], lookahead=0, early_close=0)
    assert not missing.passed and missing.metrics["only_study"] == 1
    extra = s1_result(want[1:], got, lookahead=0, early_close=0)
    assert not extra.passed and extra.metrics["only_sim"] == 1
    gone = s1_result(
        want,
        got,
        lookahead=0,
        early_close=0,
        excluded=[(want[0][0].symbol, want[0][0].published_at, want[0][0].signal)],
    )
    assert gone.passed  # an excluded observation leaves both sides
    assert not s1_result([], [], lookahead=0, early_close=0).passed


def test_s1_excludes_close_entries_published_after_an_early_close() -> None:
    sessions = _calendar(date(2016, 11, 1), date(2017, 3, 31))
    bars = _bars(np.random.default_rng(6), sessions, ["AAA", "SPY"])
    half = date(2016, 11, 25)
    obs = [Observation("AAA", ny(half, 14), 1.0), Observation("AAA", ny(half, 12), 2.0)]
    got, look, _ = daily_mode(
        legacy.DailyBarSource(bars),
        obs,
        dict.fromkeys(((o.symbol, o.published_at) for o in obs), 10),
    )
    assert look == 1 and {o.signal for o, _, _ in got} == {2.0}


def _flat(day: date, price: float, *, minutes: list[tuple[int, int]] | None = None) -> BarArrays:
    """Bars at a few minutes of ``day``, flat at ``price``."""
    keep = minutes or [(9, 30), (10, 0), (12, 0), (15, 55), (15, 56), (15, 59)]
    ts = np.asarray([epoch(day, h, m) for h, m in keep], dtype=np.int64)
    p = np.full(len(keep), price)
    return BarArrays(ts, p, p.copy(), p.copy(), p.copy(), np.full(len(keep), 1e3), p.copy())


class _Daily:
    def __init__(self, closes: dict[tuple[str, date], float]) -> None:
        self.closes = closes

    def __call__(self, symbol: str, day: date) -> Daily | None:
        c = self.closes.get((symbol, day))
        return Daily(c, c, c, c, 1e6) if c is not None else None


def test_study_clock_returns_read_the_first_open_or_the_last_close_with_a_factors() -> None:
    bars = {
        ("AAA", MON): _flat(MON, 50.0, minutes=[(9, 31), (15, 59)]),
        ("AAA", WED): _flat(WED, 25.5),
        ("SPY", MON): _flat(MON, 200.0),
        ("SPY", WED): _flat(WED, 202.0),
    }
    adj = {("AAA", MON): 0.5, ("AAA", WED): 1.0, ("SPY", MON): 1.0, ("SPY", WED): 1.0}
    view = MinuteReturns(bars, lambda s, d: adj.get((s, d)), _Daily({}))
    r, why, late = study_clock_return(view, "AAA", MON, "open", WED, 7.0)
    assert why == "" and late  # the first print is 09:31
    assert r == pytest.approx((25.5 * 1.0) / (50.0 * 0.5) - 1 - (202 / 200 - 1) - 14e-4)
    r2, _, late2 = study_clock_return(view, "AAA", MON, "close", WED, 0.0)
    assert not late2 and r2 == pytest.approx(25.5 / 25.0 - 1 - 0.01)
    assert study_clock_return(view, "BBB", MON, "open", WED, 0.0)[:2] == (None, "no_bars")
    no_spy = MinuteReturns(
        {k: v for k, v in bars.items() if k != ("SPY", WED)}, view.adj, view.daily
    )
    assert study_clock_return(no_spy, "AAA", MON, "open", WED, 0.0)[:2] == (None, "no_spy")
    no_adj = MinuteReturns(bars, lambda s, d: None if s == "AAA" else 1.0, view.daily)
    assert study_clock_return(no_adj, "AAA", MON, "open", WED, 0.0)[:2] == (None, "no_adj")


def test_realistic_returns_use_the_market_rule_and_the_flatten() -> None:
    rows = {(10, 11): (50.0, 50.4, 49.9, 50.2, 1e3, 50.6)}  # VWAP above the high: clamped
    stock_in = session_bars(MON, price=50.0, rows=rows)
    stock_out = session_bars(WED, price=52.0, rows={(15, 56): (52.0, 52.5, 51.5, 52.1, 1e3, 52.2)})
    spy_in, spy_out = session_bars(MON, price=200.0), session_bars(WED, price=201.0)
    bars = {
        ("AAA", MON): stock_in,
        ("AAA", WED): stock_out,
        ("SPY", MON): spy_in,
        ("SPY", WED): spy_out,
    }
    view = MinuteReturns(bars, lambda s, d: 1.0, _Daily({}))
    published = et(MON, 10, 0, 30)  # + 600 s: active 10:10:30, the 10:11 bar fills
    r, why = realistic_return(view, "AAA", published, MON, WED, 7.0)
    assert why == "" and r == pytest.approx(52.2 / 50.4 - 1 - (201 / 200 - 1) - 14e-4)
    early = realistic_return(view, "AAA", et(MON, 7, 0), MON, WED, 0.0)[0]  # 09:30: first bar
    assert early == pytest.approx(52.2 / 50.0 - 1 - 0.005)
    # No bar after the flatten + ORDER_LAG: the official close, else the last trade.
    cut = {**bars, ("AAA", WED): session_bars(WED, price=52.0, last=(15, 50))}
    fallback = MinuteReturns(
        cut, lambda s, d: 1.0, _Daily({("AAA", WED): 53.0, ("SPY", WED): 201.5})
    )
    r_close, _ = realistic_return(fallback, "AAA", published, MON, WED, 0.0)
    assert r_close == pytest.approx(53.0 / 50.4 - 1 - (201.5 / 200 - 1))
    last = MinuteReturns(cut, lambda s, d: 1.0, _Daily({}))
    r_last, _ = realistic_return(last, "AAA", published, MON, WED, 0.0)
    assert r_last == pytest.approx(52.0 / 50.4 - 1 - (201.0 / 200 - 1))
    late = realistic_return(view, "AAA", et(MON, 15, 55), MON, WED, 0.0)
    assert late == (None, "entry_unfilled")  # + 600 s is after the close
    assert realistic_return(view, "BBB", published, MON, WED, 0.0) == (None, "no_bars")
    empty = {**bars, ("AAA", WED): BarArrays.empty()}
    assert realistic_return(
        MinuteReturns(empty, view.adj, _Daily({})), "AAA", published, MON, WED, 0.0
    ) == (None, "no_exit")
    no_spy = {**bars, ("SPY", MON): BarArrays.empty()}
    assert realistic_return(
        MinuteReturns(no_spy, view.adj, _Daily({})), "AAA", published, MON, WED, 0.0
    ) == (None, "no_spy")


def test_calibration_compares_the_minute_and_daily_returns_of_each_pair() -> None:
    sessions = _calendar(date(2016, 2, 1), date(2016, 3, 31))
    day = date(2016, 2, 8)
    exit_day = sessions[sessions.index(day) + 4]
    daily = study.Bars(
        sessions,
        {"AAA": {day: 40.0}, "SPY": {day: 200.0}},
        {"AAA": {exit_day: 42.0}, "SPY": {exit_day: 202.0}},
    )
    bars = {
        ("AAA", day): _flat(day, 40.2),
        ("AAA", exit_day): _flat(exit_day, 42.0),
        ("SPY", day): _flat(day, 200.0),
        ("SPY", exit_day): _flat(exit_day, 202.0),
    }
    view = MinuteReturns(bars, lambda s, d: 1.0, _Daily({}))
    diffs, counts = calibration(view, daily, [("AAA", day), ("BBB", day)])
    assert diffs == [pytest.approx((42 / 40.2 - 1) - (42 / 40 - 1))] and counts["no_bars"] == 1
    ok = calib_result(diffs * 20, 20, counts)
    assert ok.passed and ok.metrics["p99_cal"] == pytest.approx(abs(diffs[0]))
    assert "p99_cal" in describe(ok)
    assert not calib_result(diffs, 2, counts).passed  # half the pairs: under 90%
    assert not calib_result([], 0, Counter()).passed


def _paired(n: int, d: float, *, h: int = 5, noise: float = 0.0, seed: int = 0) -> list[Paired]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        daily = float(rng.normal(0, 0.02))
        out.append(
            Paired(
                ("A", ny(MON, 8), float(i)),
                h,
                float(i),
                MON + timedelta(days=7 * (i % 20)),
                daily,
                daily + d + float(rng.normal(0, noise)),
            )
        )
    return out


def test_s2_passes_minute_mode_that_agrees_with_daily_mode() -> None:
    pairs = _paired(200, 0.0, noise=0.0002) + _paired(200, 0.0, h=20, noise=0.0002, seed=1)
    ok = s2_result(pairs, 200, 0.002, {"no_bars": 0})
    assert ok.passed, ok.metrics
    five = ok.metrics["horizons"]["5"]
    assert five["tost"] and five["p99_limit"] == 0.01 and five["coverage"] == 1.0
    assert ok.metrics["horizons"]["20"]["passed"] and "5d ok" in describe(ok)
    assert s2_result(pairs, 200, 0.01, {}).metrics["horizons"]["5"]["p99_limit"] == pytest.approx(
        0.015
    )


@pytest.mark.parametrize(
    ("shift", "noise", "events", "why"),
    [
        (0.002, 0.0002, 200, "tost"),  # a 0.2% bias: outside ±0.10%
        (0.0, 0.004, 200, "median"),  # |d| typically 0.27%
        (0.0, 0.0, 400, "coverage"),  # half the events paired
    ],
)
def test_s2_fails_minute_mode_that_does_not(
    shift: float, noise: float, events: int, why: str
) -> None:
    pairs = _paired(200, shift, noise=noise) + _paired(200, shift, h=20, noise=noise, seed=1)
    bad = s2_result(pairs, events, 0.002, {})
    five = bad.metrics["horizons"]["5"]
    assert not bad.passed and not five["passed"]
    if why == "tost":
        assert not five["tost"]
    elif why == "median":
        assert five["median_abs_d"] > 0.001
    else:
        assert five["coverage"] == 0.5


def test_s2_fails_a_tail_beyond_its_p99_bound() -> None:
    pairs = _paired(200, 0.0) + _paired(200, 0.0, h=20, seed=1)
    tail = [
        replace(p, minute=p.daily + (0.05 if i % 25 == 0 else 0.0)) for i, p in enumerate(pairs)
    ]
    bad = s2_result(tail, 200, 0.002, {})
    assert not bad.passed and bad.metrics["horizons"]["5"]["p99_abs_d"] > 0.01
    assert not s2_result(pairs[:1], 200, 0.002, {}).passed  # one cluster shows nothing


def _sue_pairs(real_shift: float) -> list[Paired]:
    out = []
    rng = np.random.default_rng(7)
    for h in (5, 20):
        for i in range(400):
            signal = float(rng.normal())
            daily = 0.004 * signal + float(rng.normal(0, 0.01))
            out.append(
                Paired(
                    ("A", ny(MON, 8), signal),
                    h,
                    signal,
                    MON + timedelta(days=i % 60),
                    daily,
                    daily + real_shift * signal,
                )
            )
    return out


def test_s3_passes_realistic_fills_inside_the_daily_interval_and_fails_outside() -> None:
    ok = s3_result(_sue_pairs(0.0), 400, {}, b=200)
    assert ok.passed, ok.metrics
    five = ok.metrics["horizons"]["5"]
    assert five["n"] == 400 and five["daily"]["n"] == 400
    assert describe(ok) == "5d ok (n 400); 20d ok (n 400)"
    assert five["inside"] == {"d10_d1": True, "ic": True}
    assert five["same_sign"] == {"d10_d1": True, "ic": True}
    assert five["realistic"]["d10_d1"] == pytest.approx(five["daily"]["d10_d1"])
    # Realistic fills that turn the signal around: outside, and of the other sign.
    bad = s3_result(_sue_pairs(-0.012), 400, {}, b=200)
    assert not bad.passed
    assert bad.metrics["horizons"]["20"]["same_sign"]["d10_d1"] is False
    assert "FAIL" in describe(bad)
    assert not s3_result(_sue_pairs(0.0), 1000, {}, b=200).passed  # 40% coverage
    assert not s3_result(_sue_pairs(0.0)[:5], 400, {}, b=50).passed


def test_describe_has_a_line_for_every_gate() -> None:
    for gate in GATE_IDS:
        assert isinstance(describe(GateResult(gate, False, {})), str)
    assert describe(GateResult("r0", False, {"error": "R0 asked the market for AAA"})).startswith(
        "R0"
    )


def test_one_session_with_holes_still_reproduces_legacy_after_a_gap() -> None:
    """The reactor's legacy fill and the realistic one agree where no rule bites."""
    rng = np.random.default_rng(3)
    stock = random_session(rng, MON, 40.0, holes=False)
    spy = random_session(rng, MON, 200.0, holes=False)
    st = legacy.reactor_story("AAA:z", "AAA", et(MON, 11, 0, 10))
    facts = {st.story_id: TradeFacts("r", "g", "l", cost_bps=7.0, rank=1, tech=False)}
    runs = []
    for cfg, fac in (
        (legacy.reactor_config(), legacy.HoldFactory(1, facts)),
        (realistic_config(), FlattenHoldFactory(facts)),
    ):
        (out,) = simulate_symbol(
            "AAA",
            [st],
            fac,
            {st.story_id: path(st.story_id, "AAA", [MON], [stock])},
            SpyData({MON: spy}),
            Context(),
            cfg,
        )
        assert out.trade is not None
        runs.append(out.trade)
    old, new = runs
    assert old.entry_bar_ts == new.entry_bar_ts  # 11:01:10 and 11:01:13: the 11:02 bar
    e, x = decomposition(old, new)
    assert abs(e + x - (new.r_net_abn - old.r_net_abn)) <= 1e-12
