"""Look-ahead invariance: nothing a playbook does by T may depend on what is unknown at T.

The harness is ``halabot.playbooks.lookahead`` (the Phase 0 gate G1 runs it on
real stories). For random decision times T, every bar visible after T is
scaled by U(0.5, 1.5) (one factor per bar for o/h/l/c/vw, another for
volume), every item with ``available_at > T`` is deleted, and the official
open and close of every session from S on are replaced with noise (A-factors
kept). Replaying must give the same intents and transitions at or before T,
and the same set of stories started by T.

One exception is not look-ahead: a fill's price is known when the fill is
acknowledged (the filling bar's end + 1 s), before the bar itself reaches a
delayed data feed, so a bar that filled an order acknowledged by T is not
perturbed.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from halabot.playbooks.clock import SIP_DELAYED, SIP_RT, FeedProfile
from halabot.playbooks.lookahead import (
    NoisyContext,
    Probe,
    check_symbol,
    fill_bar_violations,
    fill_bars_known_by,
    perturb,
    perturb_path,
    started_by,
    upto,
    visible_cut,
)
from halabot.playbooks.records import StoryOutcome
from halabot.playbooks.sim import simulate_symbol
from halabot.playbooks.types import (
    BarIn,
    PathData,
    SimConfig,
    SpyData,
    Submit,
    Transition,
)
from tests.halabot.playbooks._support import (
    FACTS,
    MON,
    TUE,
    Context,
    Daily,
    Story,
    Toy,
    downgrade,
    et,
    path,
    run_one,
    session_bars,
)
from tests.halabot.playbooks._synth import Market, random_market


def _run(market: Market, stories: list[Story], paths, spy, ctx, cfg, target: float):  # type: ignore[no-untyped-def]
    out: list[StoryOutcome] = []
    by_symbol: dict[str, list[Story]] = {}
    for s in stories:
        by_symbol.setdefault(s.symbol, []).append(s)
    for symbol, group in sorted(by_symbol.items()):
        out += simulate_symbol(
            symbol,
            group,
            lambda s: Toy(s, sessions=market.sessions, target=target),
            paths,
            spy,
            ctx,
            cfg,
            keep_transitions=True,
        )
    return out


@pytest.mark.parametrize(
    ("seed", "feed"), [(1, SIP_RT), (2, SIP_RT), (3, SIP_DELAYED), (4, SIP_RT)]
)
def test_nothing_before_T_depends_on_what_comes_after(seed: int, feed: FeedProfile) -> None:
    market = random_market(seed, 12, sessions=3 if seed % 2 else 1)
    cfg = SimConfig(feed=feed)
    rng = np.random.default_rng(1000 + seed)
    target = float(rng.uniform(0.002, 0.02))
    base = _run(market, market.stories, market.paths, market.spy, market.ctx, cfg, target)
    entered = sum(o.entered for o in base)
    assert entered > 0  # the harness exercises fills, not only watching
    assert fill_bar_violations(base) == []

    lo = min(o.start_at for o in base if o.start_at is not None)
    hi = lo + timedelta(days=4)
    checked = 0
    for _ in range(5):
        T = lo + timedelta(seconds=float(rng.uniform(0, (hi - lo).total_seconds())))
        T = T.replace(microsecond=0)
        cut = visible_cut(T, feed)
        keep = fill_bars_known_by(base, T)
        paths = {sid: perturb_path(p, cut, keep, rng) for sid, p in market.paths.items()}
        spy = SpyData({d: perturb(b, cut, set(), rng) for d, b in market.spy.days.items()})
        stories = [s.before(T) for s in market.stories]
        ctx = NoisyContext(market.ctx, rng)
        again = _run(market, stories, paths, spy, ctx, cfg, target)

        started = started_by(base, T)
        assert started == started_by(again, T)
        by_id = {o.story_id: o for o in again}
        for o in base:
            if o.story_id in started:
                assert upto(o, T) == upto(by_id[o.story_id], T), (o.story_id, T)
                checked += 1
    assert checked > 0


class _Peeker(Toy):
    """Enters when the close 30 bars ahead (not visible yet) is higher than the last."""

    def on(self, ev, ctx):  # type: ignore[no-untyped-def]
        if (
            self.state() == "WATCHING"
            and isinstance(ev, BarIn)
            and ev.symbol == ctx.story.symbol
            and ev.at >= ctx.session.entry_start
        ):
            full = ctx.market._bars  # reaches past the visible view, on purpose
            n = len(ctx.market.bars(ev.symbol))
            if n + 30 < len(full) and full.c[n + 30] > full.c[n - 1]:
                self._state = "ENTERING"
                return [Transition("ENTERING", "peek"), Submit("buy", facts=FACTS)]
            return []
        return super().on(ev, ctx)


def test_the_harness_detects_a_peek() -> None:
    """A playbook that reads a bar it cannot see yet fails the invariance check."""
    market = random_market(5, 8, sessions=1)
    cfg = SimConfig()

    def run(st: Story, paths: dict[str, PathData]) -> StoryOutcome:
        (out,) = simulate_symbol(
            st.symbol,
            [st],
            lambda s: _Peeker(s, target=1.0),
            paths,
            market.spy,
            market.ctx,
            cfg,
            keep_transitions=True,
        )
        return out

    rng = np.random.default_rng(9)
    differs = 0
    for st in market.stories:
        o = run(st, market.paths)
        if o.start_at is None or o.entry_decided_at is None:
            continue
        T = o.entry_decided_at  # the peeking decision itself
        cut = visible_cut(T, cfg.feed)
        p = market.paths[st.story_id]
        paths = {st.story_id: perturb_path(p, cut, set(), rng)}
        differs += upto(o, T) != upto(run(st, paths), T)
    assert differs > 0


# ── the harness, piece by piece ──


def test_perturb_scales_only_the_bars_visible_after_the_cut_and_spares_kept_fills() -> None:
    bars = session_bars(MON, price=50.0)
    rng = np.random.default_rng(1)
    t = et(MON, 10, 0)
    cut = visible_cut(t, SIP_RT)
    assert cut == int(t.timestamp()) - 65  # ts + 60 s + 5 s lag > T
    keep = {int(et(MON, 11, 0).timestamp())}
    moved = perturb(bars, cut, keep, rng)
    after = bars.ts > cut
    assert np.array_equal(moved.o[~after], bars.o[~after])
    assert np.array_equal(moved.v[~after], bars.v[~after])
    hit = after & ~np.isin(bars.ts, list(keep))
    assert np.all(moved.o[hit] != bars.o[hit]) and np.all(moved.v[hit] != bars.v[hit])
    assert np.all((moved.c[hit] / bars.c[hit] >= 0.5) & (moved.c[hit] / bars.c[hit] <= 1.5))
    # o/h/l/c/vw share one factor per bar; the volume has its own.
    assert np.allclose(moved.o[hit] / bars.o[hit], moved.vw[hit] / bars.vw[hit])
    k = np.isin(bars.ts, list(keep))
    assert np.array_equal(moved.c[k], bars.c[k])  # the exempt fill bar
    # Two draws of len(bars) whatever is hit: the generator moves the same way.
    a, b = np.random.default_rng(2), np.random.default_rng(2)
    perturb(bars, 10**12, set(), a)
    perturb(bars, 0, set(), b)
    assert a.uniform() == b.uniform()


def test_perturb_path_moves_the_spare_session_too() -> None:
    p = path(
        "A:1",
        "A",
        [MON],
        [session_bars(MON)],
        spare=TUE,
        spare_bars=session_bars(TUE),
    )
    moved = perturb_path(p, int(et(MON, 12, 0).timestamp()), set(), np.random.default_rng(3))
    assert moved.spare_bars is not None and p.spare_bars is not None
    assert not np.array_equal(moved.spare_bars.c, p.spare_bars.c)
    assert np.array_equal(moved.bars[0].c[:100], p.bars[0].c[:100])  # before noon: kept
    assert replace(p, spare_bars=None).spare_bars is None
    assert (
        perturb_path(replace(p, spare_bars=None), 0, set(), np.random.default_rng(3)).spare_bars
        is None
    )


def test_the_noisy_context_keeps_a_factors_and_the_sessions_before_since() -> None:
    base = Context(
        adj={("A", TUE): 0.5},
        verdicts={("A", TUE): "not_halal"},
        daily={("A", MON): Daily(10, 11, 9, 10.5, 1e6), ("A", TUE): Daily(20, 21, 19, 20.5, 1e6)},
        sessions=[MON, TUE],
    )
    ctx = NoisyContext(base, np.random.default_rng(4), since=TUE)
    assert ctx.adj("A", TUE) == 0.5 and ctx.screen_verdict("A", TUE) == "not_halal"
    assert list(ctx.sessions) == [MON, TUE]
    mon = ctx.daily("A", MON)
    assert mon is not None and (mon.open, mon.close) == (10, 10.5)  # known by then: kept
    tue = ctx.daily("A", TUE)
    assert tue is not None and (tue.open, tue.close) != (20, 20.5)
    assert 1 <= tue.open <= 500 and tue.low <= min(tue.open, tue.close)
    assert ctx.daily("A", TUE) is tue  # drawn once
    assert ctx.daily("A", date(2016, 3, 9)) is None
    every = NoisyContext(base, np.random.default_rng(4))  # since None: every session
    noisy_mon = every.daily("A", MON)
    assert noisy_mon is not None and noisy_mon.close != 10.5


def test_fill_bars_are_exempt_once_acknowledged_and_never_start_before_active() -> None:
    st = Story("A", MON, [downgrade(MON)])
    out = run_one(st, path(st.story_id, "A", [MON], [session_bars(MON)]), toy={"target": 0.0})
    t = out.trade
    assert t is not None and out.entry_bar_ts is not None
    ack = out.entry_bar_ts + timedelta(seconds=61)
    assert fill_bars_known_by([out], ack - timedelta(seconds=1)) == set()
    assert int(out.entry_bar_ts.timestamp()) in fill_bars_known_by([out], ack)
    assert fill_bar_violations([out]) == []
    early = replace(out, trade=replace(t, entry_bar_ts=t.entry_active_at - timedelta(minutes=1)))
    assert fill_bar_violations([early]) == [st.story_id]


def test_upto_compares_exact_reprs_and_nan_equals_nan() -> None:
    st = Story("A", MON, [downgrade(MON)])
    o = run_one(st, path(st.story_id, "A", [MON], [session_bars(MON)]))
    T = et(MON, 12, 0)
    intents, transitions = upto(o, T)
    assert intents and transitions
    assert all(isinstance(x, str) for x in intents)
    nan_facts = replace(FACTS, beta=math.nan)
    a = replace(o, intents=((T, Submit("buy", facts=nan_facts)),))
    b = replace(o, intents=((T, Submit("buy", facts=replace(FACTS, beta=float("nan")))),))
    assert upto(a, T) == upto(b, T)  # == would say NaN != NaN
    assert upto(a, T - timedelta(seconds=1))[0] == ()


def test_check_symbol_passes_the_toy_and_catches_a_peek() -> None:
    market = random_market(11, 6, sessions=1)
    cfg = SimConfig()
    rng = np.random.default_rng(12)
    by_symbol: dict[str, list[Story]] = {}
    for s in market.stories:
        by_symbol.setdefault(s.symbol, []).append(s)
    checked = 0
    for symbol, stories in sorted(by_symbol.items()):
        probes = [
            Probe(et(s.session, 9, 0) + timedelta(minutes=int(m)), since=s.session)
            for s in stories
            for m in rng.integers(0, 420, 3)
        ]
        out = check_symbol(
            symbol,
            stories,
            lambda s: Toy(s, target=0.005),
            market.paths,
            market.spy,
            market.ctx,
            cfg,
            probes,
            rng,
        )
        assert out.mismatches == [] and out.probes == len(probes)
        checked += out.checked
    assert checked > 0
    caught = 0
    for symbol, stories in sorted(by_symbol.items()):
        base = simulate_symbol(
            symbol,
            stories,
            lambda s: _Peeker(s, target=1.0),
            market.paths,
            market.spy,
            market.ctx,
            cfg,
            keep_transitions=True,
        )
        probes = [Probe(o.entry_decided_at) for o in base if o.entry_decided_at is not None]
        out = check_symbol(
            symbol,
            stories,
            lambda s: _Peeker(s, target=1.0),
            market.paths,
            market.spy,
            market.ctx,
            cfg,
            probes,
            rng,
            base=base,
        )
        caught += len(out.mismatches)
        assert all(m.what in ("intents", "transitions", "started") for m in out.mismatches)
    assert caught > 0
