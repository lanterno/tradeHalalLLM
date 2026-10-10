"""Look-ahead invariance: nothing a playbook does by T may depend on what is unknown at T.

For random decision times T, every bar visible after T is scaled by U(0.5, 1.5)
(one factor per bar for o/h/l/c/vw, another for volume), every item with
``available_at > T`` is deleted, and the official open and close of every
session from S on are replaced with noise (A-factors kept). Replaying must
give the same intents and transitions at or before T, and the same set of
stories started by T.

One exception is not look-ahead: a fill's price is known when the fill is
acknowledged (the filling bar's end + 1 s), before the bar itself reaches a
delayed data feed, so a bar that filled an order acknowledged by T is not
perturbed.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from halabot.playbooks.clock import SIP_DELAYED, SIP_RT, FeedProfile
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
from halal_trader.data.minutes import BarArrays
from tests.halabot.playbooks._support import FACTS, Context, Daily, Story, Toy
from tests.halabot.playbooks._synth import Market, random_market


def _perturb(bars: BarArrays, after: int, keep: set[int], rng: np.random.Generator) -> BarArrays:
    """Scale every bar whose visible time (epoch s) is after ``after``, except ``keep``."""
    hit = np.asarray([int(t) > after and int(t) not in keep for t in bars.ts], dtype=bool)
    f = np.where(hit, rng.uniform(0.5, 1.5, len(bars)), 1.0)
    g = np.where(hit, rng.uniform(0.5, 1.5, len(bars)), 1.0)
    return BarArrays(
        bars.ts, bars.o * f, bars.h * f, bars.l * f, bars.c * f, bars.v * g, bars.vw * f
    )


def _visible_cut(T: datetime, feed: FeedProfile) -> int:
    """Bars starting after this epoch second are visible after T (ts + 60 + lag > T)."""
    return int(T.timestamp()) - 60 - int(feed.bar_lag.total_seconds())


def _fill_bars_known_by(outcomes: list[StoryOutcome], T: datetime) -> set[int]:
    keep: set[int] = set()
    for o in outcomes:
        bars = [o.entry_bar_ts]
        if o.trade is not None:
            bars.append(o.trade.exit_bar_ts)
        for b in bars:
            if b is not None and b + timedelta(seconds=61) <= T:
                keep.add(int(b.timestamp()))
    return keep


class _NoisyContext(Context):
    def __init__(self, base: Context, rng: np.random.Generator) -> None:
        super().__init__()
        self._base = base
        self._rng = rng
        self._cache: dict[tuple[str, object], Daily | None] = {}

    def adj(self, symbol, day):  # type: ignore[no-untyped-def]
        return self._base.adj(symbol, day)

    def screen_verdict(self, symbol, day):  # type: ignore[no-untyped-def]
        return self._base.screen_verdict(symbol, day)

    def daily(self, symbol, day):  # type: ignore[no-untyped-def]
        key = (symbol, day)
        if key not in self._cache:
            dp = self._base.daily(symbol, day)
            if dp is not None:
                o, c = (float(x) for x in self._rng.uniform(1, 500, 2))
                dp = Daily(o, max(o, c), min(o, c), c, dp.volume)
            self._cache[key] = dp
        return self._cache[key]


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


def _upto(o: StoryOutcome, T: datetime):  # type: ignore[no-untyped-def]
    return (
        [x for x in o.intents if x[0] <= T],
        [x for x in o.transitions if x[0] <= T],
    )


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

    lo = min(o.start_at for o in base if o.start_at is not None)
    hi = lo + timedelta(days=4)
    checked = 0
    for _ in range(5):
        T = lo + timedelta(seconds=float(rng.uniform(0, (hi - lo).total_seconds())))
        T = T.replace(microsecond=0)
        cut = _visible_cut(T, feed)
        keep = _fill_bars_known_by(base, T)
        paths = {
            sid: replace(p, bars=tuple(_perturb(b, cut, keep, rng) for b in p.bars))
            for sid, p in market.paths.items()
        }
        spy = SpyData({d: _perturb(b, cut, set(), rng) for d, b in market.spy.days.items()})
        stories = [s.before(T) for s in market.stories]
        ctx = _NoisyContext(market.ctx, rng)
        again = _run(market, stories, paths, spy, ctx, cfg, target)

        started = {o.story_id for o in base if o.start_at is not None and o.start_at <= T}
        started_again = {o.story_id for o in again if o.start_at is not None and o.start_at <= T}
        assert started == started_again
        by_id = {o.story_id: o for o in again}
        for o in base:
            if o.story_id in started:
                assert _upto(o, T) == _upto(by_id[o.story_id], T), (o.story_id, T)
                checked += 1
    assert checked > 0


def test_the_harness_detects_a_peek() -> None:
    """A playbook that reads a bar it cannot see yet fails the invariance check."""
    market = random_market(5, 8, sessions=1)
    cfg = SimConfig()

    class Peeker(Toy):
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

    def run(st: Story, paths: dict[str, PathData]) -> StoryOutcome:
        (out,) = simulate_symbol(
            st.symbol,
            [st],
            lambda s: Peeker(s, target=1.0),
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
        cut = _visible_cut(T, cfg.feed)
        p = market.paths[st.story_id]
        paths = {st.story_id: replace(p, bars=tuple(_perturb(b, cut, set(), rng) for b in p.bars))}
        differs += _upto(o, T) != _upto(run(st, paths), T)
    assert differs > 0
