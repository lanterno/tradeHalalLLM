"""The bounce on random markets: look-ahead invariance, determinism, and the full driver.

**Look-ahead** (spec §E.1), with the simulator's own harness
(``test_lookahead``): for each story, at random times T inside its path
and at every moment it decided something (trigger, arming, entry, exit),
every bar visible after T is scaled by U(0.5, 1.5) (o/h/l/c/vw by one
factor, volume by another), every item usable after T is deleted, and the
official opens and closes of the sessions are replaced with noise (A-factors
kept). Replaying the symbol must give the same intents and transitions at
or before T, and the same stories started by T. A bar that filled an order
acknowledged by T keeps its prices: the fill price is known 4 s before the
bar reaches the feed (the harness's exemption).

The markets (``_bounce.bounce_market``) put a 2-10% dip with a random
recovery at each story's news, so the runs reach every terminal state:
targets, stops, aborts, compliance and time stops, cutoffs, vetoes, market
breaks and dismissals.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.playbooks.bounce import BounceFactory, BounceParams
from halabot.playbooks.clock import SIP_DELAYED, SIP_RT, FeedProfile
from halabot.playbooks.interfaces import StoryView
from halabot.playbooks.loader import Window, WindowUnlock
from halabot.playbooks.records import MemorySink, StoryOutcome, outcomes_sha256
from halabot.playbooks.sim import run, simulate_many, simulate_symbol
from halabot.playbooks.types import PathData, SimConfig, SpyData
from halal_trader.data.minutes import BarArrays
from tests.halabot.playbooks._bounce import (
    TStory,
    bars,
    bounce_market,
    eligibility,
    pre_event,
    spy,
    tstory,
)
from tests.halabot.playbooks._seed import seed_calendar, seed_market
from tests.halabot.playbooks._support import MON, THU, TUE, WED, Context, Story, et, path
from tests.halabot.playbooks.test_bounce import (
    IN_LEVELS,
    IN_ROWS,
    IN_SPY,
    SPY_ROWS,
    A,
    base_bars,
    base_story,
)
from tests.halabot.playbooks.test_lookahead import (
    _fill_bars_known_by,
    _NoisyContext,
    _perturb,
    _upto,
    _visible_cut,
)

SYMBOLS = 16
RUN_ID = "00000000-0000-0000-0000-0000000000b2"
SECOND = timedelta(seconds=1)


def _moments(o: StoryOutcome) -> set[datetime]:
    """Every instant the story decided something, and the second before it."""
    out = {at for at, *_ in o.transitions}
    if o.trade is not None:
        out |= {o.trade.entry_decided_at, o.trade.exit_decided_at}
    return out | {t - SECOND for t in out}


def _invariant(
    symbol: str,
    stories: Sequence[StoryView],
    paths: Mapping[str, PathData],
    spy: SpyData,
    ctx: Context,
    fac: BounceFactory,
    cfg: SimConfig,
    rng: np.random.Generator,
    *,
    random_t: int = 5,
) -> tuple[int, list[StoryOutcome]]:
    """Replay ``symbol`` at many T with everything after T perturbed; returns (checks, base).

    T: ``random_t`` random instants inside each started story's path, every
    instant it decided something and the second before, and the second
    before each item becomes usable (a peek at an item would act then).
    """
    base = simulate_symbol(symbol, stories, fac, paths, spy, ctx, cfg, keep_transitions=True)
    times: set[datetime] = set()
    for o in base:
        if o.start_at is None:
            continue
        span = timedelta(days=fac.path_sessions + 2).total_seconds()
        times |= {
            o.start_at + timedelta(seconds=int(rng.uniform(0, span))) for _ in range(random_t)
        }
        times |= _moments(o)
    for st in stories:
        times |= {t - SECOND for t in st.news_times()}
    checked = 0
    for T in sorted(times):
        cut = _visible_cut(T, cfg.feed)
        keep = _fill_bars_known_by(base, T)
        moved = {
            sid: replace(p, bars=tuple(_perturb(b, cut, keep, rng) for b in p.bars))
            for sid, p in paths.items()
            if p.symbol == symbol
        }
        spy_moved = SpyData({d: _perturb(b, cut, set(), rng) for d, b in spy.days.items()})
        again = simulate_symbol(
            symbol,
            [s.before(T) for s in stories],  # type: ignore[attr-defined]
            fac,
            moved,
            spy_moved,
            _NoisyContext(ctx, rng),
            cfg,
            keep_transitions=True,
        )
        started = {o.story_id for o in base if o.start_at is not None and o.start_at <= T}
        assert started == {o.story_id for o in again if o.start_at and o.start_at <= T}, T
        by_id = {o.story_id: o for o in again}
        for o in base:
            if o.story_id in started:
                assert _upto(o, T) == _upto(by_id[o.story_id], T), (o.story_id, T)
                checked += 1
    return checked, base


@pytest.mark.parametrize(
    ("seed", "sessions", "feed"),
    [
        (1, 1, SIP_RT),
        (2, 3, SIP_RT),
        (3, 1, SIP_DELAYED),
        (4, 3, SIP_RT),
        (5, 1, SIP_RT),
        (6, 3, SIP_DELAYED),
    ],
)
def test_nothing_the_bounce_does_by_T_depends_on_what_comes_after(
    seed: int, sessions: int, feed: FeedProfile
) -> None:
    market, context = bounce_market(seed, SYMBOLS, sessions=sessions)
    fac = BounceFactory(context, BounceParams(hold_sessions=sessions))
    cfg = SimConfig(feed=feed)
    rng = np.random.default_rng(3000 + seed)
    by_symbol: dict[str, list[Story]] = {}
    for s in market.stories:
        by_symbol.setdefault(s.symbol, []).append(s)
    checked = 0
    states: Counter[str] = Counter()
    for symbol, stories in sorted(by_symbol.items()):
        n, base = _invariant(symbol, stories, market.paths, market.spy, market.ctx, fac, cfg, rng)
        checked += n
        states.update(f"{o.terminal_state}/{o.reason}" for o in base)
    assert checked > 100
    # The markets exercise the machine, not only its watching.
    assert sum(n for k, n in states.items() if k.startswith("EXITED/")) > 0
    assert sum(n for k, n in states.items() if k.startswith("EXPIRED/")) > 0


def _scenarios() -> dict[str, tuple[list[TStory], dict[str, PathData], SpyData, Context, int]]:
    """Some of ``test_bounce``'s hand-built paths: each decides near an item or a bar."""

    def one(st: TStory, days: list[date], day_bars: list[BarArrays], **kw: object):  # type: ignore[no-untyped-def]
        rows = kw.get("spy_rows", {MON: SPY_ROWS})
        price = float(kw.get("spy_price", 200.0))  # type: ignore[arg-type]
        sp = spy([*days, THU] if len(days) > 1 else days, rows, price=price)  # type: ignore[arg-type]
        ctx = kw.get("ctx", Context())
        return [st], {st.story_id: path(st.story_id, A, days, day_bars)}, sp, ctx, len(days)

    md3 = [MON, TUE, WED]
    gap_tue = bars(TUE, [((9, 30), 93.2)], {(9, 30): (93.5, 93.6, 93.0, 93.2, 1_000.0, 93.3)})
    split_tue = bars(
        TUE, [((9, 30), 47.9), ((10, 1), 48.5)], {(10, 0): (47.95, 48.6, 47.9, 48.5, 2e3, 48.3)}
    )
    a, b = (
        base_story(),
        tstory(A, TUE, (et(TUE, 7, 0), "analyst_downgrade"), (et(TUE, 10, 0), "dilution")),
    )
    flat = [bars(d, [((9, 30), 96.0)]) for d in (TUE, WED, THU)]
    blocked = (
        [a, b],
        {
            a.story_id: path(a.story_id, A, md3, [base_bars(target=False), flat[0], flat[1]]),
            b.story_id: path(b.story_id, A, [TUE, WED, THU], flat),
        },
        spy([*md3, THU], {MON: SPY_ROWS}),
        Context(),
        3,
    )
    return {
        "target": one(base_story(), [MON], [base_bars()]),
        "abort": one(base_story(MON, (et(MON, 10, 5, 30), "dilution")), [MON], [base_bars()]),
        "abort entering": one(
            base_story(MON, (et(MON, 9, 56, 30), "dilution")), [MON], [base_bars()]
        ),
        "veto": one(base_story(MON, (et(MON, 9, 50, 30), "legal_adverse")), [MON], [base_bars()]),
        "veto at entry": one(
            base_story(MON, (et(MON, 9, 56, 5), "legal_adverse")), [MON], [base_bars()]
        ),
        "late detection": one(
            tstory(
                A, MON, (et(MON, 8, 0), "analyst_pt_cut"), (et(MON, 10, 50), "analyst_downgrade")
            ),
            [MON],
            [bars(MON, IN_LEVELS, IN_ROWS)],
            spy_rows={MON: IN_SPY},
            spy_price=201.0,
        ),
        "md3 gap": one(base_story(), md3, [base_bars(target=False), gap_tue, flat[1]]),
        "split": one(
            base_story(),
            md3,
            [base_bars(target=False), split_tue, bars(WED, [((9, 30), 48.5)])],
            ctx=Context(adj={(A, MON): 0.5, (A, TUE): 1.0, (A, WED): 1.0}),
        ),
        "compliance": one(
            base_story(),
            md3,
            [base_bars(target=False), flat[0], flat[1]],
            ctx=Context(verdicts={(A, TUE): "not_halal"}),
        ),
        "blocked, then aborted": blocked,
    }


@pytest.mark.parametrize("feed", [SIP_RT, SIP_DELAYED])
@pytest.mark.parametrize("name", list(_scenarios()))
def test_the_hand_built_paths_are_invariant_too(name: str, feed: FeedProfile) -> None:
    """The known-answer paths, replayed at every decision and every item's last second."""
    stories, paths, spy_data, ctx, sessions = _scenarios()[name]
    context = {s.story_id: (pre_event(s.session), eligibility()) for s in stories}
    if name == "late detection":
        context = {s.story_id: (pre_event(s.session, sigma=0.01), eligibility()) for s in stories}
    fac = BounceFactory(context, BounceParams(hold_sessions=sessions))
    rng = np.random.default_rng(4000)
    checked, base = _invariant(
        A, stories, paths, spy_data, ctx, fac, SimConfig(feed=feed), rng, random_t=10
    )
    assert checked > 0 and base[0].triggered_at is not None


def test_the_bounce_is_deterministic_for_any_worker_count() -> None:
    market, context = bounce_market(7, SYMBOLS, sessions=3)
    fac = BounceFactory(context, BounceParams(hold_sessions=3))

    def go(workers: int) -> str:
        out = simulate_many(
            market.stories, fac, market.paths, market.spy, market.ctx, SimConfig(), workers=workers
        )
        assert any(o.trade is not None for o in out)
        return outcomes_sha256(out)

    assert go(1) == go(4) == go(1)


@pytest.mark.parametrize("parallel", [False, "spawn"])
async def test_the_driver_runs_the_bounce_as_the_simulator_does(
    engine: AsyncEngine, parallel: bool | str
) -> None:
    """``sim.run`` over the database (loader, batches, a spawn pool that pickles the factory)
    gives the records the in-memory simulation gives."""
    market, context = bounce_market(8, SYMBOLS, sessions=3)
    await seed_calendar(engine, date(2016, 1, 4), date(2016, 3, 31))
    await seed_market(engine, market)
    fac = BounceFactory(context, BounceParams(hold_sessions=3))
    sink = MemorySink(run_id=RUN_ID)
    summary = await run(
        engine,
        market.stories,
        fac,
        context=market.ctx,
        window=Window.GATE,
        window_end=date(2016, 3, 31),
        cfg=SimConfig(),
        unlock=WindowUnlock(),
        sink=sink,
        workers=2,
        batch_paths=3,
        parallel=parallel,  # type: ignore[arg-type]
    )
    expected = simulate_many(
        market.stories,
        fac,
        market.paths,
        market.spy,
        market.ctx,
        SimConfig(),
        workers=2,
        run_id=RUN_ID,
    )
    assert summary.trades > 0
    assert sink.info is not None and sink.info.playbook == "overreaction_bounce"
    assert outcomes_sha256(sink.outcomes) == outcomes_sha256(expected)
