"""The gate-only models reproduce the legacy studies through the simulator (R1, S1).

R1: ``LegacyReactorFill`` against ``intraday.entry_and_close`` on random minute
bars. S1: ``DailyBarSource`` against ``study.outcome`` and ``study.evaluate``
on random daily bars. Both must agree to 1e-12 on every observation, and
drop the same ones. Synthetic data only: no stored price is read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import cast

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.playbooks.clock import US, to_us
from halabot.playbooks.exchange import MARKET_FILL
from halabot.playbooks.legacy import (
    BARS_CUT,
    R1_SET_ASIDE,
    R1_SET_ASIDE_CAP,
    SPY_BARS_CUT,
    DailyBarFill,
    DailyBarSource,
    HoldFactory,
    LegacyReactorFill,
    R1Dropped,
    daily_config,
    r1_dropped,
    r1_set_aside,
    reactor_config,
    reactor_plausible,
    reactor_story,
)
from halabot.playbooks.records import TradeRecord
from halabot.playbooks.sim import RunSummary, id_set_sha, simulate_symbol
from halabot.playbooks.types import (
    DATA_SKIPS,
    BarSeries,
    Execution,
    OrderKind,
    PathSkip,
    Session,
    SpyData,
    TradeFacts,
    WorkingOrder,
)
from halal_trader.data.minutes import BarArrays, MinuteBar
from halal_trader.events import intraday, study
from halal_trader.market_hours import MARKET_TZ, is_trading_day
from tests.halabot.playbooks._support import MON, TUE, Context, et, path, session_bars
from tests.halabot.playbooks._synth import random_session

EPS = 1e-12


def _facts(cost_bps: float) -> TradeFacts:
    return TradeFacts("gate", "gate", "legacy", cost_bps=cost_bps, rank=-1, tech=False)


def _order(side: str, active: datetime, k: int = 0) -> WorkingOrder:
    return WorkingOrder("o", side, OrderKind.MARKET, k, 0, to_us(active), None)  # type: ignore[arg-type]


# ── the reactor fill, piece by piece ──


def test_the_legacy_fill_takes_the_vwap_or_open_with_no_clamp_and_no_gap_rule() -> None:
    rows = {
        (10, 16): (100.0, 100.5, 99.5, 100.2, 1e3, 101.7),  # VWAP above the high
        (10, 30): (101.0, 101.5, 100.5, 101.2, 1e3, math.nan),  # no VWAP: the open
        (10, 31): (101.0, 101.5, 100.5, 101.2, 1e3, 0.0),  # a zero VWAP is falsy too
    }
    skip = [(10, m) for m in range(17, 30)]  # a 13-minute gap before 10:30
    bars = BarSeries.build([session_bars(MON, rows=rows, skip=skip)], [1.0], 0)
    fill = LegacyReactorFill()
    assert fill.gate_only and fill.name == "legacy-reactor" and not MARKET_FILL.gate_only

    def at(hh: int, mm: int, ss: int = 0) -> int | None:
        return fill.due(bars, _order("buy", et(MON, hh, mm, ss)))

    i = at(10, 15, 30)
    assert i is not None and bars.bar_time(i) == et(MON, 10, 16)
    assert fill.price(bars, i, _order("buy", et(MON, 10, 15, 30)), gap_us=0, session_open_us=0) == (
        101.7,
        (),
    )
    j = at(10, 16, 1)  # waits 14 minutes for the 10:30 bar, right after a gap: still no open
    assert j is not None and bars.bar_time(j) == et(MON, 10, 30)
    assert (
        fill.price(bars, j, _order("buy", et(MON, 10, 16, 1)), gap_us=0, session_open_us=0)[0]
        == 101.0
    )
    k = at(10, 30, 30)
    assert (
        k is not None
        and fill.price(bars, k, _order("buy", et(MON, 10, 30, 30)), gap_us=0, session_open_us=0)[0]
        == 101.0
    )
    assert fill.due(bars, _order("sell", et(MON, 10, 15))) is None  # exits wait for the close
    assert at(16, 0, 0) is None


def test_the_legacy_spy_leg_is_spys_own_first_bar_and_last_close() -> None:
    stock = BarSeries.build([session_bars(MON, price=50.0)], [1.0], 0)
    spy_rows = {
        (10, 17): (200.0, 201.0, 199.0, 200.5, 1e5, 200.3),
        (15, 58): (202.0, 202.5, 201.5, 202.2, 1e5, 202.1),
    }
    spy = BarSeries.build(
        [session_bars(MON, price=200.0, rows=spy_rows, skip=[(10, 16), (15, 59)])], [1.0], 0
    )
    fill = LegacyReactorFill()
    order = _order("buy", et(MON, 10, 15, 30))
    i = fill.due(stock, order)
    assert i is not None
    ex = Execution(order, 0, i, int(stock.ts[i]), 50.0, 0)
    assert fill.spy_price(spy, ex) == (200.3, ())  # SPY's 10:17, not a proxy for 10:16
    close = fill.at_close(stock, spy, 0)
    assert close is not None
    assert (close.rule, close.price, close.spy_price, close.flags) == (
        "last_close",
        50.0,
        202.2,
        ("last_close",),
    )
    assert close.bar_ts == int(et(MON, 15, 59).timestamp())
    empty = BarSeries.build([BarArrays.empty()], [1.0], 0)
    assert fill.at_close(empty, spy, 0) is None  # no bar: the market rule's fallback
    late = _order("buy", et(MON, 15, 59, 30))  # SPY's last print is 15:58
    px, flags = fill.spy_price(spy, Execution(late, 0, 0, 0, 1.0, 0))
    assert math.isnan(px) and flags == ("no_spy",)


# ── R1: the reactor study through the simulator ──


def _minute_rows(bars: BarArrays) -> list[MinuteBar]:
    return [
        MinuteBar(
            datetime.fromtimestamp(int(bars.ts[i]), UTC),
            float(bars.o[i]),
            float(bars.h[i]),
            float(bars.l[i]),
            float(bars.c[i]),
            float(bars.v[i]),
            None if math.isnan(float(bars.vw[i])) else float(bars.vw[i]),
        )
        for i in range(len(bars))
    ]


def _legacy_same_day(stock: BarArrays, spy: BarArrays, published: datetime, bps: float):  # type: ignore[no-untyped-def]
    """``intraday.run``'s same-day return of one headline (None where it drops it)."""
    at = published + intraday.LATENCY
    s = intraday.entry_and_close(_minute_rows(stock), at)
    q = intraday.entry_and_close(_minute_rows(spy), at)
    if s is None or q is None:
        return None
    (e, c), (se, sc) = s, q
    if not intraday._PLAUSIBLE[0] < c / e < intraday._PLAUSIBLE[1]:
        return None
    return (c / e - 1) - (sc / se - 1) - 2 * bps / 10_000


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_r1_the_simulator_reproduces_the_reactor_study(seed: int) -> None:
    rng = np.random.default_rng(seed)
    day = MON
    raw = random_session(rng, day, 40.0)
    vw = raw.vw.copy()
    vw[rng.random(len(vw)) < 0.05] *= 1.02  # some VWAPs outside the bar: never clamped
    keep = slice(0, len(raw) - 45)  # no print after about 15:14: later headlines find none
    stock = BarArrays(
        raw.ts[keep], raw.o[keep], raw.h[keep], raw.l[keep], raw.c[keep], raw.v[keep], vw[keep]
    )
    spy = random_session(rng, day, 200.0)  # holes of its own: SPY's first bar may differ
    ctx = Context(adj={("AAA", day): 0.97, ("SPY", day): 0.995})
    spy_data = SpyData({day: spy})
    open_ = datetime.combine(day, time(9, 30), MARKET_TZ)
    checked = dropped = 0
    for n in range(80):
        published = open_ + timedelta(seconds=float(rng.uniform(0, 6 * 3600)))  # 09:30-15:30
        bps = 7.0 if n % 2 else 15.0
        want = _legacy_same_day(stock, spy, published, bps)
        st = reactor_story(f"AAA:{n}", "AAA", published)
        (out,) = simulate_symbol(
            "AAA",
            [st],
            HoldFactory(1, {st.story_id: _facts(bps)}),
            {st.story_id: path(st.story_id, "AAA", [day], [stock])},
            spy_data,
            ctx,
            reactor_config(),
        )
        if want is None:
            assert out.trade is None, published
            dropped += 1
            continue
        t = out.trade
        assert t is not None, published
        assert abs(t.r_net_abn - want) <= EPS, (published, t.r_net_abn, want)
        assert t.entry_active_at == published + intraday.LATENCY
        checked += 1
    assert checked > 40 and dropped > 0  # the late headlines find no bar, as in the study


def test_r1_admits_what_the_study_took_and_never_flattens() -> None:
    """Before 09:50, after the 15:00 cutoff, on a screen that is not halal; exit at the last bar."""
    bars = session_bars(MON, last=(15, 57))  # the last print is 15:57: no flatten fill
    ctx = Context(verdicts={("AAA", MON): "not_halal"})
    for hh, mm in ((9, 30), (15, 20)):
        published = et(MON, hh, mm)
        st = reactor_story("AAA:x", "AAA", published)
        (out,) = simulate_symbol(
            "AAA",
            [st],
            HoldFactory(1, {st.story_id: _facts(7.0)}),
            {st.story_id: path(st.story_id, "AAA", [MON], [bars])},
            SpyData({MON: session_bars(MON, price=200.0)}),
            ctx,
            reactor_config(),
        )
        t = out.trade
        assert t is not None and t.entry_bar_ts == published + timedelta(minutes=1)
        assert t.exit_bar_ts == et(MON, 15, 57) and t.flags == ("last_close",)
        assert t.exit_decided_at == et(MON, 16, 0)  # taken at the close, not at the flatten


# ── R1: the dropped sets ──


def _summary(ids: list[str], **kw: object) -> RunSummary:
    """Run ``r``'s summary over the explicit set ``ids``."""
    n = len(ids)
    base: dict[str, object] = dict(
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
    return RunSummary(**{**base, **kw})  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class _Trade:
    """The fields of a TradeRecord the comparison reads (run ``r``'s, a finite return)."""

    story_id: str
    entry_px: float
    exit_px: float
    run_id: str = "r"
    r_net_abn: float = 0.01


def test_r1_sets_aside_the_loader_rules_the_study_never_had() -> None:
    """Each id with every reason; SPY's cut bars apart from the stock's; coverage skips stay."""
    summary = _summary(
        list("abcmnpstuxy"),
        skip_ids={
            "bad_bars": ("b",),
            "adjust_defect": ("a",),
            "no_daily": ("n",),
            "spy_thin": ("t",),
            "spy_missing": ("m",),  # compared: the study drops a SPY session with no bar too
            "units_missing": ("u",),
            "halted_all_day": ("h",),
        },
        bar_drop_ids=("c", "x"),
        spy_drop_ids=("s", "x", "y"),  # SPY's bars cut on S
        spy_drop_from_start_ids=("s", "x"),  # at or after the decision: y decided after the cut
        spare_drop_ids=("p",),  # R1 never reads the spare session
    )
    assert R1_SET_ASIDE == ("adjust_defect", "bad_bars", "no_daily", "spy_thin")
    assert not set(R1_SET_ASIDE) <= DATA_SKIPS  # not the Stage A data skips
    assert r1_set_aside(summary) == {
        "a": ("adjust_defect",),
        "b": ("bad_bars",),
        "c": (BARS_CUT,),
        "n": ("no_daily",),
        "s": (SPY_BARS_CUT,),
        "t": ("spy_thin",),
        "x": (BARS_CUT, SPY_BARS_CUT),
    }


def test_r1_compares_the_rest_and_caps_the_set_aside() -> None:
    ids = [f"h{i:03d}" for i in range(200)]
    trades = [_Trade(i, 50.0, 51.0) for i in ids[:190]]
    trades[5] = _Trade(ids[5], 50.0, 101.0)  # exit / entry 2.02: implausible
    summary = _summary(
        ids,
        dropped=dict.fromkeys(ids[190:], "entry_unfilled"),
        bar_drop_ids=(ids[0],),
        spy_drop_ids=(ids[0],),
        spy_drop_from_start_ids=(ids[0],),  # one headline, two reasons
        skip_ids={"no_daily": (ids[199],)},  # set aside, and dropped by both sides
    )
    study = {ids[5], *ids[190:]}
    check = r1_dropped(ids, summary, cast(list[TradeRecord], trades), study)
    assert check.set_aside == {ids[0]: (BARS_CUT, SPY_BARS_CUT), ids[199]: ("no_daily",)}
    assert check.share == 0.01 and check.within_cap  # the cap is inclusive
    assert check.identical and check.passed and check.compared == 198
    assert check.kept == tuple(i for i in ids[1:190] if i != ids[5])
    config = check.as_config()
    assert config["set_aside_by_reason"] == {BARS_CUT: 1, SPY_BARS_CUT: 1, "no_daily": 1}
    assert config["set_aside_cap"] == R1_SET_ASIDE_CAP == 0.01
    assert config["sim_only"] == {} and config["study_only"] == []
    # One more headline set aside is over the cap; one-sided drops are listed with reasons.
    over = _summary(ids, dropped=summary.dropped, skip_ids={"bad_bars": (ids[0], ids[1], ids[2])})
    mixed = r1_dropped(ids, over, cast(list[TradeRecord], trades), {*study, ids[7]} - {ids[195]})
    assert mixed.share == 0.015 and not mixed.within_cap and not mixed.passed
    assert mixed.sim_only == {ids[195]: "entry_unfilled"} and mixed.study_only == (ids[7],)
    with pytest.raises(ValueError, match="not in H"):
        r1_dropped(ids, summary, cast(list[TradeRecord], trades), {"elsewhere"})
    assert reactor_plausible(cast(TradeRecord, _Trade("p", 50.0, 99.9)))
    assert not reactor_plausible(cast(TradeRecord, _Trade("p", 50.0, 25.0)))  # bounds excluded


def test_r1_needs_its_run_its_trades_and_h_to_be_one_set() -> None:
    """A run over H with no drops passes only with each kept id's own trade of that run."""
    ids = ["a", "b", "c"]
    trades = [_Trade(i, 50.0, 51.0) for i in ids]

    def check(summary: RunSummary, given: list[_Trade], study: set[str] | None = None) -> R1Dropped:
        return r1_dropped(ids, summary, cast(list[TradeRecord], given), study or set())

    ok = check(_summary(ids), trades)
    assert ok.passed and ok.kept == tuple(ids)
    # No drops and no trades: three headlines never compared, which used to pass.
    with pytest.raises(ValueError, match="3 ids of H the run kept have no trade of it"):
        check(_summary(ids), [])
    with pytest.raises(ValueError, match=r"kept have no trade of it .*: b$"):
        check(_summary(ids), [trades[0], _Trade("b", 50.0, 51.0, r_net_abn=math.nan), trades[2]])
    # A run over another set of the same size, or not over an explicit set at all.
    with pytest.raises(ValueError, match="not H's"):
        check(_summary(["a", "b", "d"]), trades)
    with pytest.raises(ValueError, match=r"sha \(none\)"):
        check(_summary(ids, expected_sha=""), trades)
    # Another run's trades, a trade twice, a trade outside H.
    with pytest.raises(ValueError, match="trade a is from run other, not r"):
        check(_summary(ids), [_Trade("a", 50.0, 51.0, run_id="other"), *trades[1:]])
    with pytest.raises(ValueError, match="trade c is given twice"):
        check(_summary(ids), [*trades, trades[2]])
    with pytest.raises(ValueError, match="trade z is not in H"):
        check(_summary(ids), [*trades, _Trade("z", 50.0, 51.0)])
    # An id the run dropped or set aside needs no trade; a NaN trade it dropped is allowed.
    nospy = _Trade("c", 50.0, 51.0, r_net_abn=math.nan)
    partial = _summary(ids, dropped={"b": "bad_bars", "c": "no_spy"}, skip_ids={"bad_bars": ("b",)})
    out = check(partial, [trades[0], nospy], {"c"})
    assert out.identical and out.kept == ("a",) and out.set_aside == {"b": ("bad_bars",)}


def test_r1_fails_when_no_headline_is_kept() -> None:
    """Nothing compared is no pass: an empty H, or every id dropped by both sides."""
    empty = r1_dropped([], _summary([]), [], set())
    assert empty.identical and empty.within_cap and empty.kept == () and not empty.passed
    ids = ["a", "b"]
    both = r1_dropped(ids, _summary(ids, dropped=dict.fromkeys(ids, "units_missing")), [], set(ids))
    assert both.identical and both.within_cap and both.kept == () and not both.passed


# ── S1: the daily-bar study through the simulator ──


def _calendar(lo: date, hi: date) -> list[date]:
    return [
        lo + timedelta(days=i)
        for i in range((hi - lo).days + 1)
        if is_trading_day(lo + timedelta(days=i))
    ]


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


def test_daily_entries_follow_study_entry_point() -> None:
    sessions = _calendar(date(2016, 11, 21), date(2016, 12, 2))
    src = DailyBarSource(_bars(np.random.default_rng(0), sessions, ["AAA", "SPY"]))
    half = date(2016, 11, 25)  # closes at 13:00

    def entry(d: date, hh: int, mm: int):  # type: ignore[no-untyped-def]
        e = src.entry(datetime.combine(d, time(hh, mm), MARKET_TZ))
        assert e is not None
        return e.session, e.at, e.decide_at, e.lookahead

    assert entry(half, 8, 0) == (half, "open", et(half, 9, 30), False)
    assert entry(half, 12, 0) == (half, "close", et(half, 9, 30, 1), False)
    assert entry(half, 14, 0) == (
        half,
        "close",
        et(half, 9, 30, 1),
        True,
    )  # entry_point's look-ahead
    assert entry(date(2016, 11, 24), 10, 0)[:2] == (half, "open")  # Thanksgiving
    assert entry(date(2016, 11, 23), 17, 0)[:2] == (half, "open")
    assert entry(date(2016, 11, 26), 10, 0)[:2] == (date(2016, 11, 28), "open")  # a Saturday
    assert src.entry(datetime(2016, 10, 1, 12, tzinfo=UTC)) is None  # before the calendar
    pseudo = src.pseudo("AAA", half)
    assert [datetime.fromtimestamp(int(t), UTC) for t in pseudo.ts] == [
        et(half, 9, 30),
        et(half, 12, 59),
    ]
    assert (float(pseudo.o[0]), float(pseudo.c[1])) == (
        src._bars.open["AAA"][half],
        src._bars.close["AAA"][half],
    )
    assert np.all(pseudo.o == pseudo.vw) and np.all(np.isnan(pseudo.v))
    assert src.adj("AAA", half) == 1.0 and src.daily("AAA", half) is None
    assert DailyBarFill().gate_only and daily_config().fill is not None


@pytest.mark.parametrize("seed", [4, 5])
def test_s1_the_simulator_reproduces_study_outcome(seed: int) -> None:
    rng = np.random.default_rng(seed)
    sessions = _calendar(date(2016, 2, 1), date(2016, 7, 29))
    bars = _bars(rng, sessions, ["AAA", "BBB", "SPY"])
    # Holes: a missing close (an exit), a missing open (an open entry), a zero price.
    del bars.close["AAA"][date(2016, 3, 14)]
    del bars.open["BBB"][date(2016, 4, 4)]
    bars.close["BBB"][date(2016, 5, 2)] = 0.0
    src = DailyBarSource(bars)
    lo = datetime(2016, 2, 1, tzinfo=UTC)
    checked = skipped = multi = 0
    by_h: dict[int, list[int]] = {h: [0, 0] for h in study.HORIZONS}  # h -> [checked, skipped]
    cases = [(lo + timedelta(seconds=float(rng.uniform(0, 150 * 86400)))) for _ in range(60)]
    cases += [
        datetime.combine(date(2016, 3, 11), time(15, 0), MARKET_TZ),  # close entry, exit 03-14+
        datetime.combine(date(2016, 4, 4), time(8, 0), MARKET_TZ),  # BBB: no open
        datetime.combine(date(2016, 4, 29), time(16, 30), MARKET_TZ),  # BBB: zero close 05-02
    ]
    for n, published in enumerate(cases):
        for symbol in ("AAA", "BBB"):
            point = study.entry_point(published, sessions)
            entry = src.entry(published)
            assert (entry is None) == (point is None)
            if entry is None or point is None:
                continue
            assert (entry.index, entry.at) == point
            for h in study.HORIZONS:  # 1, 5, 20 and 60 (spec §E.3 S1)
                bps = (7.0, 15.0, 30.0)[n % 3]
                want = study.outcome(bars, symbol, point, h, bps)
                got = src.simulate(
                    entry, story_id=f"{symbol}:{n}:{h}", symbol=symbol, horizon=h, facts=_facts(bps)
                )
                if want is None:
                    assert isinstance(got, PathSkip), (published, symbol, h)
                    skipped += 1
                    by_h[h][1] += 1
                    continue
                assert not isinstance(got, PathSkip), (published, symbol, h)
                t = got.trade
                assert t is not None
                assert abs(t.r_net_abn - want) <= EPS, (published, symbol, h)
                # The study held to its exit: no pre-open compliance sell (the source has
                # no screen), the exit decided at the deadline's close.
                assert t.exit_reason == "time_stop", (published, symbol, h)
                assert t.exit_decided_at == Session.of(t.exit_session).close
                assert got.reason == "time_stop" and got.terminal_state == "EXITED"
                checked += 1
                by_h[h][0] += 1
                multi += t.sessions_held > 1
    assert checked > 200 and skipped > 0 and multi > 100
    assert by_h[60][0] > 20 and by_h[60][1] > 20  # h=60 runs past the calendar for later events


async def test_s1_matches_study_evaluate_on_the_database(engine: AsyncEngine) -> None:
    rng = np.random.default_rng(6)
    sessions = _calendar(date(2016, 1, 4), date(2016, 6, 30))
    bars = _bars(rng, sessions, ["AAA", "BBB", "SPY"])
    rows = []
    for s in bars.close:
        for d in sessions:
            o, c = bars.open[s][d], bars.close[s][d]
            rows.append({"s": s, "d": d, "o": o, "h": max(o, c), "l": min(o, c), "c": c})
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, volume, "
                "fetched_at) VALUES (:s, :d, 'all', :o, :h, :l, :c, 1000, now())"
            ),
            rows,
        )
    observations = [
        study.Observation(
            ("AAA", "BBB")[n % 2],
            datetime(2016, 1, 4, 12, tzinfo=UTC)
            + timedelta(seconds=float(rng.uniform(0, 160 * 86400))),
            float(n),
        )
        for n in range(40)
    ]
    want = await study.evaluate(engine, observations)  # study.HORIZONS: 1, 5, 20, 60
    src = await DailyBarSource.load(engine, ["AAA", "BBB"])
    assert src.sessions == sessions
    got = []
    for n, obs in enumerate(observations):
        entry = src.entry(obs.published_at)
        if entry is None:
            continue
        for h in study.HORIZONS:
            out = src.simulate(
                entry,
                story_id=f"{n}:{h}",
                symbol=obs.symbol,
                horizon=h,
                facts=_facts(study.cost_bps(None)),  # no universe stored: rank None, 30 bps
            )
            if not isinstance(out, PathSkip):
                assert out.trade is not None and out.trade.exit_reason == "time_stop"
                got.append((obs.published_at, h, out.trade.r_net_abn))
    expected = [(o.published_at, h, r) for o, h, r in want]
    assert len(got) == len(expected) > 60
    assert sum(h == 60 for _, h, _ in expected) > 5
    for (t1, h1, r1), (t2, h2, r2) in zip(sorted(got), sorted(expected), strict=True):
        assert (t1, h1) == (t2, h2) and abs(r1 - r2) <= EPS


def test_a_sell_never_fills_on_a_bar_under_a_gate_model() -> None:
    bars = BarSeries.build([session_bars(MON), session_bars(TUE)], [1.0, 1.0], 0)
    order = WorkingOrder("s", "sell", OrderKind.MARKET, 1, 0, to_us(et(TUE, 10, 0)), 1.0)
    assert DailyBarFill().due(bars, order) is None
    assert MARKET_FILL.due(bars, order) == bars.session_start(1) + 30
    assert int(bars.ts[MARKET_FILL.due(bars, order) or 0]) * US == to_us(et(TUE, 10, 0))
