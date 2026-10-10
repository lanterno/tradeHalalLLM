"""The simulator end to end with the toy playbook: fills, exits, fallbacks, sessions, overlap.

Every expected number is worked out by hand from the synthetic bars, to 1e-12.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from halabot.playbooks.clock import from_us
from halabot.playbooks.playbook import Ctx
from halabot.playbooks.sim import PitStory, SymbolState, simulate_symbol, start_time
from halabot.playbooks.types import (
    BarIn,
    Cancel,
    ComplianceIn,
    FillIn,
    Finish,
    GapIn,
    Input,
    Intent,
    NewsIn,
    OrderClosedIn,
    PathSkip,
    SessionIn,
    SetTimer,
    SimConfig,
    Submit,
    TimerIn,
    TradeFacts,
    Transition,
)
from tests.halabot.playbooks._support import (
    FACTS,
    HALF,
    MON,
    TUE,
    WED,
    Context,
    Item,
    Recorder,
    Row,
    Toy,
    daily,
    downgrade,
    et,
    path,
    run_one,
    session_bars,
    spy_data,
    story,
)

EPS = 1e-12
UNREACHABLE = {"target": 1.0}  # a 100% target: only the time stop exits


def _rows_from(day_rows: Mapping[tuple[int, int], Row], *, after: tuple[int, int], price: float):
    rows = {
        (h, m): (price, price, price, price, 1_000.0, price)
        for h in range(9, 16)
        for m in range(60)
    }
    rows = {k: v for k, v in rows.items() if k >= after}
    rows.update(day_rows)
    return rows


def _spy_with(day, rows):  # type: ignore[no-untyped-def]
    spy = spy_data([day])
    spy.days[day] = session_bars(day, price=200.0, rows=rows)
    return spy


# ── entries and exits ──


def test_entry_and_target_exit_known_answer() -> None:
    rows = _rows_from(
        {
            (9, 51): (100.0, 101.0, 99.5, 100.5, 1_000.0, 100.2),
            (10, 30): (101.0, 101.6, 100.9, 101.5, 1_000.0, 101.3),
            (10, 32): (101.5, 101.6, 101.3, 101.5, 2_000.0, 101.4),
        },
        after=(10, 31),
        price=101.5,
    )
    st = story("AAA", MON, downgrade(MON))
    pd = path(st.story_id, "AAA", [MON], [session_bars(MON, rows=rows)])
    spy = _spy_with(MON, {(10, 32): (200.0, 201.5, 199.5, 201.0, 1e5, 201.0)})
    out = run_one(st, pd, spy=spy)

    t = out.trade
    assert t is not None
    assert out.terminal_state == "EXITED" and out.reason == "target"
    assert out.start_at == et(MON, 9, 30)
    assert t.entry_decided_at == et(MON, 9, 50, 5)  # the 09:49 bar became visible
    assert t.entry_active_at == et(MON, 9, 50, 8)
    assert t.entry_bar_ts == et(MON, 9, 51)
    assert t.entry_px == 100.2
    assert t.exit_decided_at == et(MON, 10, 31, 5)
    assert t.exit_active_at == et(MON, 10, 31, 8)
    assert t.exit_bar_ts == et(MON, 10, 32)
    assert t.exit_px == 101.4
    assert t.spy_entry_px == 200.0 and t.spy_exit_px == 201.0
    assert abs(t.r_gross - (101.4 / 100.2 - 1)) < EPS
    assert abs(t.r_spy - 0.005) < EPS
    assert abs(t.r_net_abn - (101.4 / 100.2 - 1 - 0.005 - 0.003)) < EPS
    assert abs(t.r_beta_adj - t.r_net_abn) < EPS  # beta 1
    assert t.cost_bps == 15.0 and t.exit_reason == "target"
    assert t.hold_minutes == 41
    assert abs(t.participation - (10_000 / 100.2) / 1_000) < EPS
    assert abs(t.mae - (99.5 / 100.2 - 1)) < EPS
    assert abs(t.mfe - (101.6 / 100.2 - 1)) < EPS
    assert t.flags == ()
    assert t.sessions_held == 1 and t.exit_session == MON
    assert out.triggered_at == et(MON, 9, 50, 5) and out.armed_at == et(MON, 9, 50, 5)
    assert out.entry_decided_at == et(MON, 9, 50, 5) and out.entry_bar_ts == et(MON, 9, 51)
    (leg,) = out.legs
    assert (leg.day, leg.a, leg.z, leg.spy_a, leg.spy_z, leg.cost) == (
        MON,
        100.2,
        101.4,
        200.0,
        201.0,
        0.003,
    )
    assert abs(leg.abn - t.r_net_abn) < EPS
    states = [(to, why) for _, _, to, why in out.transitions]
    assert states == [
        ("WATCHING", "start"),
        ("ARMED", "triggered"),
        ("ENTERING", "entry"),
        ("ENTERED", "filled"),
        ("EXITING", "target"),
        ("EXITED", "target"),
    ]


def test_the_time_stop_flattens_at_close_minus_five() -> None:
    rows = {(15, 56): (100.0, 100.4, 99.8, 100.1, 1_000.0, 100.3)}
    st = story("AAA", MON, downgrade(MON))
    pd = path(st.story_id, "AAA", [MON], [session_bars(MON, rows=rows)])
    out = run_one(st, pd, toy=UNREACHABLE)
    t = out.trade
    assert t is not None
    assert t.exit_reason == "time_stop"
    assert t.exit_decided_at == et(MON, 15, 55) and t.exit_active_at == et(MON, 15, 55, 3)
    assert t.exit_bar_ts == et(MON, 15, 56) and t.exit_px == 100.3
    assert out.terminal_state == "EXITED" and out.reason == "time_stop"


def test_an_early_close_moves_the_cutoff_and_the_flatten() -> None:
    rows = {(12, 56): (100.0, 100.4, 99.8, 100.1, 1_000.0, 100.2)}
    bars = session_bars(HALF, rows=rows)
    assert int(bars.ts[-1]) == int(et(HALF, 12, 59).timestamp())  # 13:00 close
    st = story("AAA", HALF, downgrade(HALF))
    out = run_one(
        st, path(st.story_id, "AAA", [HALF], [bars]), toy={"target": 1.0, "entry_bar": (11, 58)}
    )
    t = out.trade
    assert t is not None
    assert t.entry_decided_at == et(HALF, 11, 59, 5)  # inside [09:50, 12:00]
    assert t.exit_decided_at == et(HALF, 12, 55) and t.exit_bar_ts == et(HALF, 12, 56)
    assert t.exit_px == 100.2

    # The 11:59 bar is visible at 12:00:05, after the 12:00 cutoff: the toy expires first.
    late = run_one(st, path(st.story_id, "AAA", [HALF], [bars]), toy={"entry_bar": (11, 59)})
    assert late.trade is None
    assert (late.terminal_state, late.reason) == ("EXPIRED", "cutoff")
    assert late.transitions[-1][0] == et(HALF, 12, 0)


def test_an_entry_without_a_bar_expires_unfilled() -> None:
    st = story("AAA", MON, downgrade(MON))
    bars = session_bars(MON, last=(13, 59))  # no print after 13:59
    rec = Recorder(entry_bar=(13, 58))
    out = run_one(st, path(st.story_id, "AAA", [MON], [bars]), factory=rec)
    assert out.trade is None and out.entry_bar_ts is None
    assert out.entry_decided_at == et(MON, 13, 59, 5)
    assert (out.terminal_state, out.reason) == ("EXPIRED", "entry_unfilled")
    closed = [e for e in rec.made[0].seen if isinstance(e, OrderClosedIn)]
    assert [(c.status, c.reason, c.at) for c in closed] == [
        ("cancelled", "flatten", et(MON, 15, 55))  # the deadline's flatten cancels it
    ]

    # On a multi-day path the buy lives to S's close and expires there.
    rec3 = Recorder(entry_bar=(13, 58), sessions=3)
    pd3 = path(st.story_id, "AAA", [MON, TUE, WED], [bars, session_bars(TUE), session_bars(WED)])
    out3 = run_one(st, pd3, factory=rec3)
    closed3 = [e for e in rec3.made[0].seen if isinstance(e, OrderClosedIn)]
    assert [(c.status, c.reason, c.at) for c in closed3] == [("expired", "close", et(MON, 16, 0))]
    assert out3.reason == "entry_unfilled" and out3.trade is None


def test_a_gap_fills_at_the_open_of_the_bar_that_ends_it() -> None:
    skip = [(9, m) for m in range(51, 57)]  # no print 09:51-09:56
    rows = {(9, 57): (100.6, 100.9, 100.5, 100.8, 1_000.0, 100.7)}
    st = story("AAA", MON, downgrade(MON))
    rec = Recorder(**UNREACHABLE)
    out = run_one(
        st, path(st.story_id, "AAA", [MON], [session_bars(MON, rows=rows, skip=skip)]), factory=rec
    )
    t = out.trade
    assert t is not None
    assert t.entry_bar_ts == et(MON, 9, 57) and t.entry_px == 100.6  # the open, not the VWAP
    assert "gap_fill" in t.flags and "late_open" not in t.flags
    gaps = [e for e in rec.made[0].seen if isinstance(e, GapIn)]
    assert [(g.since, g.until, g.late_open) for g in gaps] == [
        (et(MON, 9, 50), et(MON, 9, 57), False)
    ]
    # The GapIn comes just before the BarIn of the bar that ends the gap.
    seen = rec.made[0].seen
    k = seen.index(gaps[0])
    assert isinstance(seen[k + 1], BarIn) and seen[k + 1].at == gaps[0].at


def test_compliance_exit_at_pre_open_of_the_next_session() -> None:
    st = story("AAA", MON, downgrade(MON))
    tue = session_bars(TUE, rows={(9, 31): (99.0, 99.5, 98.8, 99.2, 1_000.0, 99.1)})
    pd = path(st.story_id, "AAA", [MON, TUE, WED], [session_bars(MON), tue, session_bars(WED)])
    ctx = Context(
        verdicts={("AAA", TUE): "not_halal"},
        daily={("AAA", MON): daily(100.0), ("SPY", MON): daily(200.0)},
    )
    rec = Recorder(target=1.0, sessions=3)
    out = run_one(st, pd, ctx=ctx, factory=rec)
    t = out.trade
    assert t is not None
    assert t.exit_reason == "compliance"
    assert t.exit_decided_at == et(TUE, 9, 20) and t.exit_active_at == et(TUE, 9, 30, 3)
    assert t.exit_bar_ts == et(TUE, 9, 31) and t.exit_px == 99.1
    assert t.exit_session == TUE and t.sessions_held == 2
    compliance = [e for e in rec.made[0].seen if isinstance(e, ComplianceIn)]
    assert [(c.at, c.day, c.verdict) for c in compliance] == [(et(TUE, 9, 20), TUE, "not_halal")]
    assert [leg.day for leg in out.legs] == [MON, TUE]
    mon, tue_leg = out.legs
    assert (mon.a, mon.z, mon.cost) == (100.0, 100.0, 0.0015)  # entry VWAP 100 to MON's close
    assert (tue_leg.a, tue_leg.z, tue_leg.cost) == (100.0, 99.1, 0.0015)
    assert t.hold_minutes == 390 - 21 + 1  # 09:51 MON -> 16:00, then 09:30 -> 09:31 TUE


def test_a_late_first_bar_fills_at_its_open_flagged_late() -> None:
    st = story("AAA", MON, downgrade(MON))
    tue = session_bars(TUE, first=(9, 40), rows={(9, 40): (98.0, 98.5, 97.5, 98.2, 1_000.0, 98.1)})
    pd = path(st.story_id, "AAA", [MON, TUE, WED], [session_bars(MON), tue, session_bars(WED)])
    ctx = Context(verdicts={("AAA", TUE): "doubtful"})
    rec = Recorder(target=1.0, sessions=3)
    out = run_one(st, pd, ctx=ctx, factory=rec)
    t = out.trade
    assert t is not None
    assert t.exit_bar_ts == et(TUE, 9, 40) and t.exit_px == 98.0
    assert "gap_fill" in t.flags and "late_open" in t.flags
    # The exit filled (09:41:00) before the 09:40 bar was visible (09:41:05) and the toy
    # finished on its fill, so it never saw that bar. A playbook still running does:
    still = Recorder(target=1.0, sessions=3)
    run_one(st, pd, factory=still)
    late = [e for e in still.made[0].seen if isinstance(e, GapIn) and e.late_open]
    assert [(g.at, g.since, g.until) for g in late] == [(et(TUE, 9, 41, 5), None, et(TUE, 9, 40))]


def test_flatten_on_the_deadline_of_a_multi_day_path() -> None:
    st = story("AAA", MON, downgrade(MON))
    wed = session_bars(WED, rows={(15, 56): (101.0, 101.2, 100.8, 101.0, 1_000.0, 101.1)})
    pd = path(st.story_id, "AAA", [MON, TUE, WED], [session_bars(MON), session_bars(TUE), wed])
    out = run_one(st, pd, toy={"target": 1.0, "sessions": 3})
    t = out.trade
    assert t is not None
    assert (t.exit_reason, t.exit_bar_ts, t.exit_px) == ("time_stop", et(WED, 15, 56), 101.1)
    assert t.sessions_held == 3 and [leg.day for leg in out.legs] == [MON, TUE, WED]


# ── exits without a market ──


def test_close_fallback_uses_the_official_close() -> None:
    st = story("AAA", MON, downgrade(MON))
    pd = path(st.story_id, "AAA", [MON], [session_bars(MON, last=(15, 54))])
    ctx = Context(daily={("AAA", MON): daily(100.7), ("SPY", MON): daily(200.5)})
    out = run_one(st, pd, ctx=ctx, toy=UNREACHABLE)
    t = out.trade
    assert t is not None
    assert t.exit_px == 100.7 and t.spy_exit_px == 200.5
    assert t.exit_bar_ts is None and t.flags == ("close_fallback",)
    assert t.exit_reason == "time_stop" and t.exit_session == MON
    assert abs(t.r_gross - 0.007) < EPS and abs(t.r_spy - 0.0025) < EPS
    assert out.terminal_state == "EXITED"


def test_no_market_fills_on_the_spare_session_by_the_market_rule() -> None:
    st = story("AAA", MON, downgrade(MON))
    spare = session_bars(
        TUE, first=(9, 30), rows={(9, 31): (101.0, 101.5, 100.5, 101.2, 1e3, 101.3)}
    )
    pd = path(
        st.story_id, "AAA", [MON], [session_bars(MON, last=(15, 54))], spare=TUE, spare_bars=spare
    )
    spy = spy_data([MON, TUE])
    spy.days[TUE] = session_bars(TUE, price=200.0, rows={(9, 31): (200, 202, 199, 201, 1e5, 201.5)})
    out = run_one(st, pd, spy=spy, toy=UNREACHABLE)  # no daily bar for AAA on MON
    t = out.trade
    assert t is not None
    assert t.flags == ("no_market",)
    assert t.exit_session == TUE and t.exit_bar_ts == et(TUE, 9, 31) and t.exit_px == 101.3
    assert t.spy_exit_px == 201.5
    assert t.exit_active_at == et(TUE, 9, 30, 3)
    assert [leg.day for leg in out.legs] == [MON, TUE]
    assert out.legs[0].z == 100.0  # MON's mark without a daily bar: its last bar's close


def test_unresolved_is_marked_at_the_last_trade_and_kept() -> None:
    st = story("AAA", MON, downgrade(MON))
    rows = {(15, 54): (100.0, 100.6, 99.9, 100.5, 1_000.0, 100.4)}
    pd = path(st.story_id, "AAA", [MON], [session_bars(MON, rows=rows, last=(15, 54))])
    spy = _spy_with(MON, {(15, 54): (200.0, 200.3, 199.8, 200.2, 1e5, 200.1)})
    out = run_one(st, pd, spy=spy, toy=UNREACHABLE)
    t = out.trade
    assert t is not None
    assert t.flags == ("unresolved",) and t.exit_bar_ts is None
    assert t.exit_px == 100.5 and t.spy_exit_px == 200.2  # both closes of the 15:54 minute


def test_an_exit_unfilled_at_a_non_deadline_close_carries_to_the_next_open() -> None:
    st = story("AAA", MON, downgrade(MON))
    tue = session_bars(TUE, rows={(9, 31): (99.0, 99.5, 98.8, 99.2, 1_000.0, 99.3)})
    pd = path(
        st.story_id,
        "AAA",
        [MON, TUE, WED],
        [session_bars(MON, last=(15, 57)), tue, session_bars(WED)],
    )
    out = run_one(st, pd, toy={"sessions": 3, "target": 1.0, "sell_at": (0, 15, 57)})
    t = out.trade
    assert t is not None
    assert t.exit_reason == "stop"
    assert t.exit_decided_at == et(MON, 15, 58, 5)
    assert t.exit_active_at == et(TUE, 9, 30, 3)
    assert (t.exit_bar_ts, t.exit_px) == (et(TUE, 9, 31), 99.3)


def test_an_exit_decided_overnight_works_at_the_next_open() -> None:
    mon = story("AAA", MON, downgrade(MON))
    tue = story("AAA", TUE, Item(et(MON, 18, 0), "offering", structural=True))
    assert start_time(tue) is None  # never NSN: it only carries news
    pd = path(
        mon.story_id,
        "AAA",
        [MON, TUE, WED],
        [session_bars(MON), session_bars(TUE), session_bars(WED)],
    )
    out = run_one(mon, pd, toy={"sessions": 3, "target": 1.0}, news_from=[mon, tue])
    t = out.trade
    assert t is not None
    assert t.exit_reason == "abort"
    assert t.exit_decided_at == et(MON, 18, 0) and t.exit_active_at == et(TUE, 9, 30, 3)
    assert t.exit_bar_ts == et(TUE, 9, 31)


# ── admission ──


def _closed(rec: Recorder) -> list[tuple[str, str]]:
    return [(e.status, e.reason) for e in rec.made[0].seen if isinstance(e, OrderClosedIn)]


def test_buys_fail_closed_on_the_screen() -> None:
    st = story("AAA", MON, downgrade(MON))
    pd = path(st.story_id, "AAA", [MON], [session_bars(MON)])
    for verdict, why in (
        ("not_halal", "not_halal"),
        ("no_screen", "no_screen"),
        ("doubtful", "not_halal"),
    ):
        rec = Recorder()
        out = run_one(st, pd, ctx=Context(verdicts={("AAA", MON): verdict}), factory=rec)
        assert _closed(rec) == [("rejected", why)]
        assert out.trade is None and out.entry_decided_at is None
        assert (out.terminal_state, out.reason) == ("EXPIRED", "entry_unfilled")


def test_buys_outside_the_entry_window_or_without_facts_are_rejected() -> None:
    st = story("AAA", MON, downgrade(MON))
    pd = path(st.story_id, "AAA", [MON], [session_bars(MON)])
    early = Recorder(entry_bar=(9, 40))  # visible 09:41:05, before 09:50
    run_one(st, pd, factory=early)
    assert _closed(early) == [("rejected", "outside_entry_window")]

    class NoFacts:
        name, version, path_sessions = "nofacts", "1", 1

        def __init__(self) -> None:
            self.seen: list[Input] = []
            self._s = "WATCHING"

        def state(self) -> str:
            return self._s

        def start(self, ctx: Ctx) -> list[Intent]:
            return [Transition("WATCHING")]

        def on(self, ev: Input, ctx: Ctx) -> list[Intent]:
            self.seen.append(ev)
            if isinstance(ev, SessionIn) and ev.kind == "entry_start":
                return [Submit("buy"), Submit("sell")]
            if isinstance(ev, OrderClosedIn) and ev.side == "sell":
                self._s = "DONE"
                return [Finish("x")]
            return []

    pb = NoFacts()
    simulate_symbol(
        "AAA", [st], lambda s: pb, {st.story_id: pd}, spy_data([MON]), Context(), SimConfig()
    )
    assert [(e.side, e.reason) for e in pb.seen if isinstance(e, OrderClosedIn)] == [
        ("buy", "no_facts"),
        ("sell", "no_position"),
    ]


def test_timers_and_cancels_reach_the_playbook() -> None:
    st = story("AAA", MON, downgrade(MON))
    pd = path(st.story_id, "AAA", [MON], [session_bars(MON, last=(10, 0))])

    class Timed:
        name, version, path_sessions = "timed", "1", 1

        def __init__(self) -> None:
            self.seen: list[Input] = []
            self._s = "WATCHING"

        def state(self) -> str:
            return self._s

        def start(self, ctx: Ctx) -> list[Intent]:
            return [SetTimer(et(MON, 11, 0), "t1")]

        def on(self, ev: Input, ctx: Ctx) -> list[Intent]:
            self.seen.append(ev)
            if isinstance(ev, TimerIn):
                return [Submit("buy", facts=FACTS, tag="b"), Cancel(tag="b")]
            if isinstance(ev, OrderClosedIn):
                self._s = "EXPIRED"
                return [Finish("cancelled")]
            return []

    pb = Timed()
    out = simulate_symbol(
        "AAA", [st], lambda s: pb, {st.story_id: pd}, spy_data([MON]), Context(), SimConfig()
    )
    timers = [e for e in pb.seen if isinstance(e, TimerIn)]
    assert [(e.at, e.tag) for e in timers] == [(et(MON, 11, 0), "t1")]
    assert [(e.status, e.reason, e.at) for e in pb.seen if isinstance(e, OrderClosedIn)] == [
        ("cancelled", "cancel", et(MON, 11, 0))
    ]
    assert out[0].reason == "cancelled" and out[0].trade is None


def test_finish_while_holding_is_flattened_by_the_simulator() -> None:
    st = story("AAA", MON, downgrade(MON))
    pd = path(st.story_id, "AAA", [MON], [session_bars(MON)])
    out = run_one(st, pd, toy={"finish_after_entry": True})
    t = out.trade
    assert t is not None
    assert t.exit_reason == "time_stop"
    assert t.exit_decided_at == et(MON, 9, 52, 1)  # at the FillIn: 09:51 bar end + 1 s
    assert t.exit_bar_ts == et(MON, 9, 53)


# ── overlap ──


def test_a_story_starting_while_one_is_live_is_blocked_and_its_news_aborts_the_live_one() -> None:
    mon = story("AAA", MON, downgrade(MON))
    tue = story("AAA", TUE, downgrade(TUE), Item(et(TUE, 11, 0), "offering", structural=True))
    days = [MON, TUE, WED]
    bars = [session_bars(d) for d in days]
    paths = {
        mon.story_id: path(mon.story_id, "AAA", days, bars),
        tue.story_id: path(
            tue.story_id,
            "AAA",
            [TUE, WED, date_after(WED)],
            [*bars[1:], session_bars(date_after(WED))],
        ),
    }
    rec = Recorder(sessions=3, target=1.0)
    out = simulate_symbol(
        "AAA", [mon, tue], rec, paths, spy_data([*days, date_after(WED)]), Context(), SimConfig()
    )
    first, second = out
    assert (second.terminal_state, second.reason, second.skip) == (
        "DISMISSED",
        "blocked_open",
        None,
    )
    assert second.start_at == et(TUE, 9, 30)
    t = first.trade
    assert t is not None and t.exit_reason == "abort"
    assert t.exit_decided_at == et(TUE, 11, 0) and t.exit_bar_ts == et(TUE, 11, 1)
    news = [e for e in rec.made[0].seen if isinstance(e, NewsIn)]
    assert [(e.at, e.story.story_id, e.own) for e in news] == [
        (et(TUE, 8, 0), tue.story_id, False),
        (et(TUE, 11, 0), tue.story_id, False),
    ]
    assert len(rec.made) == 1  # the blocked story never got a playbook


def date_after(d):  # type: ignore[no-untyped-def]
    from halal_trader.market_hours import next_trading_day

    return next_trading_day(d)


def test_stage_a_stops_at_the_entry_and_full_hold_blocks_the_symbol() -> None:
    mon = story("AAA", MON, downgrade(MON))
    tue = story("AAA", TUE, downgrade(TUE))
    thu = date_after(WED)
    paths = {
        mon.story_id: path(
            mon.story_id, "AAA", [MON, TUE, WED], [session_bars(d) for d in (MON, TUE, WED)]
        ),
        tue.story_id: path(
            tue.story_id, "AAA", [TUE, WED, thu], [session_bars(d) for d in (TUE, WED, thu)]
        ),
    }
    spy = spy_data([MON, TUE, WED, thu])

    def go(full: bool):  # type: ignore[no-untyped-def]
        return simulate_symbol(
            "AAA",
            [mon, tue],
            Recorder(sessions=3, target=1.0),
            paths,
            spy,
            Context(),
            SimConfig(),
            stop_at="entry",
            assume_full_hold=full,
        )

    held = go(True)
    assert held[0].trade is None and held[0].terminal_state == "ENTERED"
    assert held[0].entry_bar_ts == et(MON, 9, 51)
    assert held[1].reason == "blocked_open"
    free = go(False)
    assert free[1].terminal_state == "ENTERED" and free[1].entry_bar_ts == et(TUE, 9, 51)


def test_a_story_without_a_path_is_skipped_with_the_loader_reason() -> None:
    st = story("AAA", MON, downgrade(MON))
    out = run_one(st, PathSkip(st.story_id, "adjust_defect"))
    assert (out.terminal_state, out.reason, out.skip) == (
        "SKIPPED",
        "adjust_defect",
        "adjust_defect",
    )
    assert out.start_at == et(MON, 9, 30)


def test_stories_that_never_become_nsn_do_not_start() -> None:
    other = story("AAA", MON, Item(et(MON, 8, 0), "product"))
    late = story("AAA", MON, Item(et(MON, 15, 30)))  # after the 15:00 cutoff
    in_session = story("AAA", MON, Item(et(MON, 11, 17)))
    assert start_time(other) is None and start_time(late) is None
    assert start_time(in_session) == et(MON, 11, 17)


# ── prices across sessions ──


def test_a_split_inside_the_path_is_not_a_price_move() -> None:
    """2:1 split effective TUE: raw prices halve, A(MON) = 0.5, A(TUE) = 1."""
    st = story("AAA", MON, downgrade(MON))
    tue_rows = {
        (11, 0): (50.4, 50.7, 50.3, 50.6, 2_000.0, 50.5),  # 101.2 in S units: the target
        (11, 2): (50.5, 50.6, 50.5, 50.55, 2_000.0, 50.55),
    }
    pd = path(
        st.story_id,
        "AAA",
        [MON, TUE, WED],
        [
            session_bars(MON),
            session_bars(TUE, price=50.0, rows=tue_rows),
            session_bars(WED, price=50.0),
        ],
    )
    ctx = Context(
        adj={("AAA", MON): 0.5}, daily={("AAA", MON): daily(100.0), ("SPY", MON): daily(200.0)}
    )

    seen_units: list[float] = []

    class Probe(Recorder):
        def __call__(self, story):  # type: ignore[no-untyped-def]
            toy = super().__call__(story)
            orig = toy.on

            def on(ev, ctx):  # type: ignore[no-untyped-def]
                if isinstance(ev, SessionIn) and ev.kind == "open" and ev.k == 1:
                    seen_units.append(ctx.market.to_s_units(50.0, TUE))
                return orig(ev, ctx)

            toy.on = on  # type: ignore[method-assign]
            return toy

    out = run_one(st, pd, ctx=ctx, factory=Probe(sessions=3, target=0.01))
    assert seen_units == [100.0]
    t = out.trade
    assert t is not None
    assert t.exit_reason == "target"  # no -50% "drop", no stop: the target at 101.2 S-units
    assert t.exit_decided_at == et(TUE, 11, 1, 5) and t.exit_px == 50.55
    assert (t.adj_entry, t.adj_exit) == (0.5, 1.0)
    assert abs(t.r_gross - (50.55 * 1.0 / (100.0 * 0.5) - 1)) < EPS
    assert abs(t.mae) < EPS  # the S-unit low never went below the entry
    mon_leg, tue_leg = out.legs
    assert (mon_leg.a, mon_leg.z) == (100.0, 100.0)
    assert abs(tue_leg.z - 101.1) < EPS


def test_an_ex_dividend_inside_the_path_accrues_to_the_holder() -> None:
    """$1 dividend going ex TUE: raw TUE opens at 99, A(MON) = 0.99."""
    st = story("AAA", MON, downgrade(MON))
    pd = path(
        st.story_id,
        "AAA",
        [MON, TUE, WED],
        [session_bars(MON), session_bars(TUE, price=99.0), session_bars(WED, price=99.0)],
    )
    ctx = Context(adj={("AAA", MON): 0.99})
    out = run_one(st, pd, ctx=ctx, toy={"sessions": 3, "target": 0.5, "sell_at": (1, 12, 0)})
    t = out.trade
    assert t is not None
    assert t.exit_reason == "stop" and t.exit_px == 99.0 and t.exit_bar_ts == et(TUE, 12, 2)
    assert abs(t.r_gross) < EPS  # 99 * 1 / (100 * 0.99) - 1: the dividend made up the gap


def test_spy_identity_a_stock_that_is_spy_returns_minus_two_costs() -> None:
    rng = np.random.default_rng(7)
    spy_bars = session_bars(MON, price=200.0)
    walk = 200.0 * np.exp(np.cumsum(rng.normal(0, 0.001, len(spy_bars))))
    from halal_trader.data.minutes import BarArrays

    spy_bars = BarArrays(
        spy_bars.ts, walk, walk * 1.001, walk * 0.999, walk, spy_bars.v, walk * 1.0002
    )
    spy = spy_data([MON])
    spy.days[MON] = spy_bars
    st = story("AAA", MON, downgrade(MON))
    facts = TradeFacts("x", "c", "ID", cost_bps=7.0, rank=10, tech=True, beta=1.0)
    out = run_one(
        st,
        path(st.story_id, "AAA", [MON], [spy_bars]),
        spy=spy,
        toy={"target": 1.0, "facts": facts},
    )
    t = out.trade
    assert t is not None
    assert t.entry_px == t.spy_entry_px and t.exit_px == t.spy_exit_px
    assert abs(t.r_gross - t.r_spy) < EPS
    assert abs(t.r_net_abn - (-0.0014)) < EPS


def test_surcharge_and_cost_multipliers() -> None:
    st = story("AAA", MON, downgrade(MON))
    pd = path(st.story_id, "AAA", [MON], [session_bars(MON)])
    base = run_one(st, pd, toy=UNREACHABLE).trade
    x2 = run_one(st, pd, toy=UNREACHABLE, cfg=SimConfig(cost="study_x2")).trade
    sur = run_one(st, pd, toy=UNREACHABLE, cfg=SimConfig(cost="surcharge")).trade
    assert base is not None and x2 is not None and sur is not None
    assert abs((base.r_net_abn - x2.r_net_abn) - 0.003) < EPS
    # rank 500: half-spread 10 bps (doubled? the 09:51 entry is 21 min after the first bar: no)
    # plus 5 bps impact each side, no ADV known
    assert sur.cost_bps == 15.0
    assert math.isclose(sur.r_net_abn, base.r_net_abn, abs_tol=EPS)


def test_state_carries_between_calls() -> None:
    mon = story("AAA", MON, downgrade(MON))
    tue = story("AAA", TUE, downgrade(TUE))
    days = [MON, TUE, WED]
    state = SymbolState()
    rec = Recorder(sessions=3, target=1.0)
    p = path(mon.story_id, "AAA", days, [session_bars(d) for d in days])
    simulate_symbol(
        "AAA",
        [mon],
        rec,
        {mon.story_id: p},
        spy_data(days),
        Context(),
        SimConfig(),
        news_from=[mon, tue],
        state=state,
    )
    assert state.busy_until_us > 0 and state.held
    out = simulate_symbol(
        "AAA", [tue], rec, {}, spy_data(days), Context(), SimConfig(), state=state
    )
    assert out[0].reason == "blocked_open"
    assert from_us(state.busy_until_us) > et(TUE, 9, 30)


def test_a_partial_exit_is_refused_and_a_whole_one_fills() -> None:
    st = story("AAA", MON, downgrade(MON))
    pd = path(st.story_id, "AAA", [MON], [session_bars(MON)])

    class Halves(Toy):
        def on(self, ev: Input, ctx: Ctx) -> list[Intent]:
            self.seen.append(ev)
            if isinstance(ev, FillIn) and ev.side == "buy":
                self._state = "ENTERED"
                return [Submit("sell", qty=ev.qty / 2, reason="stop")]
            if isinstance(ev, OrderClosedIn) and ev.side == "sell":
                return [Submit("sell", qty=1e9, reason="stop")]  # more than held: clamped
            if isinstance(ev, FillIn):
                self._state = "EXITED"
                return [Finish("stop")]
            return super().on(ev, ctx) if self._state == "WATCHING" else []

    rec: list[Halves] = []

    def make(s):  # type: ignore[no-untyped-def]
        rec.append(Halves(s))
        return rec[-1]

    (out,) = simulate_symbol(
        "AAA", [st], make, {st.story_id: pd}, spy_data([MON]), Context(), SimConfig()
    )
    closed = [(e.status, e.reason) for e in rec[0].seen if isinstance(e, OrderClosedIn)]
    assert closed == [("rejected", "partial_exit")]
    t = out.trade
    assert t is not None and t.exit_reason == "stop"  # the oversized sell was clamped to the whole


def test_a_path_for_another_session_is_refused() -> None:
    st = story("AAA", MON, downgrade(MON))
    wrong = path(st.story_id, "AAA", [TUE], [session_bars(TUE)])
    try:
        run_one(st, wrong)
    except ValueError as e:
        assert "the path is AAA from 2016-03-08" in str(e)
    else:  # pragma: no cover
        raise AssertionError("a path from another session was simulated")


def test_a_playbook_sees_the_story_only_as_of_now() -> None:
    st = story("AAA", MON, Item(et(MON, 10, 0)), Item(et(MON, 11, 0), "offering", structural=True))
    clock = {"now": et(MON, 10, 30)}
    view = PitStory(st, lambda: clock["now"])
    assert (view.story_id, view.symbol, view.session) == (st.story_id, "AAA", MON)
    assert view.card_at(et(MON, 12, 0)) == st.card_at(et(MON, 10, 30))  # not the 11:00 abort
    assert not view.card_at(et(MON, 12, 0)).structural and st.card_at(et(MON, 12, 0)).structural
    assert view.news_times() == [et(MON, 10, 0)]
    assert view.nsn_at(et(MON, 15, 0)) == et(MON, 10, 0)
    assert view.at_news() == st.at_news() and view.start_case() == "in"
    clock["now"] = et(MON, 9, 0)
    assert view.nsn_at(et(MON, 15, 0)) is None and view.at_news() is None
    assert view.start_case() == "out" and view.news_times() == []

    seen: list[object] = []
    pd = path(st.story_id, "AAA", [MON], [session_bars(MON)])

    class Look(Toy):
        def start(self, ctx: Ctx) -> list[Intent]:
            seen.append(ctx.story)
            return super().start(ctx)

    simulate_symbol(
        "AAA", [st], lambda s: Look(s), {st.story_id: pd}, spy_data([MON]), Context(), SimConfig()
    )
    assert isinstance(seen[0], PitStory)
