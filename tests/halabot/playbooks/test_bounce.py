"""The overreaction bounce in the simulator: spec §E.1's known answers for the playbook.

Every scenario is built by hand on synthetic minute bars; entry and exit
times, bars and prices are exact, and returns are checked to 1e-12 against
the spec §G.6 formula. Stories are labelled by the real taxonomy
(``_bounce.TStory``). The base path and the "in" path are described in
``_bounce``'s docstring.
"""

from __future__ import annotations

import pickle
from datetime import date, datetime, timedelta

import pytest

from halabot.playbooks.bounce import (
    CUTOFF_TIMER,
    FAMILY,
    NAME,
    VERSION,
    BounceFactory,
    BounceParams,
    BounceState,
    OverreactionBounce,
)
from halabot.playbooks.clock import SIP_DELAYED
from halabot.playbooks.playbook import PlaybookFactory
from halabot.playbooks.records import StoryOutcome, TradeRecord
from halabot.playbooks.types import SetTimer, SimConfig, SpyData, Submit
from halal_trader.db.repos.quant_trials import config_hash
from tests.halabot.playbooks._bounce import (
    BASE_LEVELS,
    BASE_ROWS,
    EPS,
    IN_LEVELS,
    IN_ROWS,
    IN_SPY,
    SPY_ROWS,
    A,
    AtlasStory,
    TStory,
    bars,
    base_bars,
    base_story,
    eligibility,
    late_reclaim,
    pre_event,
    r_expected,
    run,
    spy,
    titems,
    transitions,
    tstory,
    tue_story,
)
from tests.halabot.playbooks._support import (
    HALF,
    MON,
    THU,
    TUE,
    WED,
    Context,
    Row,
    daily,
    et,
    path,
)

LAG = timedelta(seconds=3)
ENTRY = (et(MON, 10, 6, 5), et(MON, 10, 7), 95.45)
OPEN = et(MON, 9, 30)


def nxt(hm: tuple[int, int], n: int = 1) -> tuple[int, int]:
    """The minute ``n`` after ``hm``."""
    m = hm[0] * 60 + hm[1] + n
    return m // 60, m % 60


def one(
    st: TStory,
    days: list[date],
    day_bars: list,  # type: ignore[type-arg]
    *,
    spy_rows: dict[date, dict[tuple[int, int], Row]] | None = None,
    spy_price: float = 200.0,
    pre=None,  # type: ignore[no-untyped-def]
    elig=None,  # type: ignore[no-untyped-def]
    params: BounceParams | None = None,
    ctx: Context | None = None,
    cfg: SimConfig | None = None,
) -> StoryOutcome:
    pd = path(st.story_id, A, days, day_bars)
    if spy_rows is None:
        spy_rows = {MON: SPY_ROWS}
    context = {st.story_id: (pre or pre_event(days[0]), elig or eligibility())}
    params = params or BounceParams(hold_sessions=len(days))
    sp = spy(days, spy_rows, price=spy_price)
    (out,) = run([st], {st.story_id: pd}, sp, context, ctx=ctx, params=params, cfg=cfg)
    return out


def check_trade(
    t: TradeRecord | None,
    *,
    entry: tuple[datetime, datetime, float],
    exit_: tuple[datetime, datetime | None, float],
    spy_px: tuple[float, float],
    reason: str,
    adj: tuple[float, float] = (1.0, 1.0),
    flags: tuple[str, ...] = (),
) -> TradeRecord:
    assert t is not None
    (e_dec, e_bar, e_px), (x_dec, x_bar, x_px) = entry, exit_
    assert (t.entry_decided_at, t.entry_active_at, t.entry_bar_ts, t.entry_px) == (
        e_dec,
        e_dec + LAG,
        e_bar,
        e_px,
    )
    assert (t.exit_decided_at, t.exit_bar_ts, t.exit_px) == (x_dec, x_bar, x_px)
    assert (t.spy_entry_px, t.spy_exit_px) == spy_px
    assert (t.adj_entry, t.adj_exit) == adj
    assert t.exit_reason == reason
    assert t.flags == flags
    g, s, n, b = r_expected(e_px, x_px, *spy_px, a_entry=adj[0], a_exit=adj[1])
    assert abs(t.r_gross - g) < EPS
    assert abs(t.r_spy - s) < EPS
    assert abs(t.r_net_abn - n) < EPS
    assert abs(t.r_beta_adj - b) < EPS
    return t


def check_facts(
    t: TradeRecord,
    *,
    p0: float = 100.0,
    spy0: float = 200.0,
    sigma: float = 0.018,
    thr: float = 0.036,
    low: float = 94.0,
    target: float = 97.0,
    anchor: datetime = OPEN,
    variant: str = "ID",
) -> None:
    assert (t.p0, t.spy0, t.sigma, t.thr, t.low_star, t.target) == (
        p0,
        spy0,
        sigma,
        thr,
        low,
        target,
    )
    assert t.anchor_ts == anchor
    assert (t.family_type, t.cell, t.variant) == (
        "analyst_downgrade",
        f"NSN_CORE/{variant}",
        variant,
    )
    assert (t.cost_bps, t.rank, t.tech, t.beta) == (15.0, 500, False, 1.2)


# ── the rule's constants and the factory ──


# Spec §G.14's PREREG["playbook"], literally.
PREREG_PLAYBOOK: dict[str, object] = {
    "k_sigma": 2.0,
    "floor": 0.03,
    "quiet_min": 20,
    "entry_start_min": 20,
    "entry_cutoff_min": 60,
    "max_retrace_at_entry": 0.25,
    "target_retrace": 0.5,
    "market_break": -0.02,
    "flatten_min": 5,
    "reclaim": "close > AVWAP from anchor",
    "stop": "bar close < L*",
    "abort": "structural item",
    "priority": ["abort", "stop", "target"],
    "compliance_exit": True,
}


def test_params_are_the_spec_constants() -> None:
    p = BounceParams()
    config = p.as_config()
    assert config == PREREG_PLAYBOOK
    # Equal is not enough for the hash: 20 == 20.0, but json.dumps writes them apart.
    assert [type(v) for v in config.values()] == [type(v) for v in PREREG_PLAYBOOK.values()]
    assert list(config) == list(PREREG_PLAYBOOK)
    assert config_hash({"playbook": config}) == config_hash({"playbook": PREREG_PLAYBOOK})
    # The hold is the cell's and the family switch the atlas's: neither is in the block.
    for other in (BounceParams(hold_sessions=3), BounceParams(require_family=False)):
        assert other.as_config() == config
    assert p.run_config() == {"hold_sessions": 1, "variant": "ID", "require_family": True}
    assert BounceParams(hold_sessions=3, require_family=False).run_config() == {
        "hold_sessions": 3,
        "variant": "MD3",
        "require_family": False,
    }
    assert BounceParams(quiet=timedelta(seconds=90)).as_config()["quiet_min"] == 1.5
    assert (p.variant, BounceParams(hold_sessions=3).variant) == ("ID", "MD3")
    with pytest.raises(ValueError):
        BounceParams(flatten_before_close=timedelta(minutes=10))
    with pytest.raises(ValueError):
        BounceParams(entry_start_after_open=timedelta(minutes=10))
    with pytest.raises(ValueError):
        BounceParams(hold_sessions=0)


def test_the_factory_is_a_picklable_playbook_factory() -> None:
    st = base_story()
    fac = BounceFactory(
        {st.story_id: (pre_event(MON), eligibility())}, BounceParams(hold_sessions=3)
    )
    typed: PlaybookFactory = fac
    assert (typed.name, typed.version, typed.path_sessions) == (NAME, VERSION, 3)
    pb = fac(st)
    assert isinstance(pb, OverreactionBounce)
    assert (pb.path_sessions, pb.state(), pb.live()) == (3, "DETECTED", False)
    again = pickle.loads(pickle.dumps(fac))
    assert again.params == fac.params and again(st).path_sessions == 3
    with pytest.raises(KeyError, match="no pre-event context"):
        fac(base_story(TUE))


# ── spec §E.1, the playbook's known answers ──


def test_target() -> None:
    out = one(base_story(), [MON], [base_bars()])
    t = check_trade(
        out.trade,
        entry=ENTRY,
        exit_=(et(MON, 10, 31, 5), et(MON, 10, 32), 97.1),
        spy_px=(200.5, 201.0),
        reason="target",
    )
    check_facts(t)
    assert t.start_case == "out" and t.at_news == et(MON, 7, 0) and t.nsn_at == et(MON, 7, 10)
    assert (out.terminal_state, out.reason) == ("EXITED", "target")
    assert out.start_at == et(MON, 9, 30)
    assert (out.triggered_at, out.armed_at) == (et(MON, 9, 31, 5), et(MON, 10, 6, 5))
    assert transitions(out) == [
        (et(MON, 9, 30), "WATCHING", "detected"),
        (et(MON, 9, 31, 5), "WATCHING", "triggered"),
        (et(MON, 10, 6, 5), "ARMED", "quiet"),
        (et(MON, 10, 6, 5), "ENTERING", "entry"),
        (et(MON, 10, 8, 1), "ENTERED", "filled"),
        (et(MON, 10, 31, 5), "EXITING", "target"),
        (et(MON, 10, 33, 1), "EXITED", "target"),
    ]
    # The intents a run keeps: the cutoff timer at the start, the buy with its facts, the sell.
    kinds = [type(i).__name__ for _, i in out.intents]
    assert kinds.count("Submit") == 2 and kinds.count("Finish") == 1
    timer = next(i for _, i in out.intents if isinstance(i, SetTimer))
    assert timer == SetTimer(et(MON, 15, 0), CUTOFF_TIMER)
    buy = next(i for _, i in out.intents if isinstance(i, Submit) and i.side == "buy")
    assert buy.facts is not None and buy.facts.low_star == 94.0 and buy.facts.target == 97.0


@pytest.mark.parametrize("reason", ["ok", "rank"])
def test_each_transition_starts_where_the_last_ended(reason: str) -> None:
    """The simulator labels each transition with the state before it: the first from DETECTED
    (it reads the state before calling ``start``, which already leaves it)."""
    out = one(base_story(), [MON], [base_bars()], elig=eligibility(reason=reason))
    froms = [f for _, f, _, _ in out.transitions]
    tos = [t for _, _, t, _ in out.transitions]
    assert froms[0] == "DETECTED"
    assert froms[1:] == tos[:-1]
    if reason != "ok":
        assert out.transitions == ((et(MON, 9, 30), "DETECTED", "DISMISSED", reason),)


def test_stop() -> None:
    rows: dict[tuple[int, int], Row] = {
        (10, 20): (96.0, 96.0, 93.8, 93.9, 1_000.0, 94.0),  # closes below L* = 94
        (10, 22): (93.9, 94.0, 93.7, 93.8, 1_000.0, 93.85),
    }
    day = bars(MON, BASE_LEVELS + [((10, 20), 93.9)], BASE_ROWS | rows)
    out = one(base_story(), [MON], [day])
    check_trade(
        out.trade,
        entry=ENTRY,
        exit_=(et(MON, 10, 21, 5), et(MON, 10, 22), 93.85),
        spy_px=(200.5, 200.0),
        reason="stop",
    )
    assert (out.terminal_state, out.reason) == ("EXITED", "stop")
    assert transitions(out)[-2:] == [
        (et(MON, 10, 21, 5), "EXITING", "stop"),
        (et(MON, 10, 23, 1), "EXITED", "stop"),
    ]


def test_a_close_at_the_low_is_not_a_stop() -> None:
    """X1 is a close strictly below L*: a close of exactly 94 holds."""
    rows: dict[tuple[int, int], Row] = {(10, 20): (96.0, 96.0, 94.0, 94.0, 1_000.0, 94.5)}
    out = one(base_story(), [MON], [base_bars(rows=rows)])
    assert out.trade is not None and out.trade.exit_reason == "target"


def test_no_reclaim_expires_at_the_cutoff() -> None:
    """Armed at 10:06:05, the price never closes above its AVWAP: expired at 15:00 sharp."""
    day = bars(
        MON, [((9, 30), 96.0), ((9, 46), 94.8)], {k: BASE_ROWS[k] for k in [(9, 30), (9, 45)]}
    )
    out = one(base_story(), [MON], [day])
    assert out.trade is None and out.entry_decided_at is None
    assert (out.terminal_state, out.reason) == ("EXPIRED", "cutoff")
    assert transitions(out) == [
        (et(MON, 9, 30), "WATCHING", "detected"),
        (et(MON, 9, 31, 5), "WATCHING", "triggered"),
        (et(MON, 10, 6, 5), "ARMED", "quiet"),
        (et(MON, 15, 0), "EXPIRED", "cutoff"),
    ]


def test_a_new_low_disarms_and_the_quiet_clock_restarts() -> None:
    rows: dict[tuple[int, int], Row] = {
        (10, 5): (94.8, 95.1, 94.7, 95.0, 1_000.0, 94.9),  # armed; 95.0 < AVWAP 95.32
        (10, 10): (94.8, 94.8, 93.9, 94.2, 1_000.0, 94.3),  # a new low: L = 93.9
        (10, 30): (94.5, 95.4, 94.4, 95.3, 1_000.0, 95.1),  # 20 min later: re-armed, reclaimed
        (10, 32): (95.3, 95.5, 95.2, 95.4, 1_000.0, 95.35),
    }
    levels = [((9, 30), 96.0), ((9, 46), 94.8), ((10, 11), 94.5), ((10, 31), 95.3)]
    opening = {k: BASE_ROWS[k] for k in [(9, 30), (9, 45)]}
    out = one(base_story(), [MON], [bars(MON, levels, opening | rows)])
    assert transitions(out)[:6] == [
        (et(MON, 9, 30), "WATCHING", "detected"),
        (et(MON, 9, 31, 5), "WATCHING", "triggered"),
        (et(MON, 10, 6, 5), "ARMED", "quiet"),
        (et(MON, 10, 11, 5), "WATCHING", "new_low"),
        (et(MON, 10, 31, 5), "ARMED", "quiet"),
        (et(MON, 10, 31, 5), "ENTERING", "entry"),
    ]
    assert out.armed_at == et(MON, 10, 6, 5)  # the first arming
    t = out.trade
    assert t is not None and t.entry_bar_ts == et(MON, 10, 32) and t.entry_px == 95.35
    assert t.low_star == 93.9 and t.target == 93.9 + 0.5 * (100.0 - 93.9)
    assert t.exit_reason == "time_stop"


def test_an_equal_low_restarts_the_quiet_clock() -> None:
    """t_L is the latest bar at the low: 09:55 touching 94 again moves the arming to 10:15."""
    rows: dict[tuple[int, int], Row] = {
        (9, 55): (94.8, 94.8, 94.0, 94.8, 1_000.0, 94.5),
        (10, 15): (94.8, 95.5, 94.7, 95.4, 1_000.0, 95.2),
        (10, 17): (95.4, 95.6, 95.3, 95.5, 1_000.0, 95.45),
    }
    opening = {k: BASE_ROWS[k] for k in [(9, 30), (9, 45), (10, 5)]}
    levels = [((9, 30), 96.0), ((9, 46), 94.8), ((10, 16), 95.4)]
    out = one(base_story(), [MON], [bars(MON, levels, opening | rows)])
    assert out.armed_at == et(MON, 10, 16, 5) and out.entry_decided_at == et(MON, 10, 16, 5)
    assert out.trade is not None and out.trade.entry_bar_ts == et(MON, 10, 17)


def test_a_drop_of_exactly_thr_triggers() -> None:
    """σ = 1/64, so thr = 2σ = 0.03125, and 96.875/100 - 1 = -0.03125 exactly: D <= -thr."""
    rows: dict[tuple[int, int], Row] = {(9, 30): (97.0, 97.0, 96.875, 96.875, 1_000.0, 96.9)}
    out = one(
        base_story(),
        [MON],
        [bars(MON, [((9, 30), 97.0)], rows)],
        pre=pre_event(MON, sigma=1 / 64),
    )
    assert out.triggered_at == et(MON, 9, 31, 5)


def test_a_close_equal_to_the_avwap_is_not_a_reclaim() -> None:
    """Every weight from the anchor is 95, so AVWAP = 95 exactly: the 09:50 bar's close of 95
    arms (L = 94 at 09:30) but does not enter; 10:30's 95.1 does."""
    rows: dict[tuple[int, int], Row] = {
        (9, 30): (96.0, 96.0, 94.0, 95.0, 1_000.0, 95.0),
        (10, 30): (95.0, 95.2, 95.0, 95.1, 1_000.0, 95.05),
    }
    out = one(base_story(), [MON], [bars(MON, [((9, 30), 95.0), ((10, 31), 95.1)], rows)])
    assert out.armed_at == et(MON, 9, 51, 5)
    assert out.entry_decided_at == et(MON, 10, 31, 5)


@pytest.mark.parametrize(("spy_close", "breaks"), [(196.0, True), (196.01, False)])
def test_market_break(spy_close: float, breaks: bool) -> None:
    """A SPY close of S at or below 0.98 x 200 = 196 expires a watching story."""
    rows = SPY_ROWS | {(9, 55): (200.0, 200.0, spy_close, spy_close, 1e5, 198.0)}
    out = one(base_story(), [MON], [base_bars()], spy_rows={MON: rows})
    if breaks:
        assert out.trade is None
        assert (out.terminal_state, out.reason) == ("EXPIRED", "market_break")
        assert transitions(out)[-1] == (et(MON, 9, 56, 5), "EXPIRED", "market_break")
    else:
        assert out.trade is not None and out.trade.exit_reason == "target"


def test_a_market_break_before_the_start_expires_at_the_start() -> None:
    st = tstory(A, MON, (et(MON, 9, 50), "analyst_downgrade"))  # "in", starts 10:00
    rows = SPY_ROWS | {(9, 40): (200.0, 200.0, 195.0, 195.5, 1e5, 197.0)}
    out = one(st, [MON], [base_bars()], spy_rows={MON: rows})
    assert transitions(out) == [
        (et(MON, 10, 0), "WATCHING", "detected"),
        (et(MON, 10, 0), "EXPIRED", "market_break"),
    ]


def test_structural_abort() -> None:
    """A dilution item public at 10:05:30, usable at 10:15:30, aborts the position."""
    st = base_story(MON, (et(MON, 10, 5, 30), "dilution"))
    out = one(st, [MON], [base_bars()])
    check_trade(
        out.trade,
        entry=ENTRY,
        exit_=(et(MON, 10, 15, 30), et(MON, 10, 16), 96.0),
        spy_px=(200.5, 200.0),
        reason="abort",
    )
    assert transitions(out)[-2:] == [
        (et(MON, 10, 15, 30), "EXITING", "abort"),
        (et(MON, 10, 17, 1), "EXITED", "abort"),
    ]


@pytest.mark.parametrize("itype", ["legal_adverse", "dilution"])
def test_a_veto_before_entry(itype: str) -> None:
    """An unclear negative (legal_adverse) or a structural one (dilution) usable at 10:00:30
    vetoes the armed-to-be story (spec §B.1: both veto before the entry)."""
    st = base_story(MON, (et(MON, 9, 50, 30), itype))
    out = one(st, [MON], [base_bars()])
    assert out.trade is None and out.entry_decided_at is None
    assert (out.terminal_state, out.reason) == ("EXPIRED", "veto")
    assert (out.triggered_at, out.armed_at) == (et(MON, 9, 31, 5), None)
    assert transitions(out)[-1] == (et(MON, 10, 0, 30), "EXPIRED", "veto")


def test_an_unclear_item_after_entry_does_not_abort() -> None:
    st = base_story(MON, (et(MON, 10, 10), "legal_adverse"))
    out = one(st, [MON], [base_bars()])
    assert out.trade is not None and out.trade.exit_reason == "target"


def test_a_veto_before_the_open_expires_at_the_start() -> None:
    st = base_story(MON, (et(MON, 8, 0), "management_exit"))
    out = one(st, [MON], [base_bars()])
    assert transitions(out) == [
        (et(MON, 9, 30), "WATCHING", "detected"),
        (et(MON, 9, 30), "EXPIRED", "veto"),
    ]


@pytest.mark.parametrize("parent", [False, True])
def test_detection_when_nsn_arrives_late(parent: bool) -> None:
    """A price-target cut at 08:00 is not NSN; a downgrade public at 10:50 makes it NSN at 11:00.

    With ``parent`` the story also follows an earlier one until then (rule (b):
    every item known is reactive), and turns non-follower with the downgrade.
    """
    st = tstory(
        A,
        MON,
        (et(MON, 8, 0), "analyst_pt_cut"),
        (et(MON, 10, 50), "analyst_downgrade"),
        parent=parent,
    )
    assert st.card_at(et(MON, 10, 59)).family is None
    assert st.card_at(et(MON, 10, 59)).follower is parent
    assert st.card_at(et(MON, 11, 0)).family == FAMILY and not st.card_at(et(MON, 11, 0)).follower
    out = one(
        st,
        [MON],
        [bars(MON, IN_LEVELS, IN_ROWS)],
        spy_rows={MON: IN_SPY},
        spy_price=201.0,
        pre=pre_event(MON, sigma=0.01),
        params=BounceParams(),
    )
    # Started at the downgrade's availability; the history to 10:58 shows the trigger.
    assert out.start_at == et(MON, 11, 0) and out.nsn_at == et(MON, 11, 0)
    assert transitions(out)[:5] == [
        (et(MON, 11, 0), "WATCHING", "detected"),
        (et(MON, 11, 0), "WATCHING", "triggered"),
        (et(MON, 11, 16, 5), "ARMED", "quiet"),
        (et(MON, 11, 16, 5), "ENTERING", "entry"),
        (et(MON, 11, 18, 1), "ENTERED", "filled"),
    ]
    t = check_trade(
        out.trade,
        entry=(et(MON, 11, 16, 5), et(MON, 11, 17), 96.85),
        exit_=(et(MON, 11, 41, 5), et(MON, 11, 42), 98.25),
        spy_px=(201.0, 201.3),
        reason="target",
    )
    check_facts(
        t,
        p0=100.2,
        spy0=201.0,
        sigma=0.01,
        thr=0.03,
        low=96.0,
        target=96.0 + 0.5 * (100.2 - 96.0),
        anchor=et(MON, 10, 50),
    )
    assert (t.start_case, t.at_news, t.nsn_at) == ("in", et(MON, 10, 50), et(MON, 11, 0))


def test_in_session_news_before_any_bar_takes_the_previous_closes() -> None:
    """News public at 09:30:20: no bar of S closed by then, so P0 = 100 and SPY0 = 200."""
    st = tstory(A, MON, (et(MON, 9, 30, 20), "analyst_downgrade"))
    out = one(st, [MON], [base_bars()])
    assert out.start_at == et(MON, 9, 40, 20)
    assert out.triggered_at == et(MON, 9, 40, 20)  # from the history at the start
    t = check_trade(
        out.trade,
        entry=ENTRY,
        exit_=(et(MON, 10, 31, 5), et(MON, 10, 32), 97.1),
        spy_px=(200.5, 201.0),
        reason="target",
    )
    check_facts(t)
    assert t.start_case == "in"


def test_in_session_news_before_the_stocks_first_bar_takes_both_previous_closes() -> None:
    """News public at 09:50 on a stock that first prints at 10:00, while SPY has traded at 204
    since the open. No stock bar closed before the news, so P0 is the previous close 100, and
    SPY0 SPY's previous close 200, not its 09:49 close 204: both legs of D start from the same
    baseline. The base path, 30 minutes later: D at 10:00 = -0.04 - 0.02, entered at 10:36:05."""
    st = tstory(A, MON, (et(MON, 9, 50), "analyst_downgrade"))
    levels = [(nxt(hm, 30), price) for hm, price in BASE_LEVELS]
    rows = {nxt(hm, 30): row for hm, row in BASE_ROWS.items()}
    day = bars(MON, levels, rows, first=(10, 0))
    out = one(st, [MON], [day], spy_rows={}, spy_price=204.0)
    assert (out.start_at, out.triggered_at) == (et(MON, 10, 0), et(MON, 10, 1, 5))
    t = out.trade
    assert t is not None and t.start_case == "in"
    assert (t.entry_decided_at, t.entry_bar_ts, t.entry_px) == (
        et(MON, 10, 36, 5),
        et(MON, 10, 37),
        95.45,
    )
    check_facts(t, anchor=et(MON, 10, 0))


def test_md3_overnight_gap_through_the_low() -> None:
    """Held overnight; TUE's first bar closes at 93.2 < L*: stop decided when it is visible."""
    tue = bars(
        TUE,
        [((9, 30), 93.2)],
        {
            (9, 30): (93.5, 93.6, 93.0, 93.2, 1_000.0, 93.3),
            (9, 32): (93.2, 93.5, 93.1, 93.4, 1_000.0, 93.35),
        },
    )
    spy_rows = {MON: SPY_ROWS, TUE: {(9, 32): (200.0, 200.0, 198.9, 199.0, 1e5, 199.0)}}
    out = one(
        base_story(),
        [MON, TUE, WED],
        [base_bars(target=False), tue, bars(WED, [((9, 30), 93.4)])],
        spy_rows=spy_rows,
    )
    t = check_trade(
        out.trade,
        entry=ENTRY,
        exit_=(et(TUE, 9, 31, 5), et(TUE, 9, 32), 93.35),
        spy_px=(200.5, 199.0),
        reason="stop",
    )
    check_facts(t, variant="MD3")
    assert (t.exit_session, t.sessions_held) == (TUE, 2)


def test_ex_dividend_on_s_sets_p0_through_the_a_ratio() -> None:
    """prev_close_s = 100 x A(S-1)/A(S) = 99: the trigger, E3 and TGT all read 99, not 100."""
    levels = [((9, 30), 96.1), ((9, 46), 95.7), ((10, 6), 96.2), ((10, 8), 96.5), ((10, 31), 97.3)]
    rows: dict[tuple[int, int], Row] = {
        (9, 45): (96.1, 96.1, 95.5, 95.6, 1_000.0, 95.8),  # 95.6/99 - 1 = -0.0343: triggered
        (10, 5): (95.7, 96.3, 95.6, 96.2, 1_000.0, 96.1),
        (10, 7): (96.2, 96.4, 96.1, 96.3, 1_000.0, 96.25),
        (10, 30): (96.6, 97.4, 96.5, 97.3, 1_000.0, 97.1),  # >= TGT 97.25
        (10, 32): (97.3, 97.5, 97.2, 97.4, 1_000.0, 97.35),
    }
    day = bars(MON, levels, rows)
    out = one(base_story(), [MON], [day], pre=pre_event(MON, prev_close=99.0, sigma=0.01))
    assert out.triggered_at == et(MON, 9, 46, 5)  # 96.1/99 - 1 = -0.0293 at the open: not yet
    t = check_trade(
        out.trade,
        entry=(et(MON, 10, 6, 5), et(MON, 10, 7), 96.25),
        exit_=(et(MON, 10, 31, 5), et(MON, 10, 32), 97.35),
        spy_px=(200.5, 201.0),
        reason="target",
    )
    check_facts(t, p0=99.0, sigma=0.01, thr=0.03, low=95.5, target=97.25)

    # The unadjusted close would have triggered at the open and set TGT 97.75: no target.
    raw = one(base_story(), [MON], [day], pre=pre_event(MON, prev_close=100.0, sigma=0.01))
    assert raw.triggered_at == et(MON, 9, 31, 5)
    assert raw.trade is not None and raw.trade.target == 97.75
    assert raw.trade.exit_reason == "time_stop"


def test_a_2_for_1_split_on_s_plus_1_is_not_a_move() -> None:
    """A(MON) = 0.5, A(TUE) = 1: TUE's raw 47.9 is 95.8 in S units (no stop), 48.5 is 97 (TGT)."""
    tue = bars(
        TUE,
        [((9, 30), 47.9), ((10, 1), 48.5)],
        {
            (10, 0): (47.95, 48.6, 47.9, 48.5, 2_000.0, 48.3),
            (10, 2): (48.5, 48.6, 48.5, 48.55, 2_000.0, 48.55),
        },
    )
    adj = {(A, MON): 0.5, (A, TUE): 1.0, (A, WED): 1.0}
    spy_rows = {MON: SPY_ROWS, TUE: {(10, 2): (200.0, 200.9, 199.9, 200.0, 1e5, 200.8)}}
    out = one(
        base_story(),
        [MON, TUE, WED],
        [base_bars(target=False), tue, bars(WED, [((9, 30), 48.5)])],
        spy_rows=spy_rows,
        ctx=Context(adj=adj),
    )
    t = check_trade(
        out.trade,
        entry=ENTRY,
        exit_=(et(TUE, 10, 1, 5), et(TUE, 10, 2), 48.55),
        spy_px=(200.5, 200.8),
        reason="target",
        adj=(0.5, 1.0),
    )
    check_facts(t, variant="MD3")
    assert abs(t.r_gross - (97.1 / 95.45 - 1.0)) < EPS


@pytest.mark.parametrize(("reclaim", "enters"), [((11, 58), True), ((11, 59), False)])
def test_early_close_cutoff_at_noon_and_flatten_at_1255(
    reclaim: tuple[int, int], enters: bool
) -> None:
    """HALF (2016-11-25) closes at 13:00: E1 ends at 12:00, the flatten is at 12:55.

    The reclaim on the 11:58 bar is visible at 11:59:05 and enters; on the 11:59
    bar it is visible at 12:00:05, after the cutoff timer at 12:00:00.
    """
    rows: dict[tuple[int, int], Row] = {
        (9, 30): BASE_ROWS[(9, 30)],
        (9, 45): BASE_ROWS[(9, 45)],
        reclaim: (94.8, 95.5, 94.7, 95.4, 1_000.0, 95.2),
        (12, 0): (95.4, 95.6, 95.3, 95.5, 1_000.0, 95.45),  # the fill bar of the 11:58 reclaim
        (12, 56): (96.0, 96.2, 95.9, 96.1, 1_000.0, 96.05),
    }
    levels = [((9, 30), 96.0), ((9, 46), 94.8), (nxt(reclaim), 95.4), ((12, 1), 96.0)]
    out = one(
        base_story(HALF), [HALF], [bars(HALF, levels, rows)], spy_rows={}, pre=pre_event(HALF)
    )
    if not enters:
        assert out.trade is None
        assert transitions(out)[-1] == (et(HALF, 12, 0), "EXPIRED", "cutoff")
        return
    check_trade(
        out.trade,
        entry=(et(HALF, 11, 59, 5), et(HALF, 12, 0), 95.45),
        exit_=(et(HALF, 12, 55), et(HALF, 12, 56), 96.05),
        spy_px=(200.0, 200.0),
        reason="time_stop",
    )
    assert out.armed_at == et(HALF, 10, 6, 5)  # armed long before; E2 held off the entry
    assert transitions(out)[-2:] == [
        (et(HALF, 12, 55), "EXITING", "time_stop"),
        (et(HALF, 12, 57, 1), "EXITED", "time_stop"),
    ]


@pytest.mark.parametrize(("reclaim", "enters"), [((14, 42), True), ((14, 43), False)])
def test_a_bar_visible_at_the_cutoff_instant_may_still_enter(
    reclaim: tuple[int, int], enters: bool
) -> None:
    """On the 17-minute delayed feed the 14:42 bar is visible at 15:00:00 exactly (E1 is closed)."""
    rows: dict[tuple[int, int], Row] = {
        (9, 30): BASE_ROWS[(9, 30)],
        (9, 45): BASE_ROWS[(9, 45)],
        reclaim: (94.8, 95.5, 94.7, 95.4, 1_000.0, 95.2),
    }
    levels = [((9, 30), 96.0), ((9, 46), 94.8), (nxt(reclaim), 95.4)]
    out = one(
        base_story(), [MON], [bars(MON, levels, rows)], spy_rows={}, cfg=SimConfig(feed=SIP_DELAYED)
    )
    if enters:
        assert out.entry_decided_at == et(MON, 15, 0)
        assert out.trade is not None and out.trade.entry_bar_ts == et(MON, 15, 1)
    else:
        assert out.entry_decided_at is None
        assert transitions(out)[-1] == (et(MON, 15, 0), "EXPIRED", "cutoff")


def test_a_five_minute_gap_fills_at_the_open() -> None:
    """No print 10:06-10:11: the buy active at 10:06:08 fills at the 10:12 bar's open."""
    rows: dict[tuple[int, int], Row] = {(10, 12): (95.6, 95.8, 95.5, 95.7, 1_000.0, 95.65)}
    skip = [(10, m) for m in range(6, 12)]
    spy_rows = {MON: SPY_ROWS | {(10, 12): (200.3, 200.5, 200.2, 200.4, 1e5, 200.35)}}
    out = one(base_story(), [MON], [base_bars(rows=rows, skip=skip)], spy_rows=spy_rows)
    check_trade(
        out.trade,
        entry=(et(MON, 10, 6, 5), et(MON, 10, 12), 95.6),
        exit_=(et(MON, 10, 31, 5), et(MON, 10, 32), 97.1),
        spy_px=(200.3, 201.0),
        reason="target",
        flags=("gap_fill",),
    )


def test_a_late_open_fills_an_exit_pending_overnight() -> None:
    """MON's 15:59 bar closes below L*: decided at 16:00:05, the sell works at TUE 09:30:03
    and fills at the open of TUE's first bar, 09:40 (late)."""
    mon = base_bars(target=False, rows={(15, 59): (96.0, 96.0, 93.8, 93.9, 1_000.0, 94.0)})
    tue = bars(
        TUE, [((9, 40), 93.2)], {(9, 40): (93.0, 93.4, 92.8, 93.2, 1_000.0, 93.1)}, first=(9, 40)
    )
    spy_rows = {MON: SPY_ROWS, TUE: {(9, 40): (199.5, 199.8, 199.4, 199.6, 1e5, 199.6)}}
    out = one(
        base_story(), [MON, TUE, WED], [mon, tue, bars(WED, [((9, 30), 93.2)])], spy_rows=spy_rows
    )
    t = check_trade(
        out.trade,
        entry=ENTRY,
        exit_=(et(MON, 16, 0, 5), et(TUE, 9, 40), 93.0),
        spy_px=(200.5, 199.5),
        reason="stop",
        flags=("gap_fill", "late_open"),
    )
    assert t.exit_active_at == et(TUE, 9, 30, 3)


@pytest.mark.parametrize("sessions", [1, 3])
def test_an_entry_unfilled_at_the_close_expires(sessions: int) -> None:
    """No print after 10:05: ID's buy is cancelled at the flatten, MD3's expires at S's close."""
    days = [MON, TUE, WED][:sessions]
    day_bars = [base_bars(last=(10, 5)), bars(TUE, [((9, 30), 96.0)]), bars(WED, [((9, 30), 96.0)])]
    out = one(base_story(), days, day_bars[:sessions])
    assert out.trade is None and out.entry_bar_ts is None
    assert out.entry_decided_at == et(MON, 10, 6, 5)
    assert (out.terminal_state, out.reason) == ("EXPIRED", "entry_unfilled")
    when = et(MON, 15, 55) if sessions == 1 else et(MON, 16, 0)
    assert transitions(out)[-1] == (when, "EXPIRED", "entry_unfilled")


def test_a_buy_the_simulator_rejects_expires_with_its_reason() -> None:
    """S's screen says not halal (the eligibility, built elsewhere, said ok): the buy decided
    at 10:06:05 is refused at once, and the story ends ``entry_rejected:not_halal``."""
    out = one(base_story(), [MON], [base_bars()], ctx=Context(verdicts={(A, MON): "not_halal"}))
    assert out.trade is None and out.entry_decided_at is None and not out.entered
    assert (out.terminal_state, out.reason) == ("EXPIRED", "entry_rejected:not_halal")
    assert transitions(out)[-2:] == [
        (et(MON, 10, 6, 5), "ENTERING", "entry"),
        (et(MON, 10, 6, 5), "EXPIRED", "entry_rejected:not_halal"),
    ]


@pytest.mark.parametrize("sessions", [1, 3])
def test_a_buy_filled_at_the_deadlines_flatten_goes_straight_to_the_time_stop(
    sessions: int,
) -> None:
    """The 14:58 bar reclaims (decided 14:59:05); nothing prints until the 15:54 bar, which fills
    the buy at its open, 95, at 15:55:00: the exchange runs before the flatten marker of that
    instant, so the simulator's flatten sells it while the playbook is still ENTERING. On the fill
    (15:55:01) the ID playbook goes straight to EXITING (time_stop) and judges no exit: the fill
    bar's close of 93.2 < L* is not a stop. On MD3 the deadline is WED, so it is one."""
    rows: dict[tuple[int, int], Row] = {
        (9, 30): BASE_ROWS[(9, 30)],
        (9, 45): BASE_ROWS[(9, 45)],
        (14, 58): (94.8, 95.5, 94.7, 95.4, 1_000.0, 95.2),
        (15, 54): (95.0, 95.1, 93.0, 93.2, 1_000.0, 94.0),
        (15, 56): (93.2, 93.4, 93.1, 93.3, 1_000.0, 93.25),
    }
    skip = [(14, 59)] + [(15, m) for m in range(54)]
    levels = [((9, 30), 96.0), ((9, 46), 94.8), ((15, 55), 93.2)]
    days = [MON, TUE, WED][:sessions]
    later = [bars(d, [((9, 30), 93.2)]) for d in (TUE, WED)]
    out = one(base_story(), days, [bars(MON, levels, rows, skip=skip), *later][:sessions])
    t = out.trade
    assert t is not None and (t.entry_decided_at, t.entry_bar_ts, t.entry_px) == (
        et(MON, 14, 59, 5),
        et(MON, 15, 54),
        95.0,
    )
    assert (t.exit_bar_ts, t.exit_px) == (et(MON, 15, 56), 93.25)
    if sessions == 3:
        assert (t.exit_reason, t.exit_decided_at) == ("stop", et(MON, 15, 55, 5))
        return
    assert (t.exit_reason, t.exit_decided_at) == ("time_stop", et(MON, 15, 55))
    assert transitions(out)[-4:] == [
        (et(MON, 14, 59, 5), "ENTERING", "entry"),
        (et(MON, 15, 55, 1), "ENTERED", "filled"),
        (et(MON, 15, 55, 1), "EXITING", "time_stop"),
        (et(MON, 15, 57, 1), "EXITED", "time_stop"),
    ]
    assert not [i for _, i in out.intents if isinstance(i, Submit) and i.side == "sell"]


def test_close_fallback() -> None:
    """No print after 15:50: the time stop at 15:55 fills at the official close, 95.7."""
    ctx = Context(daily={(A, MON): daily(95.7), ("SPY", MON): daily(200.4)})
    out = one(base_story(), [MON], [base_bars(target=False, last=(15, 50))], ctx=ctx)
    t = check_trade(
        out.trade,
        entry=ENTRY,
        exit_=(et(MON, 15, 55), None, 95.7),
        spy_px=(200.5, 200.4),
        reason="time_stop",
        flags=("close_fallback",),
    )
    assert t.exit_active_at == et(MON, 15, 55, 3)
    assert transitions(out)[-2:] == [
        (et(MON, 15, 55), "EXITING", "time_stop"),
        (et(MON, 16, 0), "EXITED", "time_stop"),
    ]


def test_compliance_exit_at_s_plus_1() -> None:
    """TUE's screen is not halal: the simulator sells at 09:20, filled on the 09:31 bar."""
    tue = bars(TUE, [((9, 30), 95.8)], {(9, 31): (95.8, 96.0, 95.7, 95.9, 1_000.0, 95.85)})
    spy_rows = {MON: SPY_ROWS, TUE: {(9, 31): (200.0, 200.0, 199.5, 199.7, 1e5, 199.7)}}
    out = one(
        base_story(),
        [MON, TUE, WED],
        [base_bars(target=False), tue, bars(WED, [((9, 30), 95.8)])],
        spy_rows=spy_rows,
        ctx=Context(verdicts={(A, TUE): "not_halal"}),
    )
    t = check_trade(
        out.trade,
        entry=ENTRY,
        exit_=(et(TUE, 9, 20), et(TUE, 9, 31), 95.85),
        spy_px=(200.5, 199.7),
        reason="compliance",
    )
    assert t.exit_active_at == et(TUE, 9, 30, 3)
    assert transitions(out)[-2:] == [
        (et(TUE, 9, 20), "EXITING", "compliance"),
        (et(TUE, 9, 32, 1), "EXITED", "compliance"),
    ]


@pytest.mark.parametrize("structural", [False, True])
def test_blocked_open(structural: bool) -> None:
    """TUE's story starts at 09:30 while MON's MD3 position is held: ``blocked_open``. Its items
    still reach MON's playbook as news; only a structural one (a dilution public at 10:00,
    usable at 10:10) aborts the position."""
    a = base_story()
    extra = [(et(TUE, 10, 0), "dilution")] if structural else []
    b = tstory(A, TUE, (et(TUE, 7, 0), "analyst_downgrade"), *extra)
    tue = bars(TUE, [((9, 30), 96.0)], {(10, 11): (96.0, 96.2, 95.9, 96.1, 1_000.0, 96.05)})
    wed, thu = bars(WED, [((9, 30), 96.0)]), bars(THU, [((9, 30), 96.0)])
    paths = {
        a.story_id: path(a.story_id, A, [MON, TUE, WED], [base_bars(target=False), tue, wed]),
        b.story_id: path(b.story_id, A, [TUE, WED, THU], [tue, wed, thu]),
    }
    context = {
        a.story_id: (pre_event(MON), eligibility()),
        b.story_id: (pre_event(TUE), eligibility()),
    }
    first, second = run(
        [a, b],
        paths,
        spy([MON, TUE, WED, THU], {MON: SPY_ROWS}),
        context,
        params=BounceParams(hold_sessions=3),
    )
    assert (second.story_id, second.terminal_state, second.reason) == (
        b.story_id,
        "DISMISSED",
        "blocked_open",
    )
    assert second.start_at == et(TUE, 9, 30) and second.transitions == ()
    if structural:
        check_trade(
            first.trade,
            entry=ENTRY,
            exit_=(et(TUE, 10, 10), et(TUE, 10, 11), 96.05),
            spy_px=(200.5, 200.0),
            reason="abort",
        )
        assert transitions(first)[-2][:2] == (et(TUE, 10, 10), "EXITING")
    else:
        assert first.trade is not None
        assert (first.trade.exit_reason, first.trade.exit_session) == ("time_stop", WED)


# ── the entry conditions and the exits at their edges ──


@pytest.mark.parametrize(("close", "enters"), [(95.5, True), (95.6, False)])
def test_e3_admits_at_most_a_quarter_of_the_drop(close: float, enters: bool) -> None:
    """At 10:05, 95.5 - 94 = 1.5 = 0.25 x (100 - 94) enters; 95.6 has retraced too far, and
    the price holding there never qualifies."""
    rows: dict[tuple[int, int], Row] = {
        (9, 30): BASE_ROWS[(9, 30)],
        (9, 45): BASE_ROWS[(9, 45)],
        (10, 5): (94.8, 95.7, 94.7, close, 1_000.0, 95.2),
        (10, 7): (close, close + 0.1, close - 0.1, close, 1_000.0, close),
    }
    out = one(
        base_story(), [MON], [bars(MON, [((9, 30), 96.0), ((9, 46), 94.8), ((10, 6), close)], rows)]
    )
    if enters:
        assert out.entry_decided_at == et(MON, 10, 6, 5)
        assert out.trade is not None and out.trade.entry_px == close
    else:
        assert out.entry_decided_at is None and out.armed_at == et(MON, 10, 6, 5)
        assert (out.terminal_state, out.reason) == ("EXPIRED", "cutoff")


@pytest.mark.parametrize("vw", [120.0, float("nan")])
def test_the_avwap_clamps_each_vwap_and_falls_back_to_the_typical_price(vw: float) -> None:
    """09:45's VWAP 120 counts as its high 96 (AVWAP 95.36 < 95.4: still entered at 10:05;
    unclamped it would be 96.03); a null one counts as (96 + 94 + 94.5)/3."""
    rows = {(9, 45): (96.0, 96.0, 94.0, 94.5, 1_000.0, vw)}
    out = one(base_story(), [MON], [base_bars(rows=rows)])
    assert out.entry_decided_at == et(MON, 10, 6, 5)


@pytest.mark.parametrize(("spy_945", "triggered"), [(None, et(MON, 9, 46, 5)), (197.0, None)])
def test_the_drop_is_abnormal_to_spy_at_the_same_bar(
    spy_945: float | None, triggered: datetime | None
) -> None:
    """The stock falls 3.7% on the 09:45 bar while SPY falls 1.5% from 09:45: with SPY's 09:45
    bar the drop is -2.2% (no trigger); without it, SPY's latest bar before (09:44, at 200)
    stands in and D = -3.7% <= -3.6% triggers."""
    stock = bars(
        MON, [((9, 30), 98.0), ((9, 45), 96.3)], {(9, 45): (98.0, 98.0, 96.0, 96.3, 1e3, 97.0)}
    )
    spy_rows = {(9, 30): (200.0, 200.0, 200.0, 200.0, 1e5, 200.0)}
    skip = [] if spy_945 is not None else [(9, 45)]
    day = bars(MON, [((9, 30), 200.0), ((9, 45), 197.0)], spy_rows, skip=skip, volume=1e5)
    st = base_story()
    pd = path(st.story_id, A, [MON], [stock])
    (out,) = run(
        [st], {st.story_id: pd}, SpyData({MON: day}), {st.story_id: (pre_event(MON), eligibility())}
    )
    assert out.triggered_at == triggered


def test_an_abort_outranks_a_stop_on_the_same_bar() -> None:
    """The 10:20 bar closes below L* and is visible at 10:21:05, the instant a dilution item
    becomes usable: X3 > X1."""
    rows: dict[tuple[int, int], Row] = {(10, 20): (96.0, 96.0, 93.8, 93.9, 1_000.0, 94.0)}
    day = bars(MON, BASE_LEVELS + [((10, 20), 93.9)], BASE_ROWS | rows)
    st = base_story(MON, (et(MON, 10, 11, 5), "dilution"))
    out = one(st, [MON], [day])
    assert out.trade is not None and out.trade.exit_reason == "abort"
    assert out.trade.exit_decided_at == et(MON, 10, 21, 5)


def test_a_veto_usable_at_the_entry_instant_blocks_the_entry() -> None:
    """An unclear item usable at 10:06:05 exactly: E5 reads it on the bar (bars come before
    news at one instant), then the news expires the story."""
    st = base_story(MON, (et(MON, 9, 56, 5), "legal_adverse"))
    out = one(st, [MON], [base_bars()])
    assert out.entry_decided_at is None
    assert transitions(out)[-2:] == [
        (et(MON, 10, 6, 5), "ARMED", "quiet"),
        (et(MON, 10, 6, 5), "EXPIRED", "veto"),
    ]


def test_a_structural_item_while_entering_aborts_at_the_fill() -> None:
    """A dilution usable at 10:06:30, after the decision and before the fill (10:08:01)."""
    st = base_story(MON, (et(MON, 9, 56, 30), "dilution"))
    out = one(st, [MON], [base_bars()])
    check_trade(
        out.trade,
        entry=ENTRY,
        exit_=(et(MON, 10, 8, 1), et(MON, 10, 9), 96.0),
        spy_px=(200.5, 200.0),
        reason="abort",
    )


def with_tue(
    *items: tuple[datetime, str],
    rows: dict[tuple[int, int], Row] | None = None,
    params: BounceParams | None = None,
    stage_a: bool = False,
) -> StoryOutcome:
    """MON's base story on ``_bounce.late_reclaim``, beside TUE's story carrying ``items``."""
    a, b = base_story(), tue_story(*items)
    assert b.nsn_at(et(TUE, 15, 0)) is None  # it never starts: it only carries news
    paths = {a.story_id: path(a.story_id, A, [MON], [late_reclaim(rows)])}
    (out,) = run(
        [a, b],
        paths,
        spy([MON]),
        {a.story_id: (pre_event(MON), eligibility())},
        params=params,
        stop_at="entry" if stage_a else "end",
        assume_full_hold=stage_a,
    )
    return out


@pytest.mark.parametrize("stage_a", [False, True])
@pytest.mark.parametrize("watching", [False, True])
def test_another_storys_structural_item_vetoes_a_waiting_story(
    watching: bool, stage_a: bool
) -> None:
    """Spec §B.1's "veto before entry" holds for every story of the symbol. A dilution public at
    14:25 MON belongs to TUE's story (usable at 14:35, after 14:30) and reaches MON's playbook
    as news: armed since 10:06:05 (or watching again after a new low at 14:30), it ends EXPIRED
    (``veto``) at 14:35. It never enters at 14:46:05, so Stage A counts no entry."""
    rows: dict[tuple[int, int], Row] | None = None
    last = (et(MON, 10, 6, 5), "ARMED", "quiet")
    if watching:
        rows = {(14, 30): (94.8, 94.8, 93.9, 94.2, 1_000.0, 94.3)}
        last = (et(MON, 14, 31, 5), "WATCHING", "new_low")
    out = with_tue((et(MON, 14, 25), "dilution"), rows=rows, stage_a=stage_a)
    assert out.trade is None and out.entry_decided_at is None and not out.entered
    assert (out.terminal_state, out.reason) == ("EXPIRED", "veto")
    assert transitions(out)[-2:] == [last, (et(MON, 14, 35), "EXPIRED", "veto")]


def test_another_storys_structural_item_at_the_decision_instant_vetoes_it() -> None:
    """Usable at 14:46:05, the instant the 14:45 bar decides the entry: the bar comes first, then
    the news. The story's own item would have failed E5 at that instant, so this one vetoes as
    well: the ``Finish`` cancels the buy before it works (14:46:08), and nothing fills."""
    out = with_tue((et(MON, 14, 36, 5), "dilution"))
    assert out.trade is None and out.entry_bar_ts is None and not out.entered
    assert out.entry_decided_at == et(MON, 14, 46, 5)
    assert (out.terminal_state, out.reason) == ("EXPIRED", "veto")
    assert transitions(out)[-2:] == [
        (et(MON, 14, 46, 5), "ENTERING", "entry"),
        (et(MON, 14, 46, 5), "EXPIRED", "veto"),
    ]


@pytest.mark.parametrize("require_family", [True, False])
def test_another_storys_structural_item_after_the_decision_aborts_at_the_fill(
    require_family: bool,
) -> None:
    """Usable at 14:46:30, after the decision (14:46:05) and before the fill (14:48:01): X3 at
    the fill, under H1 and in the atlas alike."""
    params = BounceParams(require_family=require_family)
    out = with_tue((et(MON, 14, 36, 30), "dilution"), params=params)
    t = out.trade
    assert t is not None
    assert (t.entry_decided_at, t.entry_bar_ts) == (et(MON, 14, 46, 5), et(MON, 14, 47))
    assert (t.exit_reason, t.exit_decided_at, t.exit_bar_ts) == (
        "abort",
        et(MON, 14, 48, 1),
        et(MON, 14, 49),
    )


# ── the rest of the state machine ──


@pytest.mark.parametrize("reason", ["rank", "not_halal", "share_class"])
def test_an_ineligible_story_is_dismissed_with_its_reason(reason: str) -> None:
    out = one(base_story(), [MON], [base_bars()], elig=eligibility(reason=reason))
    assert (out.terminal_state, out.reason) == ("DISMISSED", reason)
    assert transitions(out) == [(et(MON, 9, 30), "DISMISSED", reason)]
    assert out.trade is None and out.triggered_at is None


def test_an_eligible_story_without_a_pre_event_is_dismissed() -> None:
    st = base_story()
    pd = path(st.story_id, A, [MON], [base_bars()])
    (out,) = run([st], {st.story_id: pd}, spy([MON]), {st.story_id: (None, eligibility())})
    assert (out.terminal_state, out.reason) == ("DISMISSED", "no_pre_event")


def test_without_the_family_check_the_rule_runs_on_other_negative_types() -> None:
    """The atlas's mode: a price-target cut (never NSN) is watched and traded, in its own cell."""
    st = AtlasStory(A, MON, titems((et(MON, 7, 0), "analyst_pt_cut")))
    assert st.card_at(et(MON, 10, 0)).family is None
    pd = path(st.story_id, A, [MON], [base_bars()])
    context = {st.story_id: (pre_event(MON), eligibility())}
    (strict,) = run([st], {st.story_id: pd}, spy([MON], {MON: SPY_ROWS}), context)
    assert (strict.terminal_state, strict.reason) == ("EXPIRED", "veto")
    (atlas,) = run(
        [st],
        {st.story_id: pd},
        spy([MON], {MON: SPY_ROWS}),
        context,
        params=BounceParams(require_family=False),
    )
    t = check_trade(
        atlas.trade,
        entry=ENTRY,
        exit_=(et(MON, 10, 31, 5), et(MON, 10, 32), 97.1),
        spy_px=(200.5, 201.0),
        reason="target",
    )
    assert (t.family_type, t.cell) == ("analyst_pt_cut", "analyst_pt_cut/ID")


@pytest.mark.parametrize(
    ("public", "itype", "reason", "decided"),
    [
        (et(MON, 10, 10), "restatement", "abort", et(MON, 10, 20)),  # usable after the fill
        (et(MON, 9, 56, 30), "restatement", "abort", et(MON, 10, 8, 1)),  # entering: at the fill
        (et(MON, 9, 56, 5), "restatement", "target", et(MON, 10, 31, 5)),  # at the decision
        (et(MON, 9, 50, 30), "restatement", "target", et(MON, 10, 31, 5)),  # before it
        (et(MON, 10, 10), "dilution", "target", et(MON, 10, 31, 5)),  # its own type again
        (et(MON, 10, 10), "legal_adverse", "target", et(MON, 10, 31, 5)),  # unclear
    ],
)
def test_the_atlas_aborts_on_a_structural_type_its_story_gains_after_the_decision(
    public: datetime, itype: str, reason: str, decided: datetime
) -> None:
    """The atlas runs a dilution story (structural from its first item) on the base path. What
    the card held at the decision (10:06:05) never aborts; an item adding a structural type the
    card lacked (a restatement) aborts once it is usable after the decision; nothing else does."""
    st = AtlasStory(A, MON, titems((et(MON, 7, 0), "dilution"), (public, itype)))
    out = one(st, [MON], [base_bars()], params=BounceParams(require_family=False))
    t = out.trade
    assert t is not None and t.entry_decided_at == et(MON, 10, 6, 5)
    assert (t.family_type, t.cell) == ("dilution", "dilution/ID")
    assert (t.exit_reason, t.exit_decided_at) == (reason, decided)


@pytest.mark.parametrize(
    ("later", "reason"),
    [
        (None, "time_stop"),
        ("restatement", "abort"),  # a structural type the card lacked
        ("dilution", "time_stop"),  # its type again
        ("analyst_pt_cut", "time_stop"),  # not structural
        ("law_firm", "time_stop"),  # types nothing
        ("noise", "time_stop"),
    ],
)
def test_the_atlas_aborts_on_a_structural_type_another_story_gains_after_the_decision(
    later: str | None, reason: str
) -> None:
    """The atlas has no vetoes: TUE's dilution usable at 14:35 neither stops MON's entry
    (14:46:05) nor aborts it, being known at the decision. A later item of that story, usable at
    14:50, aborts only when it adds a structural type the card lacked (a restatement), the check
    the story's own items take; nothing else does, so the trade runs to its 15:55 time stop."""
    items = [(et(MON, 14, 25), "dilution")]
    if later is not None:
        items.append((et(MON, 14, 40), later))
    out = with_tue(*items, params=BounceParams(require_family=False))
    t = out.trade
    assert t is not None and t.entry_decided_at == et(MON, 14, 46, 5)
    assert t.exit_reason == reason
    if reason == "abort":
        assert (t.exit_decided_at, t.exit_bar_ts) == (et(MON, 14, 50), et(MON, 14, 51))
    else:
        assert t.exit_decided_at == et(MON, 15, 55)


@pytest.mark.parametrize("full_hold", [False, True])
def test_stage_a_stops_at_the_fill_and_may_hold_the_symbol(full_hold: bool) -> None:
    """Stage A (spec §G.9: ``stop_at="entry"``): MON's MD3 story stops at its fill, with no exit
    judged and no trade record. With ``assume_full_hold`` the symbol stays taken through WED, so
    TUE's story is ``blocked_open``; without it, TUE's story runs."""
    a, b = base_story(), tstory(A, TUE, (et(TUE, 7, 0), "analyst_downgrade"))
    flat = {d: bars(d, [((9, 30), 96.0)]) for d in (TUE, WED, THU)}
    paths = {
        a.story_id: path(a.story_id, A, [MON, TUE, WED], [base_bars(), flat[TUE], flat[WED]]),
        b.story_id: path(b.story_id, A, [TUE, WED, THU], [flat[TUE], flat[WED], flat[THU]]),
    }
    context = {s.story_id: (pre_event(s.session), eligibility()) for s in (a, b)}
    first, second = run(
        [a, b],
        paths,
        spy([MON, TUE, WED, THU], {MON: SPY_ROWS}),
        context,
        params=BounceParams(hold_sessions=3),
        stop_at="entry",
        assume_full_hold=full_hold,
    )
    assert first.trade is None and first.entered
    assert (first.entry_decided_at, first.entry_bar_ts) == (et(MON, 10, 6, 5), et(MON, 10, 7))
    assert (first.terminal_state, first.reason) == ("ENTERED", "filled")
    assert transitions(first)[-1] == (et(MON, 10, 8, 1), "ENTERED", "filled")
    assert second.start_at == et(TUE, 9, 30)
    blocked = (second.terminal_state, second.reason) == ("DISMISSED", "blocked_open")
    assert blocked is full_hold


def test_states_and_liveness() -> None:
    assert [s.value for s in BounceState] == [
        "DETECTED",
        "WATCHING",
        "ARMED",
        "ENTERING",
        "ENTERED",
        "EXITING",
        "EXITED",
        "EXPIRED",
        "DISMISSED",
    ]
    st = base_story()
    pb = BounceFactory({st.story_id: (pre_event(MON), eligibility())})(st)
    for s in BounceState:
        pb._state = s
        assert pb.live() is (s in {"WATCHING", "ARMED", "ENTERING", "ENTERED"})
        assert pb.state() == s.value
