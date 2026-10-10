"""H1's pre-registration, its preconditions and its statistics (events/h1.py).

Pure parts first (the PREREG, the gate and code checks, carriers, Stage A
counts, the tests of spec §G.10, the verdict), then each data precondition
against a small synthetic database. No price outcome of any real story is
computed anywhere here.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.playbooks import bounce
from halabot.playbooks.bounce import BounceParams
from halabot.playbooks.loader import WINDOW_LAST, WINDOW_START, Window
from halabot.playbooks.records import Leg, StoryOutcome, TradeRecord
from halabot.playbooks.sim import start_time
from halabot.playbooks.types import Session, SimConfig
from halal_trader.data import minutes
from halal_trader.data.minutes import BarArrays
from halal_trader.db.repos.quant_trials import QuantTrialRepoImpl, config_hash
from halal_trader.events import h1
from halal_trader.events import stats as event_stats
from halal_trader.events.aliases import AliasMatcher
from halal_trader.events.earnings_parse import EXTRACTOR
from halal_trader.events.h1 import (
    CELLS,
    COUNT_RULE,
    Carrier,
    Cell,
    CellCounts,
    Check,
    CodeState,
    H1Locked,
    Preconditions,
    RegistrationRefused,
    build_prereg,
    cell_eligible,
    check_code,
    clean,
    count_outcomes,
    decide,
    gate_family,
    judge_gates,
    judge_window,
    last_session,
    quarter_starts,
    relag,
    subset_sensitivities,
    trial_config,
    window_span,
    window_stats,
)
from halal_trader.events.stories import NEWS_LAG, RawItem, Story, build
from halal_trader.events.taxonomy import FAMILY
from halal_trader.market_hours import MARKET_TZ, is_trading_day, next_trading_day
from tests._renames import mark_renamed_news_done
from tests._stories import news_row, store
from tests.halabot.playbooks._seed import mark_done, seed_bars
from tests.halabot.playbooks._support import session_bars

ID, MD3 = CELLS
PINS = {
    "builder_version": "stories-v1",
    "stories_sha": "aaaaaaaaaaaa",
    "alias_sha": "bbbbbbbbbbbb",
    "renames_sha": "cccccccccccc",
    "taxonomy_sha": "dddddddddddd",
    "parser_sha": "eeeeeeeeeeee",
    "headline_patterns_sha": "ffffffffffff",
}


def ny(day: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm), MARKET_TZ)


# ── cells, windows, the pre-registration ─────────────────────


def test_the_cells_are_the_spec_cells_and_name_their_trials() -> None:
    assert CELLS == (Cell("NSN_CORE", "ID", 1), Cell("NSN_CORE", "MD3", 3))
    assert {c.family for c in CELLS} == {FAMILY, bounce.FAMILY}
    assert (ID.key, MD3.key) == ("NSN_CORE/ID", "NSN_CORE/MD3")
    assert (ID.strategy, MD3.strategy) == ("news.h1.nsn_core.id", "news.h1.nsn_core.md3")
    assert MD3.params() == BounceParams(hold_sessions=3)
    assert MD3.params().variant == "MD3" and ID.params().variant == "ID"


def test_each_cell_ends_where_its_path_still_fits_the_window() -> None:
    assert window_span("train") == (WINDOW_START[Window.TRAIN], WINDOW_LAST[Window.TRAIN])
    assert window_span("validation") == (date(2022, 1, 3), date(2024, 12, 31))
    assert last_session("train", ID) == date(2021, 12, 31)
    assert last_session("train", MD3) == date(2021, 12, 29)
    assert last_session("validation", ID) == date(2024, 12, 31)
    assert last_session("validation", MD3) == date(2024, 12, 27)  # 27, 30, 31


def test_the_prereg_holds_every_pin_and_the_code_constants() -> None:
    pre = build_prereg(PINS)
    assert pre["pins"] == PINS
    assert pre["taxonomy"]["extractor"] == EXTRACTOR
    assert pre["builder"]["news_lag_s"] == int(NEWS_LAG.total_seconds())
    playbook = dict(BounceParams().as_config())
    playbook.pop("hold_sessions")
    assert {k: pre["playbook"][k] for k in playbook} == playbook
    assert "hold_sessions" not in pre["playbook"]
    assert pre["playbook"]["name"] == bounce.NAME
    sim = SimConfig().as_config()
    assert {k: pre["fills"][k] for k in sim} == sim
    assert pre["cells"] == [["NSN_CORE", "ID", 1], ["NSN_CORE", "MD3", 3]]
    assert pre["count_rule"] == {
        "validation_per_year": 200.0,
        "validation_dates": 100,
        "train_n": 500,
        "train_dates": 100,
    }
    assert pre["windows"]["train"] == ["2016-10-03", "2021-12-31"]
    assert pre["windows"]["train_last_session"] == {"ID": "2021-12-31", "MD3": "2021-12-29"}
    assert pre["book"] == {
        "slots": 8,
        "daily_loss_limit": 0.02,
        "benchmark": "SPY (exposure-matched)",
    }
    assert pre["sensitivities"] == list(h1.SENSITIVITIES)
    assert json.loads(json.dumps(pre)) == pre  # plain JSON, as the ledger stores it
    assert build_prereg(dict(reversed(list(PINS.items())))) == pre


def test_a_changed_pin_is_a_new_configuration_and_both_windows_share_a_trial() -> None:
    h = config_hash(build_prereg(PINS))
    assert config_hash(build_prereg({**PINS, "alias_sha": "000000000000"})) != h
    assert config_hash(build_prereg({**PINS, "new_pin": "1"})) != h
    cfg = trial_config(ID, h)
    assert cfg == {"prereg": h, "cell": ["NSN_CORE", "ID", 1], "feed": "sip-rt"}
    assert config_hash(trial_config(MD3, h)) != config_hash(cfg)
    assert config_hash(trial_config(ID, h, feed="sip-delayed")) != config_hash(cfg)


# ── gates and the code ───────────────────────────────────────


def test_gate_ids_count_for_their_family() -> None:
    assert [gate_family(g) for g in ("g1", "lookahead", "G2", "reactor.r1", "sue:s2")] == [
        "G1",
        "G1",
        "G2",
        "G2",
        "G3",
    ]
    assert gate_family("calib") is None and gate_family("reactors") is None


def test_every_gate_family_needs_a_row_and_every_latest_row_must_pass() -> None:
    ok = judge_gates([("lookahead", 1, "pass"), ("reactor", 2, "pass"), ("sue", 3, "pass")])
    assert ok.ok and set(ok.data["gates"]) == {"lookahead", "reactor", "sue"}
    missing = judge_gates([("lookahead", 1, "pass"), ("reactor", 2, "pass")])
    assert not missing.ok and "G3" in missing.detail
    failed = judge_gates(
        [("g1", 1, "pass"), ("g2", 2, "pass"), ("g3", 3, "pass"), ("calib", 4, "fail")]
    )
    assert not failed.ok and "calib" in failed.detail  # an unknown gate id still counts
    fixed = judge_gates(
        [("g1", 1, "pass"), ("g2", 5, "pass"), ("g2", 2, "fail"), ("g3", 3, "pass")]
    )
    assert fixed.ok  # a rerun after a fix replaces the failed row
    broke = judge_gates(
        [("g1", 1, "pass"), ("g2", 2, "pass"), ("g2", 5, "fail"), ("g3", 3, "pass")]
    )
    assert not broke.ok


def test_the_code_must_be_tagged_and_clean() -> None:
    assert not check_code(CodeState(None)).ok
    assert not check_code(CodeState("a" * 40, (), False)).ok
    assert not check_code(CodeState("a" * 40, (h1.TAG,), True)).ok
    assert not check_code(CodeState("a" * 40, (h1.TAG,), None)).ok
    good = check_code(CodeState("a" * 40, ("other", h1.TAG), False))
    assert good.ok and good.data["commit"] == "a" * 40


def test_code_state_reads_head_its_tags_and_tracked_changes(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        unsigned = ["-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"]
        subprocess.run(["git", *unsigned, *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "a.txt").write_text("1")
    git("add", "a.txt")
    git("commit", "-q", "-m", "one")
    git("tag", h1.TAG)
    (tmp_path / "untracked.txt").write_text("x")  # untracked files do not count
    state = h1.code_state(tmp_path)
    assert state.commit and len(state.commit) == 40
    assert state.tags == (h1.TAG,) and state.dirty is False and state.frozen
    (tmp_path / "a.txt").write_text("2")
    assert h1.code_state(tmp_path).dirty is True
    assert h1.code_state(tmp_path / "missing").commit is None


def test_pinned_files_are_hashed() -> None:
    shas = h1.file_shas()
    assert set(shas) == set(h1.PINNED_MODULES)
    assert shas["halal_trader.events.h1"] is not None and len(shas["halal_trader.events.h1"]) == 12


# ── stories: carriers and the news-lag sensitivity ────────────

S = date(2017, 3, 7)
ALIASES = {"AAA": AliasMatcher("AAA", ("Acme",), ("AAA",))}


def _raw(n: int, at: datetime, headline: str) -> RawItem:
    return RawItem(n, f"s{n}", "news", "AAA", at.astimezone(UTC), at.astimezone(UTC), headline, 1)


def _story() -> Story:
    (story,) = build(
        [
            _raw(1, ny(S, 8), "Morgan Stanley Downgrades Acme to Equal-Weight"),
            _raw(2, ny(S, 11), "Acme Files For Chapter 11 Bankruptcy Protection"),
        ],
        ALIASES,
    )
    return story


def test_a_carrier_brings_its_news_but_never_starts() -> None:
    story = _story()
    assert start_time(story) == Session.of(S).open
    carrier = Carrier(story)
    assert (carrier.story_id, carrier.symbol, carrier.session) == (
        story.story_id,
        "AAA",
        S,
    )
    assert carrier.nsn_at(Session.of(S).close) is None and carrier.at_news() is None
    assert start_time(carrier) is None
    assert list(carrier.news_times()) == story.news_times()
    late = ny(S, 12)
    assert carrier.card_at(late) == story.card_at(late)
    assert carrier.card_at(late).structural


def test_a_news_lag_moves_only_when_items_become_usable() -> None:
    story = _story()
    for lag in (timedelta(seconds=60), timedelta(seconds=1200)):
        moved = relag(story, lag)
        assert [i.available_at - i.at for i in moved.items] == [lag, lag]
        assert [i.at for i in moved.items] == [i.at for i in story.items]
        assert (moved.story_id, moved.session, moved.parent) == (
            story.story_id,
            story.session,
            story.parent,
        )
        assert moved.nsn_at(Session.of(S).entry_cutoff) == ny(S, 8) + lag
        assert moved.at_news() == story.at_news() == ny(S, 8)
    assert relag(story, NEWS_LAG) is story
    assert story.items[0].available_at == ny(S, 8) + NEWS_LAG  # the original is untouched


# ── Stage A ──────────────────────────────────────────────────


def _outcome(
    sid: str,
    state: str,
    reason: str,
    *,
    skip: str | None = None,
    triggered: bool = False,
    armed: bool = False,
    entry: datetime | None = None,
) -> StoryOutcome:
    t = ny(S, 10)
    return StoryOutcome(
        story_id=sid,
        symbol=sid.split(":")[0],
        session=S,
        terminal_state=state,
        reason=reason,
        skip=skip,
        triggered_at=t if triggered else None,
        armed_at=t if armed else None,
        trade=None,
        entry_decided_at=entry,
        entry_bar_ts=entry,
    )


def test_stage_a_counts_entries_dates_and_the_data_skips() -> None:
    other = date(2017, 3, 8)
    outcomes = [
        _outcome("A:1", "ENTERED", "filled", triggered=True, armed=True, entry=ny(S, 10, 30)),
        _outcome("B:1", "ENTERED", "filled", triggered=True, armed=True, entry=ny(S, 11)),
        _outcome("C:1", "ENTERED", "filled", triggered=True, armed=True, entry=ny(other, 10)),
        _outcome("D:1", "EXPIRED", "cutoff", triggered=True, armed=True),
        _outcome("E:1", "EXPIRED", "veto"),
        _outcome("F:1", "DISMISSED", "blocked_open"),
        _outcome("G:1", "DISMISSED", "no_pre_event"),
        _outcome("H:1", "SKIPPED", "units_missing", skip="units_missing"),
        _outcome("I:1", "SKIPPED", "halted_all_day", skip="halted_all_day"),
    ]
    c = count_outcomes(
        outcomes, stories=40, nsn=12, eligible=9, reasons={"ok": 9, "rank": 3}, years=3.0
    )
    assert (c.entries, c.dates, c.triggered, c.armed) == (3, 2, 4, 4)
    assert (c.blocked_open, c.dismissed, c.expired) == (
        1,
        {"no_pre_event": 1},
        {"cutoff": 1, "veto": 1},
    )
    assert c.skips == {"units_missing": 1, "halted_all_day": 1}
    assert c.data_skips == 1  # halted all day is no data skip
    assert c.skip_share == pytest.approx(1 / 9) and c.per_year == pytest.approx(1.0)
    assert c.as_dict()["entry_dates"] == 2 and c.terminal["ENTERED/filled"] == 3


def _counts(entries: int, dates: int, years: float) -> CellCounts:
    return CellCounts(
        stories=0,
        nsn=0,
        eligible=0,
        reasons={},
        outcomes=0,
        blocked_open=0,
        triggered=0,
        armed=0,
        entries=entries,
        dates=dates,
        data_skips=0,
        skips={},
        dismissed={},
        expired={},
        terminal={},
        years=years,
    )


def test_a_cell_is_eligible_by_the_count_rule() -> None:
    assert cell_eligible(_counts(500, 100, 5.25), _counts(600, 100, 3.0))
    assert not cell_eligible(_counts(499, 100, 5.25), _counts(600, 100, 3.0))
    assert not cell_eligible(_counts(500, 99, 5.25), _counts(600, 100, 3.0))
    assert not cell_eligible(_counts(500, 100, 5.25), _counts(599, 100, 3.0))  # 199.7 a year
    assert not cell_eligible(_counts(500, 100, 5.25), _counts(600, 99, 3.0))
    lenient = h1.CountRule(validation_per_year=1, validation_dates=1, train_n=1, train_dates=1)
    assert cell_eligible(_counts(1, 1, 5.25), _counts(3, 1, 3.0), lenient)
    assert COUNT_RULE == h1.CountRule()


# ── a window's tests ─────────────────────────────────────────


def _trade(
    sid: str,
    day: date,
    r: float,
    *,
    cost: float = 7.0,
    beta_adj: float | None = None,
    flags: tuple[str, ...] = (),
    tech: bool = False,
    rank: int = 100,
    participation: float = 0.01,
    family_type: str = "analyst_downgrade",
) -> TradeRecord:
    at = ny(day, 10, 23).astimezone(UTC)
    nan = math.nan
    return TradeRecord(
        run_id="r",
        story_id=sid,
        symbol=sid.split(":")[0],
        family_type=family_type,
        cell="NSN_CORE/ID",
        variant="ID",
        feed="sip-rt",
        session=day,
        exit_session=day,
        sessions_held=1,
        start_case="out",
        at_news=at,
        nsn_at=at,
        anchor_ts=at,
        entry_decided_at=at,
        entry_active_at=at,
        entry_bar_ts=at,
        exit_decided_at=at,
        exit_active_at=at,
        exit_bar_ts=at,
        p0=nan,
        spy0=nan,
        sigma=nan,
        thr=nan,
        low_star=nan,
        target=nan,
        entry_px=nan,
        exit_px=nan,
        adj_entry=1.0,
        adj_exit=1.0,
        spy_entry_px=nan,
        spy_exit_px=nan,
        spy_adj_entry=1.0,
        spy_adj_exit=1.0,
        cost_bps=cost,
        rank=rank,
        tech=tech,
        beta=1.0,
        r_gross=nan,
        r_spy=nan,
        r_net_abn=r,
        r_beta_adj=r if beta_adj is None else beta_adj,
        exit_reason="target",
        mae=nan,
        mfe=nan,
        hold_minutes=30,
        participation=participation,
        flags=flags,
    )


DAYS = [
    date(2017, 3, 6) + timedelta(days=i)
    for i in range(30)
    if is_trading_day(date(2017, 3, 6) + timedelta(days=i))
]


def _book(values: list[float]) -> list[TradeRecord]:
    return [_trade(f"S{i}:1", DAYS[i], v) for i, v in enumerate(values)]


def test_t1_needs_a_positive_mean_and_a_clustered_t_of_two() -> None:
    trades = _book([0.010, 0.012, 0.008, 0.011, 0.009])
    ws = window_stats(ID, "validation", trades, {}, DAYS)
    fit = event_stats.clustered_mean([t.r_net_abn for t in trades], [t.session for t in trades])
    assert fit is not None and ws.cr1 == fit and ws.p == fit.p_one_sided
    assert ws.t1 and ws.t3 and ws.t4 and ws.t5 and ws.t6 and ws.mean_ex_covid is None
    assert ws.mean_cost15 == pytest.approx(ws.mean - 7.0 / 1e4)  # one more side at 1.5x
    noisy = window_stats(ID, "validation", _book([0.03, -0.028, 0.002]), {}, DAYS)
    assert not noisy.t1 and noisy.cr1 is not None and noisy.cr1.t < 2
    one_day = window_stats(
        ID, "validation", [_trade("A:1", S, 0.01), _trade("B:1", S, 0.02)], {}, DAYS
    )
    assert one_day.cr1 is None and not one_day.t1 and one_day.p == 1.0 and one_day.dates == 1
    empty = window_stats(ID, "train", [], {}, DAYS)
    assert empty.n == 0 and not empty.t1 and not empty.t3 and not empty.t5 and empty.t6


def test_t3_t4_t5_and_t6() -> None:
    covid = [date(2020, 3, 2), date(2020, 3, 3), date(2020, 6, 30)]
    calm = [date(2019, 3, 4), date(2019, 3, 5)]
    trades = [_trade(f"C{i}:1", d, 0.05) for i, d in enumerate(covid)]
    trades += [_trade(f"N{i}:1", d, -0.001) for i, d in enumerate(calm)]
    ws = window_stats(ID, "train", trades, {}, [*calm, *covid])
    assert ws.mean > 0 and ws.mean_ex_covid == pytest.approx(-0.001) and not ws.t5
    beta = [_trade(f"B{i}:1", DAYS[i], 0.01, beta_adj=-0.002) for i in range(4)]
    assert not window_stats(ID, "validation", beta, {}, DAYS).t4
    costly = [_trade(f"K{i}:1", DAYS[i], 0.0004 + i * 1e-5, cost=15.0) for i in range(4)]
    k = window_stats(ID, "validation", costly, {}, DAYS)
    assert k.mean > 0 and not k.t3  # 4 bps net, minus 15 bps more
    # T6: at most 0.5% of the entries unresolved: 2 of 400, not 3 of 400.
    many = [_trade(f"U{i}:1", DAYS[i % len(DAYS)], 0.01 + 1e-4 * i) for i in range(400)]
    two = [*many[:398], *(_unresolved(t) for t in many[398:])]
    three = [*many[:397], *(_unresolved(t) for t in many[397:])]
    assert window_stats(ID, "validation", two, {}, DAYS).t6
    worse = window_stats(ID, "validation", three, {}, DAYS)
    assert worse.unresolved == 3 and not worse.t6


def _unresolved(t: TradeRecord) -> TradeRecord:
    from dataclasses import replace

    return replace(t, flags=("unresolved",))


def _legs(t: TradeRecord, abn: float) -> tuple[Leg, ...]:
    return (Leg(t.session, 100.0, 100.0 * (1.0 + abn), 200.0, 200.0),)


def test_md3_also_needs_the_calendar_time_newey_west_t() -> None:
    trades = [
        _trade(f"M{i}:1", DAYS[i], r)
        for i, r in enumerate([0.010, 0.012, 0.008, 0.011, 0.009, 0.010])
    ]
    legs = {t.story_id: _legs(t, t.r_net_abn) for t in trades}
    ws = window_stats(MD3, "validation", trades, legs, DAYS)
    series = event_stats.calendar_series(legs, DAYS)
    nw = event_stats.newey_west_mean([x for _, x in series])
    assert ws.nw == nw and nw is not None and ws.t1
    assert ws.p == max(ws.cr1.p_one_sided, nw.p_one_sided)  # type: ignore[union-attr]
    flat_legs = {t.story_id: _legs(t, -0.001 if i % 2 else 0.0005) for i, t in enumerate(trades)}
    weak = window_stats(MD3, "validation", trades, flat_legs, DAYS)
    assert (
        weak.cr1 == ws.cr1 and weak.nw is not None and not weak.t1
    )  # the CR1 t alone is not enough


def test_holm_and_t6_decide_each_cells_status() -> None:
    strong = window_stats(ID, "validation", _book([0.010, 0.012, 0.008, 0.011, 0.009]), {}, DAYS)
    weak = window_stats(MD3, "validation", _book([0.03, -0.028, 0.002]), {}, DAYS)
    judged = judge_window({ID: strong, MD3: weak})
    assert judged[ID].t2 and judged[ID].status == "pass"
    assert judged[MD3].status == "fail"
    expected = event_stats.holm({ID.key: strong.p, MD3.key: weak.p}, alpha=0.05)
    assert {c.key: ws.t2 for c, ws in judged.items()} == expected
    from dataclasses import replace

    unresolved = judge_window({ID: replace(strong, t6=False)})
    assert unresolved[ID].status == "inconclusive"
    assert judge_window({}) == {}


def test_the_verdict_needs_one_cell_to_pass_both_windows() -> None:
    assert decide({"A": "pass", "B": "fail"}, {"A": "pass"}) == "pass"
    assert decide({"A": "pass", "B": "pass"}, {"A": "fail", "B": "pass"}) == "pass"
    assert decide({"A": "pass"}, {"A": "fail"}) == "fail"
    assert decide({"A": "fail", "B": "fail"}, {}) == "fail"
    assert decide({"A": "inconclusive", "B": "fail"}, {}) == "inconclusive"
    assert decide({"A": "pass"}, {"A": "inconclusive"}) == "inconclusive"


def test_the_sensitivities_read_from_the_base_trades() -> None:
    trades = [
        _trade("T:1", DAYS[0], 0.02, tech=True, rank=50),
        _trade("T:2", DAYS[1], 0.01, tech=True, rank=400, participation=0.2),
        _trade("O:1", DAYS[2], -0.01, rank=700, flags=("unresolved",)),
    ]
    out = subset_sensitivities(ID, "train", trades, {}, DAYS)
    assert out["tech"]["tech"]["trades"] == 2 and out["tech"]["other"]["trades"] == 1
    assert out["rank_split"]["lt300"]["trades"] == 1 and out["rank_split"]["300_999"]["trades"] == 2
    assert out["unresolved_minus100"]["all"]["mean_trade"] == pytest.approx((0.02 + 0.01 - 1.0) / 3)
    assert out["participation_le_10pct"]["dropped"] == 1
    assert out["participation_le_10pct"]["kept"]["trades"] == 2


def test_clean_makes_ledger_json() -> None:
    value = {
        "a": math.nan,
        "b": [np.float64(1.5), np.int64(2), (date(2020, 1, 2),)],
        "c": {1: frozenset({"y", "x"})},
        "d": math.inf,
        "e": True,
    }
    assert clean(value) == {
        "a": None,
        "b": [1.5, 2, ["2020-01-02"]],
        "c": {"1": ["x", "y"]},
        "d": None,
        "e": True,
    }


def test_quarters_and_sessions() -> None:
    assert quarter_starts(date(2016, 10, 1), date(2017, 6, 30)) == [
        date(2016, 10, 1),
        date(2017, 1, 1),
        date(2017, 4, 1),
    ]
    days = h1.sessions_between(date(2016, 12, 23), date(2016, 12, 28))
    assert days == [date(2016, 12, 23), date(2016, 12, 27), date(2016, 12, 28)]


# ── preconditions against the database ──────────────────────


async def _spy_raw(
    engine: AsyncEngine, first: date, last: date, *, skip: set[date] | None = None
) -> None:
    rows = []
    d = first
    while d <= last:
        if is_trading_day(d) and d not in (skip or set()):
            rows.append({"d": d})
        d += timedelta(days=1)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, volume, "
                "fetched_at) VALUES ('SPY', :d, 'raw', 200, 200, 200, 200, 1e6, now())"
            ),
            rows,
        )


async def test_d1_holds_when_spy_traded_every_session(engine: AsyncEngine) -> None:
    await _spy_raw(engine, *h1.D1_SPAN, skip={date(2019, 5, 7)})
    bad = await h1.check_calendar_d1(engine)
    assert not bad.ok and "2019-05-07" in bad.detail
    await _spy_raw(engine, date(2019, 5, 7), date(2019, 5, 7))
    assert (await h1.check_calendar_d1(engine)).ok


async def _screen(
    engine: AsyncEngine, as_of: date, rows: list[tuple[str, int | None, str]]
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, cik, sic_description, verdict, "
                "reasons, metrics, method, screened_at) VALUES (:a, :s, :c, :sic, :v, '[]', '{}', "
                "'v12', now())"
            ),
            [
                {"a": as_of, "s": s, "c": cik, "sic": sic, "v": "halal" if cik else "doubtful"}
                for s, cik, sic in rows
            ],
        )


async def _ticker(engine: AsyncEngine, symbol: str, status: str) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO ticker_ciks (symbol, status, cik, matched_at) "
                "VALUES (:s, :st, NULL, now())"
            ),
            {"s": symbol, "st": status},
        )


async def test_d2_measures_each_screens_unmapped_share(engine: AsyncEngine) -> None:
    from halal_trader.compliance.runner import UNMAPPED

    software = "SERVICES-PREPACKAGED SOFTWARE"
    names = [(f"C{i:03d}", i + 1, software) for i in range(97)]
    unmapped = [("U1", None, UNMAPPED), ("U2", None, UNMAPPED)]
    await _screen(engine, date(2016, 9, 30), [*names, *unmapped, ("FUND", None, UNMAPPED)])
    for symbol, status in (("U1", "no_match"), ("U2", "ambiguous"), ("FUND", "fund")):
        await _ticker(engine, symbol, status)
    ok = await h1.check_screens_d2(engine)
    assert ok.ok, ok.detail
    assert ok.data["residual"] == {"2016-09-30": pytest.approx(2 / 99)}  # the fund is left out
    await _screen(
        engine, date(2016, 12, 30), [*names[:40], ("U3", None, UNMAPPED), ("U1", None, UNMAPPED)]
    )
    bad = await h1.check_screens_d2(engine)
    assert not bad.ok and "unmapped ticker(s) never matched" in bad.detail  # U3: no outcome
    assert "above 3%" in bad.detail and bad.data["unmatched"] == 1
    await _ticker(engine, "U3", "mapped")
    pending = await h1.check_screens_d2(engine)
    assert "not re-screened" in pending.detail and pending.data["pending_rescreen"] == 1


async def _monthly(engine: AsyncEngine, symbols: list[str], first: date, last: date) -> None:
    rows = []
    d = first
    while d <= last:
        rows += [
            {"s": s, "m": d, "c": 50.0, "v": 1e6 * (len(symbols) - i)}
            for i, s in enumerate(symbols)
        ]
        d = date(d.year + d.month // 12, d.month % 12 + 1, 1)
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO monthly_bars (symbol, month, close, volume) VALUES (:s, :m, :c, :v)"),
            rows,
        )


async def _story_row(
    engine: AsyncEngine,
    symbol: str,
    session: date,
    event_ids: list[int],
    itype: str = "analyst_downgrade",
    type_close: str = "analyst_downgrade",
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO news_stories (builder_version, story_id, symbol, session, start_case, "
                "type_detect, type_close, follower_close, n_items, n_distinct, items, flags) "
                "VALUES ('stories-v1', :id, :s, :d, 'out', :t, :tc, false, :n, :n, "
                "CAST(:items AS jsonb), '{}')"
            ),
            {
                "id": f"{symbol}:{session.isoformat()}",
                "s": symbol,
                "d": session,
                "t": type_close,
                "tc": type_close,
                "n": len(event_ids),
                "items": json.dumps([{"event_id": e, "itype": itype} for e in event_ids]),
            },
        )


@pytest.mark.usefixtures("small_map")
async def test_d3_needs_the_renamed_news_and_news_for_every_halal_name(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(h1, "D3_SPAN", (date(2017, 1, 1), date(2017, 6, 30)))
    await _screen(
        engine,
        date(2016, 12, 30),
        [("AAA", 1, "X"), ("BBB", 2, "X"), ("CCC", None, "X")],  # CCC: not halal
    )
    await _monthly(engine, ["AAA", "BBB", "CCC"], date(2016, 1, 1), date(2017, 6, 1))
    q1, q2 = date(2017, 2, 7), date(2017, 5, 9)
    ids = await store(
        engine,
        [news_row(1, "AAA", ny(q1, 8), "Acme news"), news_row(2, "AAA", ny(q2, 8), "Acme news")],
        facts=False,
    )
    await _story_row(engine, "AAA", q1, [ids["alpaca:1"]])
    await _story_row(engine, "AAA", q2, [ids["alpaca:2"]])
    async with engine.begin() as conn:  # BBB: only a filing in Q1, which is not news
        await conn.execute(
            text(
                "INSERT INTO events (source, source_id, kind, symbol, published_at, seen_at, "
                "payload) VALUES ('sec', 'acc-1', '8-k', 'BBB', :t, :t, '{}')"
            ),
            {"t": ny(q1, 18)},
        )
        filing = await conn.scalar(text("SELECT id FROM events WHERE source_id = 'acc-1'"))
    await _story_row(
        engine, "BBB", q1, [int(filing)], itype="filing_other", type_close="filing_other"
    )
    bad = await h1.check_news_d3(engine)
    assert not bad.ok and "renamed-news month(s) not fetched" in bad.detail
    quarters = bad.data["quarters"]
    assert quarters["2017-01-01"] == {
        "names": 2,
        "silent": ["BBB"],
        "share": 0.5,
        "screen": "2016-12-30",
    }
    await mark_renamed_news_done(engine)
    ids = await store(
        engine,
        [news_row(3, "BBB", ny(q1, 9), "Bbb news"), news_row(4, "BBB", ny(q2, 9), "Bbb news")],
        facts=False,
    )
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM news_stories WHERE symbol = 'BBB'"))
    await _story_row(engine, "BBB", q1, [int(filing), ids["alpaca:3"]])
    await _story_row(engine, "BBB", q2, [ids["alpaca:4"]])
    ok = await h1.check_news_d3(engine)
    assert ok.ok, ok.detail
    assert {q: r["silent"] for q, r in ok.data["quarters"].items()} == {
        "2017-01-01": [],
        "2017-04-01": [],
    }


class _Plan:
    def __init__(self, parts: dict[str, set[tuple[str, date]]]) -> None:
        self.parts = parts


def _days(first: date, n: int) -> list[date]:
    out = [first]
    while len(out) < n:
        out.append(next_trading_day(out[-1]))
    return out


async def test_d4_needs_99_percent_of_each_part_and_every_spy_session(engine: AsyncEngine) -> None:
    days = _days(date(2017, 3, 1), 100)
    train = {(f"N{i:03d}", d) for i, d in enumerate(days)}
    spy = {("SPY", d) for d in days[:3]}
    plan = _Plan({"spy": spy, "train": train})
    for d in days[:3]:
        await seed_bars(engine, "SPY", session_bars(d, price=200.0))
    await mark_done(engine, sorted(spy) + sorted(train)[:99])
    ok = await h1.check_units_d4(engine, plan, await minutes.done_units(engine))
    assert ok.ok, ok.detail
    assert ok.data["parts"]["train"] == {"units": 100, "done": 99, "share": 0.99}
    short = ("SPY", days[3])
    await seed_bars(engine, "SPY", session_bars(days[3], price=200.0, last=(14, 28)))  # 299 bars
    await mark_done(engine, [short])
    plan.parts["spy"] = {*spy, short}
    bad = await h1.check_units_d4(engine, plan, await minutes.done_units(engine))
    assert not bad.ok and bad.data["spy_short"] == [days[3].isoformat()]
    plan.parts["gate_g1"] = {("ZZZ", days[0])}
    worse = await h1.check_units_d4(engine, plan, await minutes.done_units(engine))
    assert "gate_g1 0.00% done" in worse.detail
    assert h1.spy_complete(date(2016, 11, 25), 150) and not h1.spy_complete(date(2016, 11, 28), 299)


async def test_d5_scans_every_done_unit_through_the_readers(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = date(2017, 3, 7)
    await seed_bars(engine, "AAA", session_bars(d, price=50.0))
    outside = session_bars(d, price=50.0, first=(9, 30), last=(9, 30))
    early = BarArrays(
        outside.ts - 3600, outside.o, outside.h, outside.l, outside.c, outside.v, outside.vw
    )
    await seed_bars(engine, "AAA", early)  # a stored pre-market bar the readers must leave out
    await mark_done(engine, [("AAA", d)])
    done = await minutes.done_units(engine)
    ok = await h1.check_readers_d5(engine, {("AAA", d), ("BBB", d)}, done)
    assert ok.ok, ok.detail
    assert (ok.data["units"], ok.data["bars"], ok.data["sampled_read"]) == (1, 390, 1)

    real = minutes.read_windows

    async def leaky(engine_: Any, units: Any) -> dict[tuple[str, date], BarArrays]:
        got = await real(engine_, units)
        return {
            u: BarArrays(
                np.append(a.ts, a.ts[-1] + 3600),
                *(np.append(x, x[-1]) for x in (a.o, a.h, a.l, a.c, a.v, a.vw)),
            )
            for u, a in got.items()
        }

    monkeypatch.setattr(minutes, "read_windows", leaky)
    bad = await h1.check_readers_d5(engine, {("AAA", d)}, done)
    assert not bad.ok and bad.data["outside_read_windows"] == 1


async def test_d6_needs_every_news_event_read_and_reports_the_202_rate(engine: AsyncEngine) -> None:
    day = date(2017, 3, 7)
    ids = await store(
        engine, [news_row(1, "AAA", ny(day, 8), "Acme Q4 EPS $1.00 Beats $0.90 Estimate")]
    )
    await _story_row(
        engine, "AAA", day, [ids["alpaca:1"]], itype="earnings_8k", type_close="earnings_unparsed"
    )
    await _story_row(
        engine, "BBB", day, [ids["alpaca:1"]], itype="earnings_8k", type_close="earnings_beat"
    )
    ok = await h1.check_facts_d6(engine)
    assert ok.ok, ok.detail
    assert (ok.data["stories_202"], ok.data["stories_202_unparsed"]) == (2, 1)
    await store(engine, [news_row(2, "AAA", ny(day, 9), "no facts stored")], facts=False)
    bad = await h1.check_facts_d6(engine)
    assert not bad.ok and bad.data["news_unread"] == 1


# ── registration ─────────────────────────────────────────────


def _report(*, fail: str | None = None, drop: str | None = None) -> Preconditions:
    checks = []
    for i in h1.REQUIRED_CHECKS:
        if i == drop:
            continue
        data = {"commit": "c" * 40} if i == "C0" else {}
        checks.append(Check(i, i != fail, f"{i} detail", data))
    return Preconditions(tuple(checks))


async def test_registration_needs_every_check_and_happens_once(engine: AsyncEngine) -> None:
    with pytest.raises(RegistrationRefused, match="D3: D3 detail"):
        await h1.register(engine, report=_report(fail="D3"))
    with pytest.raises(RegistrationRefused, match="D5: not run"):
        await h1.register(engine, report=_report(drop="D5"))
    with pytest.raises(H1Locked, match="not registered"):
        await h1.registration(engine)
    assert await h1.existing_registration(engine) is None
    trial_id = await h1.register(engine, report=_report())
    reg = await h1.registration(engine)
    assert reg.id == trial_id == await h1.existing_registration(engine)
    assert reg.commit == "c" * 40 and reg.prereg == await h1.prereg(engine)
    async with engine.connect() as conn:
        row = (
            await conn.execute(text("SELECT * FROM quant_trials WHERE id = :i"), {"i": trial_id})
        ).one()
    assert (row.name, row.kind, row.verdict, row.criterion) == (
        "research.news.h1",
        "preregistration",
        None,
        h1.CRITERION,
    )
    assert row.config_hash == config_hash(reg.prereg)
    assert set(row.metrics["checks"]) == set(h1.REQUIRED_CHECKS)
    assert (
        row.metrics["files"]["halal_trader.events.h1"] == h1.file_shas()["halal_trader.events.h1"]
    )
    assert "active_sr_period" not in row.metrics  # never a trial
    with pytest.raises(RegistrationRefused, match="registered already"):
        await h1.register(engine, report=_report())


async def test_a_changed_alias_is_a_new_configuration(engine: AsyncEngine) -> None:
    await h1.register(engine, report=_report())
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO story_aliases (builder_version, symbol, alias, source) "
                "VALUES ('stories-v1', 'AAA', 'Acme', 'name')"
            )
        )
    with pytest.raises(H1Locked, match="alias_sha"):
        await h1.registration(engine)


async def test_without_plan_h_d4_and_d5_fail(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "halal_trader.events.units", None)  # import fails

    async def ok_check(*_: Any, **__: Any) -> Check:
        return Check("X", True, "")

    for name in (
        "check_calendar_d1",
        "check_screens_d2",
        "check_news_d3",
        "check_facts_d6",
        "story_counts_d9",
    ):
        monkeypatch.setattr(h1, name, ok_check)
    report = await h1.preconditions(engine, code=CodeState("c" * 40, (h1.TAG,), False))
    d4, d5 = report.get("D4"), report.get("D5")
    assert d4 is not None and not d4.ok and "plan H is not available" in d4.detail
    assert d5 is not None and not d5.ok and not report.ok
    gates = report.get("G")
    assert gates is not None and not gates.ok  # no gate row at all


async def test_gate_rows_are_read_from_the_ledger(engine: AsyncEngine) -> None:
    repo = QuantTrialRepoImpl(engine)
    for gate, verdict in (
        ("lookahead", "pass"),
        ("reactor", "fail"),
        ("reactor", "pass"),
        ("sue", "pass"),
    ):
        await repo.record_trial(
            name=h1.GATE_NAME, kind="gate", config={"sim": {}, "gate": gate}, verdict=verdict
        )
    await repo.record_trial(
        name="research.news.sim-gate.units", kind="gate-units", config={"gate": "g1"}
    )
    rows = await h1.gate_rows(engine)
    assert sorted(g for g, _, _ in rows) == ["lookahead", "reactor", "reactor", "sue"]
    assert judge_gates(rows).ok
