"""The path atlas (events/atlas.py), pure parts: measures, regimes, cells, tables, stories.

Synthetic bars and rows only; the database side is ``test_news_atlas_db.py``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from halabot.playbooks.records import StoryOutcome
from halabot.playbooks.sim import start_time
from halabot.playbooks.types import Session
from halal_trader.data.minutes import BarArrays
from halal_trader.events import atlas
from halal_trader.events.aliases import AliasMatcher
from halal_trader.events.atlas import (
    BUCKETS,
    MIN_DATES,
    MIN_N,
    Atlas,
    AtlasRow,
    AtlasStory,
    MachineRun,
    PathMeasures,
    bucket,
    cell_stats,
    cells_of,
    check_range,
    context_end,
    detection,
    path_measures,
    spy_regimes,
    substantive,
    tables,
    timing_of,
    to_json,
    write_atlas,
)
from halal_trader.events.earnings_parse import parse_headline
from halal_trader.events.stories import NEWS_LAG, RawItem, build
from halal_trader.market_hours import MARKET_TZ, next_trading_day
from tests.halabot.playbooks._support import Row, session_bars
from tests.halabot.playbooks._support import et as _et

S = date(2017, 3, 7)  # a Tuesday
S1, S2 = date(2017, 3, 8), date(2017, 3, 9)
SESSION = Session.of(S)
EPS = 1e-12


def et(day: date, hh: int, mm: int = 0, ss: int = 0) -> datetime:
    """A New York wall-clock time on ``day``, as UTC."""
    return _et(day, hh, mm, ss)


def spy_flat(day: date, price: float = 200.0) -> BarArrays:
    return session_bars(day, price=price)


def bar(o: float, h: float, low: float, c: float, v: float = 1_000.0) -> Row:
    return (o, h, low, c, v, c)


# ── path measures ─────────────────────────────────────────────


def _out_path() -> list[BarArrays]:
    s_rows = {(9, 40): bar(96.0, 96.0, 94.0, 94.5)}
    return [
        session_bars(S, price=96.0, rows=s_rows),
        session_bars(S1, price=97.0),  # a 50% retrace on S+1
        session_bars(S2, price=96.5, rows={(11, 0): bar(96.5, 96.5, 93.0, 93.5)}),
    ]


def test_an_out_story_is_measured_from_the_previous_close() -> None:
    m = path_measures(
        _out_path(),
        [1.0, 1.0, 1.0],
        spy_flat(S),
        session=SESSION,
        ref_at=et(S, 8, 0),
        start_case="out",
        prev_close_s=100.0,
        spy_prev_close_s=200.0,
        sigma=0.01,
    )
    assert m is not None
    assert (m.p0, m.spy0) == (100.0, 200.0)
    assert m.anchor_ts == int(et(S, 9, 30).timestamp())
    assert m.gap_sigma == pytest.approx(-4.0, abs=EPS)  # 96 against 100, sigma 1%
    assert m.low_sigma == pytest.approx(-5.5, abs=EPS)  # the 94.5 close
    assert (m.low, m.t_low) == (94.0, 10.0)
    assert m.retrace_close == pytest.approx(2.0 / 6.0, abs=EPS)
    assert m.retrace_max_s == pytest.approx(2.0 / 6.0, abs=EPS)
    assert m.retrace_max_s2 == pytest.approx(0.5, abs=EPS)
    assert m.fade is True  # S+2's low of 93 comes after a retrace of 0.25


def test_spy_moves_are_taken_out_bar_by_bar() -> None:
    spy = session_bars(S, price=200.0, rows={(9, 40): bar(202.0, 202.0, 202.0, 202.0)})
    m = path_measures(
        _out_path(),
        [1.0] * 3,
        spy,
        session=SESSION,
        ref_at=et(S, 8),
        start_case="out",
        prev_close_s=100.0,
        spy_prev_close_s=200.0,
        sigma=0.01,
    )
    assert m is not None
    # 09:40: the stock -5.5%, SPY +1%: abnormal -6.5%. From 09:41 SPY is back at 200.
    assert m.low_sigma == pytest.approx(-6.5, abs=EPS)


def test_an_in_story_takes_its_references_from_the_bars_before_the_news() -> None:
    rows = {(10, 59): bar(51.0, 51.0, 51.0, 51.0), (11, 0): bar(50.0, 50.0, 48.0, 48.5)}
    rows |= {(11, m): bar(48.5, 48.6, 48.4, 48.5) for m in range(1, 60)}
    spy_rows = {(10, 59): bar(201.0, 201.0, 201.0, 201.0)}
    s_bars = session_bars(S, price=50.0, rows=rows)
    m = path_measures(
        [s_bars],
        [1.0],
        session_bars(S, price=200.0, rows=spy_rows),
        session=SESSION,
        ref_at=et(S, 11, 0, 30),
        start_case="in",
        prev_close_s=40.0,
        spy_prev_close_s=150.0,
        sigma=0.02,
    )
    assert m is not None
    # P0 is the last bar with ts + 60 s <= 11:00:30 (10:59), SPY0 SPY's 10:59 close.
    assert (m.p0, m.spy0) == (51.0, 201.0)
    assert m.anchor_ts == int(et(S, 11, 0).timestamp())
    assert m.gap_sigma == pytest.approx(((48.5 / 51.0 - 1.0) - (200.0 / 201.0 - 1.0)) / 0.02)
    assert m.t_low == 0.0  # L is 48.0, set on the anchor bar itself


def test_news_before_the_first_bar_uses_the_previous_closes() -> None:
    m = path_measures(
        [session_bars(S, price=50.0)],
        [1.0],
        spy_flat(S),
        session=SESSION,
        ref_at=et(S, 9, 30, 20),
        start_case="in",
        prev_close_s=52.0,
        spy_prev_close_s=199.0,
        sigma=0.02,
    )
    assert m is not None and (m.p0, m.spy0) == (52.0, 199.0)


def test_the_latest_bar_at_the_low_sets_t_low() -> None:
    rows = {(9, 35): bar(99.0, 99.0, 95.0, 96.0), (10, 15): bar(96.0, 96.0, 95.0, 96.0)}
    m = path_measures(
        [session_bars(S, price=97.0, rows=rows)],
        [1.0],
        spy_flat(S),
        session=SESSION,
        ref_at=et(S, 8),
        start_case="out",
        prev_close_s=100.0,
        spy_prev_close_s=200.0,
        sigma=0.01,
    )
    assert m is not None and m.t_low == 45.0


def test_without_a_drop_the_retrace_is_undefined() -> None:
    m = path_measures(
        [session_bars(S, price=101.0)],
        [1.0],
        spy_flat(S),
        session=SESSION,
        ref_at=et(S, 8),
        start_case="out",
        prev_close_s=100.0,
        spy_prev_close_s=200.0,
        sigma=0.01,
    )
    assert m is not None
    assert (m.retrace_close, m.retrace_max_s, m.retrace_max_s2, m.fade) == (None,) * 4


def test_later_sessions_are_put_in_session_s_units() -> None:
    # A 2-for-1 split effective on S+1: raw prices halve, A(S+1)/A(S) = 2.
    path = [
        session_bars(S, price=96.0, rows={(9, 40): bar(96.0, 96.0, 94.0, 94.5)}),
        session_bars(S1, price=48.5),  # 97 in S units: a 50% retrace
    ]
    m = path_measures(
        path,
        [1.0, 2.0],
        spy_flat(S),
        session=SESSION,
        ref_at=et(S, 8),
        start_case="out",
        prev_close_s=100.0,
        spy_prev_close_s=200.0,
        sigma=0.01,
    )
    assert m is not None
    assert m.retrace_max_s2 == pytest.approx(0.5, abs=EPS)
    assert m.fade is False  # 48.5 * 2 never trades below 94


def test_no_bar_from_the_anchor_on_is_no_measure() -> None:
    m = path_measures(
        [session_bars(S, price=50.0, last=(10, 0))],
        [1.0],
        spy_flat(S),
        session=SESSION,
        ref_at=et(S, 11, 0),
        start_case="in",
        prev_close_s=50.0,
        spy_prev_close_s=200.0,
        sigma=0.01,
    )
    assert m is None


@pytest.mark.parametrize(
    ("low", "label"),
    [
        (-7.0, "(-inf,-5]"),
        (-5.0, "(-inf,-5]"),
        (-4.99, "(-5,-3]"),
        (-3.0, "(-5,-3]"),
        (-2.5, "(-3,-2]"),
        (-1.0, "(-2,-1]"),
        (-0.5, "(-1,inf)"),
        (0.3, "(-1,inf)"),
    ],
)
def test_low_sigma_buckets_hold_their_upper_edge(low: float, label: str) -> None:
    assert bucket(low) == label


# ── SPY regimes ───────────────────────────────────────────────


def test_spy_regimes_use_only_closes_before_the_session() -> None:
    rng = np.random.default_rng(7)
    sessions = [date(2016, 1, 4) + timedelta(days=i) for i in range(400)]
    closes: list[float | None] = list(200.0 * np.cumprod(1 + rng.normal(0, 0.01, 400)))
    reg = spy_regimes(sessions, closes, edges_from=sessions[250], edges_to=sessions[-1])
    j = 300
    window = np.asarray(closes[j - 21 : j], dtype=float)
    assert reg.vol[sessions[j]] == pytest.approx(
        float(np.std(window[1:] / window[:-1] - 1, ddof=1))
    )
    sma = np.asarray(closes[j - 200 : j], dtype=float)
    assert reg.above[sessions[j]] == (closes[j - 1] > sma.mean())  # type: ignore[operator]
    # A change to S's own close changes nothing known at S.
    moved = list(closes)
    moved[j] = 10.0 * (closes[j] or 0.0)
    again = spy_regimes(sessions, moved, edges_from=sessions[250], edges_to=sessions[-1])
    assert again.vol[sessions[j]] == reg.vol[sessions[j]]
    assert again.above[sessions[j]] == reg.above[sessions[j]]
    inside = [v for d, v in reg.vol.items() if d >= sessions[250]]
    assert reg.edges == tuple(np.quantile(inside, [1 / 3, 2 / 3]))
    terciles = {reg.vol_tercile(d) for d in sessions[250:]}
    assert terciles == {"low", "mid", "high"}
    assert reg.vol_tercile(sessions[5]) == "n/a" and reg.trend(sessions[150]) == "n/a"


def test_a_missing_close_leaves_its_windows_unknown() -> None:
    sessions = [date(2016, 1, 4) + timedelta(days=i) for i in range(30)]
    closes: list[float | None] = [200.0 + i for i in range(30)]
    closes[25] = None
    reg = spy_regimes(sessions, closes, edges_from=sessions[0], edges_to=sessions[-1])
    assert sessions[26] not in reg.vol and sessions[25] in reg.vol  # S-1 = 25 is needed


# ── rows and cells ────────────────────────────────────────────


def measures(low_sigma: float = -3.5, **kw: Any) -> PathMeasures:
    base: dict[str, Any] = {
        "p0": 100.0,
        "spy0": 200.0,
        "anchor_ts": 0,
        "gap_sigma": -1.0,
        "low_sigma": low_sigma,
        "t_low": 30.0,
        "low": 95.0,
        "retrace_close": 0.3,
        "retrace_max_s": 0.4,
        "retrace_max_s2": 0.6,
        "fade": False,
    }
    return PathMeasures(**(base | kw))


def row(i: int, session: date, **kw: Any) -> AtlasRow:
    at = datetime.combine(session, datetime.min.time(), MARKET_TZ).astimezone(UTC)
    base: dict[str, Any] = {
        "story_id": f"S{i:03d}:{session}",
        "symbol": f"S{i:03d}",
        "session": session,
        "type": "analyst_downgrade",
        "type_close": "analyst_downgrade",
        "family_ever": "NSN_CORE",
        "direction": "neg",
        "follower": False,
        "nsn": True,
        "start_case": "out",
        "detect_at": at,
        "ref_at": at,
        "timing": "pre_open",
        "rank": 100,
        "tech": False,
        "cost_bps": 7.0,
        "spy_vol": "mid",
        "spy_trend": "above",
        "sigma": 0.02,
        "skip": None,
        "measures": measures(),
        "cont": (0.01, 0.02, 0.03),
        "machine": True,
    }
    return AtlasRow(**(base | kw))


def sessions_from(first: date, n: int) -> list[date]:
    out = [first]
    while len(out) < n:
        out.append(next_trading_day(out[-1]))
    return out


DAYS = sessions_from(date(2017, 1, 3), 60)


def test_a_cell_under_thirty_stories_or_twenty_dates_shows_counts_only() -> None:
    small = [row(i, DAYS[i]) for i in range(MIN_N - 1)]
    few_dates = [row(i, DAYS[i % (MIN_DATES - 1)]) for i in range(MIN_N)]
    enough = [row(i, DAYS[i % MIN_DATES]) for i in range(MIN_N)]
    for rows, shown in ((small, False), (few_dates, False), (enough, True)):
        cell = next(c for c in cells_of(rows, years=1.0) if c.table == "type")
        assert (cell.stats is not None) == shown
        assert cell.n == len(rows) and cell.dates == len({r.session for r in rows})
    assert next(c for c in cells_of(enough, years=2.0) if c.table == "type").per_year == 15.0


def _cr1(values: Sequence[float], clusters: Sequence[date]) -> float:
    """CR1 by cluster, as spec §G.8 writes it."""
    n, mean = len(values), sum(values) / len(values)
    groups: dict[date, float] = {}
    for v, g in zip(values, clusters):
        groups[g] = groups.get(g, 0.0) + (v - mean)
    big_g = len(groups)
    return math.sqrt(big_g / (big_g - 1) * sum(e * e for e in groups.values()) / n**2)


def _run(entered: bool, exit_reason: str | None, r: float | None, **kw: Any) -> MachineRun:
    base: dict[str, Any] = {
        "state": "EXITED" if entered else "EXPIRED",
        "reason": exit_reason or "cutoff",
        "skip": None,
        "triggered": True,
        "armed": entered,
        "entered": entered,
        "exit_reason": exit_reason,
        "r": r,
    }
    return MachineRun(**(base | kw))


def test_the_cell_statistics() -> None:
    rng = np.random.default_rng(3)
    rows = []
    for i in range(40):
        drop = i % 4 != 0  # a quarter have no drop: their retrace is undefined
        m = measures(
            low_sigma=float(rng.normal(-3, 1)),
            t_low=float(i),
            retrace_close=float(rng.uniform(0, 1.2)) if drop else None,
            retrace_max_s=float(rng.uniform(0, 1.2)) if drop else None,
            retrace_max_s2=float(rng.uniform(0, 1.5)) if drop else None,
            fade=(i % 3 == 0) if drop else None,
        )
        cont = (float(rng.normal()), None if i == 5 else float(rng.normal()), 0.0)
        reason = ("target", "stop", "time_stop")[i % 3]
        run = (
            _run(True, reason, float(rng.normal(0, 0.02)))
            if i % 2
            else _run(False, None, None, triggered=i % 4 == 0)
        )
        if i == 38:
            run = _run(False, None, None, state="DISMISSED", reason="blocked_open")
        rows.append(row(i, DAYS[i % 25], measures=m, cont=cont, id_run=run))
    stats = cell_stats(rows)
    lows = [r.measures.low_sigma for r in rows if r.measures]
    assert stats["low_sigma_q10"] == pytest.approx(float(np.quantile(lows, 0.10)))
    assert stats["t_low_q50"] == pytest.approx(19.5)
    drops = [r.measures for r in rows if r.measures and r.measures.retrace_max_s is not None]
    assert stats["drops"] == len(drops) == 30
    assert stats["p_retrace_0.50_close"] == pytest.approx(
        sum(m.retrace_max_s >= 0.5 for m in drops) / 30  # type: ignore[operator]
    )
    assert stats["p_retrace_1.00_s2"] == pytest.approx(
        sum(m.retrace_max_s2 >= 1.0 for m in drops) / 30  # type: ignore[operator]
    )
    fades = [m.fade for m in drops]
    assert stats["p_fade"] == pytest.approx(sum(bool(f) for f in fades) / len(fades))
    c3 = [(r.cont[1], r.session) for r in rows if r.cont[1] is not None]
    assert stats["cont_3_n"] == 39
    assert stats["cont_3_mean"] == pytest.approx(sum(v for v, _ in c3) / 39)  # type: ignore[misc]
    assert stats["cont_3_se"] == pytest.approx(_cr1([v for v, _ in c3], [d for _, d in c3]))  # type: ignore[misc]
    # The machine: the blocked story is not a run; P(trigger) over the runs.
    runs = [r.id_run for r in rows if r.id_run and r.id_run.ran]
    assert stats["id_runs"] == len(runs) == 39
    assert stats["id_p_trigger"] == pytest.approx(sum(m.triggered for m in runs) / 39)
    triggered = [m for m in runs if m.triggered]
    assert stats["id_p_entry_given_trigger"] == pytest.approx(
        sum(m.entered for m in triggered) / len(triggered)
    )
    trades = [(r.session, r.id_run) for r in rows if r.id_run and r.id_run.entered]
    assert stats["id_trades"] == 20
    targets = [(d, m.r) for d, m in trades if m.exit_reason == "target"]  # type: ignore[union-attr]
    assert stats["id_share_target"] == pytest.approx(len(targets) / 20)
    assert stats["id_r_mean_target"] == pytest.approx(sum(r for _, r in targets) / len(targets))  # type: ignore[misc]
    assert stats["id_r_se"] == pytest.approx(
        _cr1([m.r for _, m in trades], [d for d, _ in trades])  # type: ignore[union-attr,misc]
    )
    assert "id_r_mean_abort" not in stats and stats["id_share_abort"] == 0.0
    assert stats["md3_runs"] == 0.0 and "md3_p_trigger" not in stats


def test_a_value_that_is_not_finite_is_left_out_and_counted() -> None:
    rows = []
    for i in range(MIN_N):
        bad = i < 3
        m = measures(
            low_sigma=math.nan if i == 0 else -3.0 - i / 10,
            retrace_max_s=math.inf if i == 1 else 0.4,
        )
        cont = (math.nan if bad else 0.01 * i, 0.02, 0.03)
        run = _run(True, "target", math.nan if i == 2 else 0.001 * i)
        rows.append(row(i, DAYS[i % MIN_DATES], measures=m, cont=cont, id_run=run))
    stats = cell_stats(rows)  # no ValueError from the clustered means
    # One NaN low_sigma, one infinite retrace (its drop), three NaN cont_1, one NaN r.
    assert stats["nonfinite"] == 6
    assert stats["drops"] == MIN_N - 1
    assert stats["cont_1_n"] == MIN_N - 3
    finite = [0.01 * i for i in range(3, MIN_N)]
    assert stats["cont_1_mean"] == pytest.approx(sum(finite) / len(finite))
    assert math.isfinite(stats["cont_1_se"])
    rs = [0.001 * i for i in range(MIN_N) if i != 2]
    assert stats["id_trades"] == MIN_N  # the NaN trade is still a trade (and a target)
    assert stats["id_share_target"] == 1.0
    assert stats["id_r_mean"] == pytest.approx(sum(rs) / len(rs))
    assert stats["id_r_mean_target"] == stats["id_r_mean"]
    lows = sorted(-3.0 - i / 10 for i in range(1, MIN_N))
    assert stats["low_sigma_q50"] == pytest.approx(float(np.quantile(lows, 0.5)))
    # The whole table builds, and the file stays plain JSON.
    cells = cells_of(rows, years=1.0)
    assert next(c for c in cells if c.table == "type").stats is not None
    json.dumps(to_json(Atlas(rows=rows, cells=cells)), allow_nan=False)


def test_cells_cover_every_table_sorted_by_key() -> None:
    rows = [
        row(1, DAYS[0], type="earnings_miss", measures=measures(-6.0), timing="in_session"),
        row(2, DAYS[1], measures=measures(-0.2), rank=500, tech=True),
        row(3, DAYS[2], measures=measures(-4.0), spy_vol="high"),
        row(4, DAYS[3], measures=None, skip="units_missing"),
        row(5, date(2021, 3, 1), measures=measures(-2.5), spy_trend="below"),
    ]
    cells = cells_of(rows, years=1.0)
    assert [c.table for c in cells] == sorted((c.table for c in cells), key=atlas.TABLES.index)
    base = [c.key for c in cells if c.table == "base"]
    assert base == [
        ("analyst_downgrade", "(-5,-3]"),
        ("analyst_downgrade", "(-3,-2]"),
        ("analyst_downgrade", "(-1,inf)"),
        ("earnings_miss", "(-inf,-5]"),
    ]
    assert [c.key[1] for c in cells if c.table == "base"][0] in BUCKETS
    assert {c.key: c.n for c in cells if c.table == "rank"} == {
        ("analyst_downgrade", "rank<300"): 2,
        ("analyst_downgrade", "rank300-999"): 1,
        ("earnings_miss", "rank<300"): 1,
    }
    regimes = {c.key: c.n for c in cells if c.table == "screen_regime"}
    assert regimes[("analyst_downgrade", "before")] == 2
    assert regimes[("analyst_downgrade", "from")] == 1
    analyst = {c.key: c.n for c in cells if c.table == "analyst_regime"}
    assert analyst[("analyst_downgrade", "before")] == 2  # 2017
    assert analyst[("analyst_downgrade", "from")] == 1  # 2021
    # The skipped row is counted in the coverage table only.
    coverage = {c.key: c.n for c in cells if c.table == "coverage"}
    assert coverage == {
        ("analyst_downgrade", "measured"): 3,
        ("analyst_downgrade", "units_missing"): 1,
        ("earnings_miss", "measured"): 1,
    }
    assert next(c for c in cells if c.key == ("analyst_downgrade",)).n == 3


def test_tables_print_counts_only_cells_and_never_a_p_value() -> None:
    rows = [row(i, DAYS[i % MIN_DATES]) for i in range(MIN_N)]
    rows.append(row(99, DAYS[0], type="dilution", measures=measures(-1.5)))
    lines = tables(cells_of(rows, years=1.0))
    text_ = "\n".join(lines)
    assert "== type x low_sigma bucket ==" in text_
    assert "dilution / (-2,-1]" in text_ and "counts only" in text_
    assert "p-value" not in text_.replace("no p-values", "")
    downgrade = next(line for line in lines if line.startswith("analyst_downgrade  "))
    assert "counts only" not in downgrade


def test_the_file_is_plain_json(tmp_path: Path) -> None:
    rows = [row(1, DAYS[0], cont=(None, math.nan, 0.1), id_run=_run(True, "stop", -0.02))]
    result = Atlas(rows=rows, cells=cells_of(rows, years=1.0), meta={"end": DAYS[0]})
    path = write_atlas(result, tmp_path / "x" / "atlas.json")
    data = json.loads(path.read_text())
    assert data["meta"]["end"] == DAYS[0].isoformat()
    assert data["rows"][0]["cont"] == {"cont_1": None, "cont_3": None, "cont_5": 0.1}
    assert data["rows"][0]["id_run"]["exit_reason"] == "stop"
    assert data["rows"][0]["session"] == DAYS[0].isoformat()
    assert to_json(result)["cells"][0]["stats"] is None


# ── machine outcomes ──────────────────────────────────────────


def test_a_machine_run_reads_the_outcome_and_its_trade() -> None:
    # Only the trade's exit reason, return and flags are read.
    trade = SimpleNamespace(exit_reason="target", r_net_abn=0.0123, flags=("gap_fill",))
    o = StoryOutcome(
        "A:2017-03-07",
        "A",
        S,
        "EXITED",
        "target",
        None,
        et(S, 10),
        et(S, 10, 20),
        trade,  # type: ignore[arg-type]
        entry_bar_ts=et(S, 10, 30),
    )
    run = MachineRun.of(o)
    assert (run.triggered, run.armed, run.entered, run.ran) == (True, True, True, True)
    assert (run.exit_reason, run.r, run.flags) == ("target", 0.0123, ("gap_fill",))
    blocked = StoryOutcome(
        "A:2017-03-08", "A", S1, "DISMISSED", "blocked_open", None, None, None, None
    )
    skipped = StoryOutcome(
        "A:2017-03-09", "A", S2, "SKIPPED", "units_missing", "units_missing", None, None, None
    )
    assert not MachineRun.of(blocked).ran and not MachineRun.of(skipped).ran
    assert MachineRun.of(blocked).exit_reason is None
    assert MachineRun.of(skipped).skip == "units_missing"


# ── stories as the machine sees them ──────────────────────────

ALIASES = {"ALFA": AliasMatcher("ALFA", ("Alfa",), ("ALFA",))}


def _news(n: int, at: datetime, headline: str) -> RawItem:
    return RawItem(
        n,
        f"alpaca:{n}",
        "news",
        "ALFA",
        at.astimezone(UTC),
        at.astimezone(UTC),
        headline,
        1,
        (),
        tuple(parse_headline(headline)),
    )


def test_a_story_that_is_not_nsn_starts_at_its_first_substantive_item() -> None:
    [story] = build(
        [
            _news(1, et(S, 9, 45), "Earnings Preview: Alfa"),  # noise: never the detection
            _news(2, et(S, 11, 0), "Alfa Prices $50M Public Offering"),
        ],
        ALIASES,
    )
    found = detection(story)
    assert found == (et(S, 11, 0) + NEWS_LAG, et(S, 11, 0), False)
    detect, at, _ = found
    view = AtlasStory(story, detect, at)
    assert view.nsn_at(SESSION.entry_cutoff) == detect and view.at_news() == at
    assert view.start_case() == "in" and story.at_news() is None
    assert start_time(view) == detect
    assert view.card_at(detect).type == "dilution"
    assert view.news_times() == story.news_times()
    news_only = AtlasStory.news_only(story)
    assert start_time(news_only) is None and news_only.start_case() == "out"
    assert news_only.news_times() == story.news_times()


def test_an_nsn_story_starts_as_h1_starts_it() -> None:
    [vetoed] = build(
        [
            _news(1, et(S, 7, 0), "Alfa Prices $50M Public Offering"),
            _news(2, et(S, 8, 0), "Morgan Stanley Downgrades Alfa to Equal-Weight"),
        ],
        ALIASES,
    )
    # The offering vetoes NSN: the detection is the first substantive item.
    assert vetoed.nsn_at(SESSION.entry_cutoff) is None
    assert detection(vetoed) == (et(S, 7) + NEWS_LAG, et(S, 7), False)
    [nsn] = build(
        [_news(3, et(S, 8, 0), "Morgan Stanley Downgrades Alfa to Equal-Weight")], ALIASES
    )
    found = detection(nsn)
    assert found == (nsn.nsn_at(SESSION.entry_cutoff), nsn.at_news(), True)
    assert found is not None
    view = AtlasStory(nsn, found[0], found[1])
    assert start_time(view) == start_time(nsn) == SESSION.open
    assert view.start_case() == nsn.start_case() == "out"


def test_a_story_of_noise_has_no_detection() -> None:
    [story] = build([_news(1, et(S, 9, 45), "Earnings Preview: Alfa")], ALIASES)
    assert detection(story) is None


@pytest.mark.parametrize(
    ("at", "timing"),
    [
        (et(S, 9, 30), "in_session"),
        (et(S, 15, 59), "in_session"),
        (et(S, 7, 0), "pre_open"),
        (et(S, 0, 5), "pre_open"),
        (et(date(2017, 3, 6), 18, 0), "evening_weekend"),
        (et(date(2017, 3, 6), 15, 0), "evening_weekend"),  # late news rolled to S
        (et(date(2017, 3, 4), 12, 0), "evening_weekend"),
    ],
)
def test_news_timing(at: datetime, timing: str) -> None:
    assert timing_of(at, S) == timing


def test_a_unit_has_an_item_outside_the_excluded_types() -> None:
    assert not substantive(["noise", "mover", "law_firm", "other", "analyst_other"])
    assert not substantive(["filing_other"])  # plan H never fetches these paths
    assert substantive(["noise", "product"]) and substantive(["earnings_unparsed"])


def test_the_range_never_leaves_train() -> None:
    check_range(date(2016, 10, 3), date(2021, 12, 23))
    with pytest.raises(ValueError, match="ends by 2021-12-23"):
        check_range(date(2016, 10, 3), date(2021, 12, 27))
    with pytest.raises(ValueError, match="starts on"):
        check_range(date(2016, 9, 30), date(2017, 1, 3))
    with pytest.raises(ValueError, match="before start"):
        check_range(date(2018, 1, 3), date(2017, 1, 3))
    # A weekend or a holiday end is refused before anything is read.
    with pytest.raises(ValueError, match="2019-06-30 is not a trading session.*2019-06-28"):
        check_range(date(2016, 10, 3), date(2019, 6, 30))
    with pytest.raises(ValueError, match="2017-07-04 is not a trading session.*2017-07-03"):
        check_range(date(2016, 10, 3), date(2017, 7, 4))
    # The context reads daily bars to its end + 7 days: never past 2021-12-31.
    assert context_end(date(2021, 12, 23)) == date(2021, 12, 24)
    assert context_end(date(2017, 3, 10)) == date(2017, 3, 17)
    for end in (date(2021, 12, 23), date(2021, 12, 17), date(2016, 10, 3)):
        assert context_end(end) >= end
        assert context_end(end) + timedelta(days=7) <= date(2021, 12, 31)
