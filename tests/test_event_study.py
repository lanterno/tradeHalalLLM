"""The event-study harness: clock-based entry, net abnormal outcomes, deciles; SUE."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from halal_trader.events.study import (
    Bars,
    Observation,
    cost_bps,
    entry_point,
    outcome,
    summarise,
)
from halal_trader.events.sue import Quarter, quarterly_series, sue_values

SESSIONS = [date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30)]


def _utc(d: int, h: int, m: int = 0) -> datetime:
    return datetime(2026, 9, d, h, m, tzinfo=UTC)


def test_entry_follows_the_clock() -> None:
    assert entry_point(_utc(28, 12), SESSIONS) == (0, "open")  # 08:00 ET: that open
    assert entry_point(_utc(28, 15), SESSIONS) == (0, "close")  # 11:00 ET: that close
    assert entry_point(_utc(28, 21), SESSIONS) == (1, "open")  # 17:00 ET: next open
    assert entry_point(_utc(27, 15), SESSIONS) == (0, "open")  # Sunday: Monday's open
    assert entry_point(_utc(30, 21), SESSIONS) is None


def test_outcomes_are_net_abnormal_returns() -> None:
    bars = Bars(
        SESSIONS,
        open={"X": {SESSIONS[1]: 100.0}, "SPUS": {SESSIONS[1]: 50.0}},
        close={
            "X": {SESSIONS[1]: 103.0, SESSIONS[2]: 110.0},
            "SPUS": {SESSIONS[1]: 50.5, SESSIONS[2]: 51.0},
        },
    )
    # Open entry, 1 session: same day's close. 3% - 1% - 2 x 7 bps.
    assert outcome(bars, "X", (1, "open"), 1, 7.0) == pytest.approx(0.03 - 0.01 - 0.0014)
    assert outcome(bars, "X", (1, "open"), 2, 7.0) == pytest.approx(0.10 - 0.02 - 0.0014)
    assert outcome(bars, "X", (1, "open"), 3, 7.0) is None  # not yet elapsed


def test_costs_rise_as_liquidity_falls() -> None:
    assert cost_bps(10) < cost_bps(500) < cost_bps(1500) == cost_bps(None)


def test_deciles_order_signal_and_report_ic() -> None:
    obs = [
        (Observation("X", _utc(28, 12), float(i), {"year": 2026}), 5, i / 1000) for i in range(200)
    ]
    result = summarise(obs)
    top = [r for r in result.rows if r.decile == 10][0]
    bottom = [r for r in result.rows if r.decile == 1][0]
    assert top.mean > bottom.mean and top.n == 20
    assert result.ic[("all", 5)] == pytest.approx(1.0)


def _q(year: int, q: int) -> tuple[date, date]:
    starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
    ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
    return date(year, *starts[q]), date(year, *ends[q])


def test_quarterly_series_keeps_first_filed_values_and_derives_q4() -> None:
    d = "EarningsPerShareDiluted"
    facts = []
    for q in (1, 2, 3):
        s, e = _q(2024, q)
        facts.append((d, s, e, 1.0, date(2024, 3 * q + 1, 30)))
    facts.append((d, *_q(2024, 1), 9.0, date(2025, 4, 30)))  # a later restatement: ignored
    facts.append((d, date(2024, 1, 1), date(2024, 12, 31), 5.0, date(2025, 2, 20)))
    series = quarterly_series(facts)
    assert [(q.end, q.eps) for q in series] == [
        (date(2024, 3, 31), 1.0),
        (date(2024, 6, 30), 1.0),
        (date(2024, 9, 30), 1.0),
        (date(2024, 12, 31), 2.0),  # 5 - 3 quarters
    ]
    assert series[-1].filed == date(2025, 2, 20)


def test_sue_is_the_seasonal_change_over_its_own_volatility() -> None:
    series = []
    eps = [1.0, 1.1, 1.2, 1.3, 1.1, 1.2, 1.3, 1.4, 1.3, 1.3, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 2.5]
    for i, v in enumerate(eps):
        year, q = 2020 + i // 4, i % 4 + 1
        series.append(Quarter(_q(year, q)[1], v, _q(year, q)[1]))
    values = sue_values(series)
    last_q, last_sue = values[-1]
    assert last_q.eps == 2.5 and last_sue > 5  # a jump far outside its usual seasonal change
