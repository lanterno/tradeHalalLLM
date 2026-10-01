"""S1 first-look backtest: scoring, rebalancing, costs, drift, stats -- on synthetic prices."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from halal_trader.research.factor_backtest import (
    Prices,
    backtest,
    month_ends,
    scores_at,
    stats,
)


def _days(n: int) -> list[date]:
    out, d = [], date(2020, 1, 1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _prices(n: int, drifts: dict[str, float], noise: dict[str, float] | None = None) -> Prices:
    rng = np.random.default_rng(7)
    noise = noise or {}
    cols = []
    for sym, mu in drifts.items():
        shocks = rng.normal(0.0, noise.get(sym, 0.0), n)
        cols.append(100.0 * np.cumprod(1.0 + mu + shocks))
    return Prices(_days(n), list(drifts), np.column_stack(cols))


def test_month_ends_are_last_sessions() -> None:
    days = [date(2020, 1, 30), date(2020, 1, 31), date(2020, 2, 3), date(2020, 2, 28)]
    assert month_ends(days) == [1, 3]


def test_momentum_and_low_volatility_both_score() -> None:
    p = _prices(300, {"WIN": 0.002, "FLAT": 0.0, "WILD": 0.002}, noise={"WILD": 0.04})
    s = scores_at(p.close, 299)
    assert s[0] > s[1]  # momentum beats flat
    assert s[0] > s[2]  # same trend, lower volatility wins


def test_no_score_before_a_year_of_history() -> None:
    p = _prices(100, {"A": 0.001, "B": 0.0})
    assert np.isnan(scores_at(p.close, 99)).all()


def test_backtest_holds_the_best_eligible_name_and_pays_costs() -> None:
    p = _prices(400, {"WIN": 0.002, "LOSE": -0.001, "HARAM": 0.004})
    free = backtest(p, eligible={"WIN", "LOSE"}, top_n=1, cost_bps=0)
    costly = backtest(p, eligible={"WIN", "LOSE"}, top_n=1, cost_bps=50)

    assert all(h == ["WIN"] for h in free.holdings.values())  # HARAM never held
    assert free.stats.total_return > costly.stats.total_return  # costs bite
    first_rebalance_cost = 1.0 * 50 / 10_000  # 100% turnover into WIN once
    assert free.stats.total_return == pytest.approx(
        (1 + costly.stats.total_return) / (1 - first_rebalance_cost) - 1, rel=1e-6
    )


def test_stats_on_known_returns() -> None:
    s = stats(np.array([0.10, -0.20, 0.05]))
    assert s.total_return == pytest.approx(1.10 * 0.80 * 1.05 - 1)
    assert s.max_drawdown == pytest.approx(-0.20)
