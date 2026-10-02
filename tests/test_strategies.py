"""Pre-registered strategies T3-T5: caps rolled forward, quality lagged, weights as specified."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.sec import Fact
from halal_trader.data.fundamentals import quality_by_year, sync_annual, usable_year
from halal_trader.research.factor_backtest import Prices, backtest_targets
from halal_trader.research.pit import Firm
from halal_trader.research.strategies import Inputs, cap_tilt, cap_weighted, three_factor


def _days(n: int, start: date = date(2020, 1, 1)) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _prices(n: int, drifts: dict[str, float], noise: dict[str, float] | None = None) -> Prices:
    rng = np.random.default_rng(3)
    noise = noise or {}
    cols = [
        100.0 * np.cumprod(1.0 + mu + rng.normal(0.0, noise.get(s, 0.005), n))
        for s, mu in drifts.items()
    ]
    return Prices(_days(n), list(drifts), np.column_stack(cols))


def test_annual_figures_are_used_only_once_every_10k_is_due() -> None:
    assert usable_year(date(2026, 4, 30)) == 2024
    assert usable_year(date(2026, 5, 1)) == 2025
    assert usable_year(date(2026, 12, 31)) == 2025


def test_caps_roll_forward_by_price_from_a_weekend_screen_date() -> None:
    p = Prices(
        [date(2021, 1, 1), date(2021, 1, 4), date(2021, 1, 5)],  # Fri, Mon, Tue
        ["A"],
        np.array([[10.0], [11.0], [20.0]]),
    )
    firms = {0: Firm(1, 1_000.0, date(2021, 1, 3))}  # screened on a Sunday: Friday's close
    caps = Inputs(p, {}, {}).caps(2, firms)
    assert caps[0] == pytest.approx(2_000.0)


def _inputs(prices: Prices, caps: dict[str, float], quality: dict[int, dict[int, float]]):
    firms = {s: Firm(i + 1, cap, prices.days[0]) for i, (s, cap) in enumerate(caps.items())}
    return Inputs(prices, {prices.days[0]: firms}, quality)


def test_t3_weights_are_proportional_to_market_cap_and_ineligible_names_get_none() -> None:
    p = _prices(300, {"BIG": 0.0, "SMALL": 0.0, "OUT": 0.0})
    w = cap_weighted(_inputs(p, {"BIG": 3e9, "SMALL": 1e9}, {}))(299)
    assert w[2] == 0.0
    assert w[0] / w[1] == pytest.approx(
        3.0 * p.close[299, 0] / p.close[0, 0] / (p.close[299, 1] / p.close[0, 1])
    )
    assert w.sum() == pytest.approx(1.0)


def test_t4_tilts_toward_the_better_composite_and_stays_fully_invested() -> None:
    p = _prices(300, {"WIN": 0.002, "LOSE": -0.001})
    caps = {"WIN": 1e9, "LOSE": 1e9}
    w = cap_tilt(_inputs(p, caps, {}))(299)
    plain = cap_weighted(_inputs(p, caps, {}))(299)
    assert w[0] > plain[0]
    assert w.sum() == pytest.approx(1.0)


def test_quality_counts_only_after_its_publication_date() -> None:
    p = _prices(600, {"HIGHQ": 0.0005, "LOWQ": 0.0005, "MID": 0.0005})  # identical prices
    p.close[:, 1] = p.close[:, 0]
    p.close[:, 2] = p.close[:, 0]
    caps = {"HIGHQ": 1e9, "LOWQ": 1e9, "MID": 1e9}
    quality = {2020: {1: 0.9, 2: 0.1, 3: 0.5}}
    inputs = _inputs(p, caps, quality)
    before = next(t for t, d in enumerate(p.days) if d >= date(2021, 4, 1))
    after = next(t for t, d in enumerate(p.days) if d >= date(2021, 5, 3))
    z_before = inputs.composite(before, inputs.firms(before))
    z_after = inputs.composite(after, inputs.firms(after))
    assert np.allclose(z_before, 0.0)  # 2020's figures are not public in April 2021
    assert z_after[0] > z_after[2] > z_after[1]
    picks = three_factor(inputs, top_n=1)(after)
    assert picks.tolist() == [1.0, 0.0, 0.0]


def test_uninvested_weight_is_cash_and_earns_nothing() -> None:
    p = _prices(300, {"A": 0.01}, noise={"A": 0.0})
    half = backtest_targets(p, lambda t: np.array([0.5]), cost_bps=0.0)
    full = backtest_targets(p, lambda t: np.array([1.0]), cost_bps=0.0)
    assert half.returns[1] == pytest.approx(full.returns[1] / 2)


class AnnualSec:
    async def frame(self, taxonomy: str, concept: str, unit: str, period: str) -> dict[int, Fact]:
        end = date(2024, 12, 31)
        table: dict[str, dict[int, float]] = {
            "GrossProfit": {1: 40.0},
            "Revenues": {1: 100.0, 2: 200.0},
            "CostOfRevenue": {2: 150.0},
            "Assets": {1: 100.0, 2: 500.0, 3: 50.0},
        }
        return {c: Fact(v, end, "a") for c, v in table.get(concept, {}).items()}


async def test_sync_annual_prefers_gross_profit_and_falls_back_to_revenue_minus_cost(
    engine: AsyncEngine,
) -> None:
    assert await sync_annual(AnnualSec(), engine, [2024]) == 3  # type: ignore[arg-type]
    assert await quality_by_year(engine) == {2024: {1: 0.4, 2: 0.1}}  # CIK 3: assets only
