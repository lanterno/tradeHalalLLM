"""Point-in-time portfolio rules beyond S1, pre-registered in the plan before any run.

All three hold only that month's eligible names (liquid then, halal then;
research/pit.py) and rebalance at month-ends through
``factor_backtest.backtest_targets``:

* ``cap_weighted`` (T3): weight proportional to market cap -- a self-built
  SPUS. Caps are the screen-date cap (price x shares) rolled forward by
  the adjusted price since, so no share count is read before it was filed.
* ``cap_tilt`` (T4): T3's weights times ``exp(TILT * z)``.
* ``three_factor`` (T5): equal weight on the top ``n`` by ``z``.

``z`` standardises the sum of three z-scores taken across the eligible
names: momentum 12-1, low volatility (63 sessions) and quality (gross
profit / assets from the newest annual figures certainly published;
data/fundamentals.py). A name with no quality figure scores 0 there.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
from datetime import date

import numpy as np

from halal_trader.data.fundamentals import usable_year
from halal_trader.research.factor_backtest import (
    FloatArray,
    Prices,
    TargetFn,
    equal_top,
    factor_parts,
    zscore,
)
from halal_trader.research.pit import Firm

TILT = 0.5
TOP_N = 50

Quality = Mapping[int, Mapping[int, float]]  # year -> CIK -> gross profit / assets


class Inputs:
    """Each rebalance's eligible names, caps and quality, aligned to the price matrix."""

    def __init__(
        self, prices: Prices, firms_from: Mapping[date, Mapping[str, Firm]], quality: Quality
    ) -> None:
        self.prices = prices
        self._schedule = sorted(firms_from.items())
        self._quality = quality
        self._col = {s: i for i, s in enumerate(prices.symbols)}

    def firms(self, t: int) -> dict[int, Firm]:
        """Column -> Firm for the names eligible at session ``t``."""
        day = self.prices.days[t]
        current = [firms for d, firms in self._schedule if d <= day]
        if not current:
            return {}
        return {self._col[s]: f for s, f in current[-1].items() if s in self._col}

    def _row_on_or_before(self, day: date) -> int | None:
        i = bisect_right(self.prices.days, day) - 1
        return i if i >= 0 else None

    def caps(self, t: int, firms: Mapping[int, Firm]) -> FloatArray:
        """Market cap per column. A company listed in several share classes
        (GOOG, GOOGL) reports one share count for all of them, so each class
        gets an equal part of the company's cap rather than all of it."""
        close = self.prices.close
        out = np.full(close.shape[1], np.nan)
        classes: dict[int, int] = {}
        for f in firms.values():
            if f.cik is not None:
                classes[f.cik] = classes.get(f.cik, 0) + 1
        for j, f in firms.items():
            then = self._row_on_or_before(f.screened)
            if f.cap is None or then is None:
                continue
            ratio = close[t, j] / close[then, j]
            if np.isfinite(ratio) and ratio > 0:
                out[j] = f.cap * ratio / (classes.get(f.cik, 1) if f.cik is not None else 1)
        return out

    def composite(self, t: int, firms: Mapping[int, Firm]) -> FloatArray:
        """Standardised momentum + low-vol + quality across eligible names (NaN elsewhere)."""
        momentum, vol, complete = factor_parts(self.prices.close, t)
        cols = np.array([j for j in firms if complete[j]], dtype=int)
        out = np.full(self.prices.close.shape[1], np.nan)
        if len(cols) < 2:
            return out
        year = self._quality.get(usable_year(self.prices.days[t]), {})
        ciks = [firms[int(j)].cik for j in cols]
        qual = np.array([np.nan if c is None else year.get(c, np.nan) for c in ciks])
        zq = np.zeros(len(cols))
        known = ~np.isnan(qual)
        if known.sum() >= 2:
            zq[known] = zscore(qual[known])
        out[cols] = zscore(zscore(momentum[cols]) + zscore(-vol[cols]) + zq)
        return out


def _normalise(w: FloatArray) -> FloatArray:
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    total = w.sum()
    return w / total if total > 0 else w


def cap_weighted(inputs: Inputs) -> TargetFn:
    def target(t: int) -> FloatArray:
        return _normalise(inputs.caps(t, inputs.firms(t)))

    return target


def cap_tilt(inputs: Inputs, tilt: float = TILT) -> TargetFn:
    def target(t: int) -> FloatArray:
        firms = inputs.firms(t)
        z = inputs.composite(t, firms)
        # A name without a full price window keeps its cap weight (z = 0).
        return _normalise(inputs.caps(t, firms) * np.exp(tilt * np.nan_to_num(z)))

    return target


def one_class_per_company(score: FloatArray, firms: Mapping[int, Firm]) -> FloatArray:
    """Keep each company's best-scored share class; the others score NaN."""
    out = score.copy()
    best: dict[int, int] = {}
    for j, f in firms.items():
        if f.cik is None or np.isnan(out[j]):
            continue
        k = best.get(f.cik)
        if k is None or out[j] > out[k]:
            if k is not None:
                out[k] = np.nan
            best[f.cik] = j
        else:
            out[j] = np.nan
    return out


def three_factor(inputs: Inputs, top_n: int = TOP_N) -> TargetFn:
    def target(t: int) -> FloatArray:
        firms = inputs.firms(t)
        return equal_top(one_class_per_company(inputs.composite(t, firms), firms), top_n)

    return target


STRATEGIES = {"t3-cap": cap_weighted, "t4-tilt": cap_tilt, "t5-3factor": three_factor}
