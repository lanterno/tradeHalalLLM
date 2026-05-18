"""Counterfactual trade analyzer — "what would equity have been if
I'd skipped trades matching pattern X?".

Round-4 wave 7.C: research / post-hoc tool. Operators routinely want
to ask things like:

* "What if I hadn't traded BTC during last week's regime shift?"
* "What if I'd stopped trading after my third loss in a row?"
* "What's the contribution of trades placed in the first 30 minutes
  after the open vs the rest of the day?"

The analyzer answers all of these as a *single* function that takes
the closed-trade history (any list of dict-shaped rows with at least
``return_pct`` and the columns the predicate filters on) plus a
predicate that picks the hypothetically-skipped subset, then returns
both the actual and counterfactual equity curves plus a headline
comparison block.

Picked a predicate-based API rather than a DSL because operators
already write SQL for ad-hoc queries — Python lambdas / functions
match how they think about filters, and we don't have to maintain
a parser. The trade rows can be plain dicts (`row.get("regime") ==
"downtrend"`) or pydantic / SQLModel objects (the predicate uses
attribute access — ``getattr`` works on both).

Halal alignment: the counterfactual is purely informational — it
never causes a trade. The output is for the human operator's
research session, not the automated cycle.

Pure-NumPy core; reuses :func:`core.ab_compare.cohort_stats` so the
"actual vs counterfactual" comparison uses the same Sharpe / drawdown
/ profit-factor formulas as the A/B comparator. Keeps the dashboard
numbers consistent across views.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from halal_trader.core.ab_compare import CohortStats, cohort_stats


@dataclass(frozen=True)
class CounterfactualReport:
    """Side-by-side comparison of actual vs "skipped X" trade history.

    ``actual`` and ``counterfactual`` are the per-cohort headline
    stats (n_trades, win_rate, mean / median / std return, Sharpe,
    max drawdown, total compound return, profit factor).

    ``skipped_count`` is how many trades the predicate matched —
    useful for the operator to gut-check ("am I dropping the right
    rows?").

    ``actual_curve`` and ``counterfactual_curve`` are the equity
    curves (starting equity = 1.0, multiplied by ∏(1+r) over the
    relevant trades). Same length: the counterfactual curve simply
    holds flat across the skipped trades, so the time axis is
    preserved and the curves are directly comparable on a dashboard
    plot.

    ``return_uplift`` is ``counterfactual.total_return -
    actual.total_return`` — positive means skipping the matched
    trades would have *helped*; negative means they were
    contributing positively to the bottom line.
    """

    actual: CohortStats
    counterfactual: CohortStats
    skipped_count: int
    actual_curve: list[float]
    counterfactual_curve: list[float]
    return_uplift: float


def _row_return(row: Any) -> float | None:
    """Extract ``return_pct`` from a dict-shaped row OR an
    attribute-shaped row (pydantic / SQLModel).

    Returns ``None`` for rows where the field is missing or
    non-numeric — the caller decides whether to skip or raise.
    """
    if isinstance(row, dict):
        val = row.get("return_pct")
    else:
        val = getattr(row, "return_pct", None)
    if val is None:
        return None
    try:
        f = float(val)
    except TypeError, ValueError:
        return None
    if not np.isfinite(f):
        return None
    return f


def _build_curve(returns: Iterable[float], starting: float = 1.0) -> list[float]:
    """Cumulative equity curve. Returns a Python list (not numpy)
    because the result is dashboard-bound and JSON needs Python
    floats."""
    curve: list[float] = []
    equity = starting
    for r in returns:
        equity *= 1.0 + r
        curve.append(float(equity))
    return curve


def analyze_counterfactual(
    trades: Iterable[Any],
    skip_predicate: Callable[[Any], bool],
    *,
    starting_equity: float = 1.0,
) -> CounterfactualReport:
    """Compute the counterfactual: "what if every trade for which
    ``skip_predicate`` returns True had been skipped?".

    ``trades`` is any iterable of trade rows in *chronological order*
    (oldest first). Each row must expose ``return_pct`` either as a
    dict key or an attribute. Rows with missing / non-finite
    ``return_pct`` are silently dropped (legacy crypto trades that
    pre-date the field, manual exits, etc.).

    ``skip_predicate`` is called once per row and must return a
    boolean. Any exception in the predicate is treated as "do not
    skip" — we'd rather show a slightly-wrong counterfactual than
    crash the operator's research session.

    Pin: the counterfactual curve flatlines (holds equity constant)
    over the skipped trades. The alternative — fully removing them
    from the time axis — produces a curve that's not directly
    plottable against the actual curve. Flat-line is the convention
    the dashboard plot uses.
    """
    if starting_equity <= 0:
        raise ValueError(f"starting_equity must be positive; got {starting_equity}")

    actual_returns: list[float] = []
    cf_returns: list[float] = []  # counterfactual: 0.0 for skipped rows
    skipped_returns: list[float] = []
    skipped = 0

    for row in trades:
        r = _row_return(row)
        if r is None:
            continue
        try:
            should_skip = bool(skip_predicate(row))
        except Exception:
            should_skip = False
        actual_returns.append(r)
        if should_skip:
            cf_returns.append(0.0)  # equity holds flat
            skipped += 1
            skipped_returns.append(r)
        else:
            cf_returns.append(r)

    actual_stats = cohort_stats(actual_returns)
    # Counterfactual stats are computed on *kept* trades only —
    # the n_trades / win_rate readouts wouldn't be meaningful if
    # they counted the held-flat zeros.
    kept_returns = [r for r, c in zip(actual_returns, cf_returns) if c != 0.0 or r == 0.0]
    # Edge: if every kept trade really was 0.0 we keep them; the
    # comprehension above retains a 0.0-return kept trade because
    # ``r == 0.0`` is True. Explicit pin so a refactor doesn't
    # accidentally drop legitimate zero-return trades.
    cf_stats = cohort_stats(kept_returns)

    actual_curve = _build_curve(actual_returns, starting=starting_equity)
    cf_curve = _build_curve(cf_returns, starting=starting_equity)

    return CounterfactualReport(
        actual=actual_stats,
        counterfactual=cf_stats,
        skipped_count=skipped,
        actual_curve=actual_curve,
        counterfactual_curve=cf_curve,
        return_uplift=cf_stats.total_return - actual_stats.total_return,
    )


# ── Convenience predicate factories ───────────────────────


def by_symbol(symbol: str) -> Callable[[Any], bool]:
    """Predicate: skip trades on a given symbol.

    Tolerant to dict / attribute shapes and to the ``symbol`` vs
    ``pair`` field naming difference between Trade (stocks) and
    CryptoTrade (crypto)."""

    target = symbol.upper()

    def _pred(row: Any) -> bool:
        if isinstance(row, dict):
            sym = row.get("symbol") or row.get("pair") or ""
        else:
            sym = getattr(row, "symbol", None) or getattr(row, "pair", "") or ""
        return str(sym).upper() == target

    return _pred


def by_regime(regime: str) -> Callable[[Any], bool]:
    """Predicate: skip trades labelled with a given regime tag."""

    target = regime.lower()

    def _pred(row: Any) -> bool:
        if isinstance(row, dict):
            label = row.get("regime") or ""
        else:
            label = getattr(row, "regime", "") or ""
        return str(label).lower() == target

    return _pred


def by_loss_streak(min_streak: int) -> Callable[[Any], bool]:
    """Predicate: skip trades that follow ``min_streak`` consecutive
    losses. Implemented as a closure over per-call streak state.

    Useful for the "what if I'd stopped after three losses in a row"
    question. Note this predicate has memory — it only makes sense
    when the trades are passed in chronological order.
    """
    state = {"streak": 0}

    def _pred(row: Any) -> bool:
        r = _row_return(row)
        skip = state["streak"] >= min_streak
        if r is None:
            return skip
        if r < 0:
            state["streak"] += 1
        else:
            state["streak"] = 0
        return skip

    return _pred


__all__ = [
    "CounterfactualReport",
    "analyze_counterfactual",
    "by_loss_streak",
    "by_regime",
    "by_symbol",
]
