"""Execution-quality A/B testing — Round-5 Wave 12.I.

When the operator wants to compare two execution algos (TWAP vs VWAP
vs SMART_ROUTER vs ICEBERG), naive Sharpe-on-trade-PnL isn't enough —
slippage variance is high and trade counts are small. This module
runs a deterministic bootstrap to put a confidence interval around
the slippage delta between two cohorts.

Pipeline:
1. Operator tags each fill with the algo used + arrival mid-price.
2. Slippage = signed (fill - arrival) / arrival × side_sign. Pinned
   *signed-positive-bad* convention so the comparator handles BUY +
   SELL uniformly.
3. `compare_cohorts(a, b)` returns: mean slippage delta, bootstrap
   95% CI, and a significance verdict.
4. `power_estimate(n_a, n_b, target_delta)` reports whether the
   sample sizes are large enough to detect the target delta.

Pinned semantics:

- **Signed-positive-bad slippage**. BUY @ price > arrival_mid → positive
  slippage (bad); SELL @ price < arrival_mid → positive slippage (bad).
- **Bootstrap is seeded** — identical inputs + identical seed produce
  identical outputs. Critical for ops review.
- **CI is percentile** (not BCa) — readable + deterministic.
- **`is_significant`** = CI does not include 0.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    """Closed-set fill side ladder."""

    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class Fill:
    """One fill tagged with its execution algo + arrival mid-price."""

    fill_id: str
    algo_label: str
    """Operator-chosen label (e.g. 'twap', 'smart_router')."""
    side: Side
    arrival_mid: float
    fill_price: float
    quantity: float

    def __post_init__(self) -> None:
        if not self.fill_id or not self.fill_id.strip():
            raise ValueError("fill_id must be non-empty")
        if not self.algo_label or not self.algo_label.strip():
            raise ValueError("algo_label must be non-empty")
        if self.arrival_mid <= 0:
            raise ValueError("arrival_mid must be positive")
        if self.fill_price <= 0:
            raise ValueError("fill_price must be positive")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")


def slippage_bps(fill: Fill) -> float:
    """Signed-positive-bad slippage in basis points (1bp = 0.01%).

    BUY:  (fill - arrival) / arrival × 1e4  (positive = paid up)
    SELL: (arrival - fill) / arrival × 1e4  (positive = sold low)
    """
    if fill.side is Side.BUY:
        raw = (fill.fill_price - fill.arrival_mid) / fill.arrival_mid
    else:
        raw = (fill.arrival_mid - fill.fill_price) / fill.arrival_mid
    return raw * 1e4


@dataclass(frozen=True)
class CohortStats:
    """Output of `summarise_cohort`."""

    label: str
    n_fills: int
    total_quantity: float
    mean_slippage_bps: float
    median_slippage_bps: float
    std_slippage_bps: float


def _sorted_slippage(fills: Sequence[Fill]) -> list[float]:
    return sorted(slippage_bps(f) for f in fills)


def summarise_cohort(label: str, fills: Sequence[Fill]) -> CohortStats:
    """Per-cohort summary stats."""
    if not fills:
        raise ValueError("cohort must be non-empty")
    if any(f.algo_label != label for f in fills):
        raise ValueError("all fills must share the cohort label")
    slips = _sorted_slippage(fills)
    n = len(slips)
    mean = sum(slips) / n
    if n % 2 == 1:
        median = slips[n // 2]
    else:
        median = 0.5 * (slips[n // 2 - 1] + slips[n // 2])
    var = sum((s - mean) ** 2 for s in slips) / max(1, n - 1)
    std = math.sqrt(var)
    total_qty = sum(f.quantity for f in fills)
    return CohortStats(
        label=label,
        n_fills=n,
        total_quantity=total_qty,
        mean_slippage_bps=mean,
        median_slippage_bps=median,
        std_slippage_bps=std,
    )


@dataclass(frozen=True)
class CompareResult:
    """Output of `compare_cohorts`."""

    label_a: str
    label_b: str
    n_a: int
    n_b: int
    mean_slippage_a_bps: float
    mean_slippage_b_bps: float
    delta_a_minus_b_bps: float
    """Positive delta means cohort A has WORSE slippage than B
    (signed-positive-bad)."""
    ci_low_bps: float
    ci_high_bps: float
    is_significant: bool
    """True iff the CI does not contain 0."""


def compare_cohorts(
    fills_a: Sequence[Fill],
    fills_b: Sequence[Fill],
    *,
    label_a: str | None = None,
    label_b: str | None = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> CompareResult:
    """Compare two execution-algo cohorts with a bootstrap CI on the
    slippage delta.

    Pinned: cohorts may overlap in fill IDs across batches but the
    A and B sequences must be disjoint per call.
    """
    if not fills_a or not fills_b:
        raise ValueError("both cohorts must be non-empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be ≥ 100")
    actual_a = label_a or fills_a[0].algo_label
    actual_b = label_b or fills_b[0].algo_label
    slip_a = [slippage_bps(f) for f in fills_a]
    slip_b = [slippage_bps(f) for f in fills_b]
    mean_a = sum(slip_a) / len(slip_a)
    mean_b = sum(slip_b) / len(slip_b)
    delta = mean_a - mean_b
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        sample_a = [rng.choice(slip_a) for _ in range(len(slip_a))]
        sample_b = [rng.choice(slip_b) for _ in range(len(slip_b))]
        deltas.append(sum(sample_a) / len(sample_a) - sum(sample_b) / len(sample_b))
    deltas.sort()
    alpha = 1.0 - confidence
    lo_idx = int(round((alpha / 2) * (len(deltas) - 1)))
    hi_idx = int(round((1 - alpha / 2) * (len(deltas) - 1)))
    ci_low = deltas[lo_idx]
    ci_high = deltas[hi_idx]
    is_sig = (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)
    return CompareResult(
        label_a=actual_a,
        label_b=actual_b,
        n_a=len(fills_a),
        n_b=len(fills_b),
        mean_slippage_a_bps=mean_a,
        mean_slippage_b_bps=mean_b,
        delta_a_minus_b_bps=delta,
        ci_low_bps=ci_low,
        ci_high_bps=ci_high,
        is_significant=is_sig,
    )


@dataclass(frozen=True)
class PowerEstimate:
    """Output of `power_estimate`."""

    n_a: int
    n_b: int
    target_delta_bps: float
    estimated_se_bps: float
    estimated_z: float
    """Number of SEs the target delta is from zero. > 1.96 ≈ 95% power
    rule-of-thumb."""
    is_well_powered: bool


def power_estimate(
    cohort_a_std_bps: float,
    cohort_b_std_bps: float,
    *,
    n_a: int,
    n_b: int,
    target_delta_bps: float,
) -> PowerEstimate:
    """Quick power check: given expected per-cohort stdevs and sample
    sizes, can the comparator reliably detect a `target_delta_bps`?

    Uses the standard pooled-variance SE for the difference of means.
    Operators should treat |z| > 1.96 as well-powered.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("n_a and n_b must be positive")
    if cohort_a_std_bps < 0 or cohort_b_std_bps < 0:
        raise ValueError("std must be non-negative")
    if target_delta_bps == 0:
        raise ValueError("target_delta_bps must be non-zero")
    se = math.sqrt((cohort_a_std_bps**2) / n_a + (cohort_b_std_bps**2) / n_b)
    z = abs(target_delta_bps) / max(se, 1e-12)
    return PowerEstimate(
        n_a=n_a,
        n_b=n_b,
        target_delta_bps=target_delta_bps,
        estimated_se_bps=se,
        estimated_z=z,
        is_well_powered=z >= 1.96,
    )


def render_compare(result: CompareResult) -> str:
    sig = "✅ significant" if result.is_significant else "⚠️ inconclusive"
    return (
        f"🆚 {result.label_a} vs {result.label_b}: "
        f"Δ slippage = {result.delta_a_minus_b_bps:+.2f}bps "
        f"({result.ci_low_bps:+.2f}, {result.ci_high_bps:+.2f}) | "
        f"n_a={result.n_a} n_b={result.n_b} | {sig}"
    )


def render_power(p: PowerEstimate) -> str:
    flag = "✅ powered" if p.is_well_powered else "❌ underpowered"
    return (
        f"⚡ Power n_a={p.n_a} n_b={p.n_b}: "
        f"target Δ={p.target_delta_bps:+.2f}bps, "
        f"SE={p.estimated_se_bps:.2f}bps, z={p.estimated_z:.2f} ({flag})"
    )
