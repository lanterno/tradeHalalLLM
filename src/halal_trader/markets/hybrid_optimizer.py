"""Hybrid equity-sukuk portfolio optimizer — Round-5 Wave 3.G.

Most halal portfolios mix equity (compliant stocks) and sukuk (Islamic
fixed income). Conventional pension-fund optimisers are bond-heavy
near retirement, equity-heavy in accumulation; the halal analogue
swaps bonds for sukuk and applies AAOIFI-compliant sector caps to the
equity sleeve.

This module is the unified mean-variance optimizer respecting:
- per-asset-class weight bands (e.g. 30-70% equity / 30-70% sukuk),
- equity sector caps (financials, defense, alcohol — see Std 21),
- sukuk-side caps from `markets/sukuk_allocation.py`,
- portfolio-level duration target (sukuk-side only),
- portfolio-level dividend-yield target (income strategies),
- single-name concentration cap.

The implementation reuses the projected-gradient solver from
`sukuk_allocation` — the universe is concatenated [equity..., sukuk...]
and the simplex projection guarantees the long-only sum-to-one.

Pinned semantics:

- **Asset class is a hard label.** An entry with `asset_class=EQUITY`
  is never optimised against a sukuk-side constraint and vice versa.
- **Class bands are absolute.** If band=(0.30, 0.70), the optimiser
  scales the within-class sub-weights so the class sum lands inside
  the band — same water-filling pattern as the sukuk side.
- **Equity sector caps default to AAOIFI Standard 21 caps** (e.g.
  financials 33% as a Standard 21 prudential limit). Operators can
  override.
- **Sukuk duration target** is enforced only over the sukuk sleeve;
  equity entries don't contribute (they carry no duration).
- **Pure-Python deterministic.** No NumPy / SciPy.
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from halal_trader.halal.aaoifi_standard_17 import (
    SukukType,
    is_tradable_in_secondary,
)
from halal_trader.markets.sukuk_allocation import (
    InfeasibleBasketError,
    _project_to_simplex,
)


class AssetClass(str, Enum):
    """Closed-set asset class for the hybrid universe."""

    EQUITY = "equity"
    SUKUK = "sukuk"


@dataclass(frozen=True)
class HybridAsset:
    """A single asset (equity or sukuk) in the hybrid universe.

    For EQUITY: `sukuk_type` and `duration_years` are ignored.
    For SUKUK: `sector` doubles as the sukuk's economic sector.
    """

    symbol: str
    asset_class: AssetClass
    sector: str
    jurisdiction: str
    expected_return: float
    duration_years: float = 0.0
    dividend_yield: float = 0.0
    sukuk_type: SukukType | None = None

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if not self.sector or not self.sector.strip():
            raise ValueError("sector must be non-empty")
        if not self.jurisdiction or not self.jurisdiction.strip():
            raise ValueError("jurisdiction must be non-empty")
        if not -0.20 < self.expected_return < 0.50:
            raise ValueError("expected_return outside reasonable bounds")
        if not 0.0 <= self.dividend_yield < 0.30:
            raise ValueError("dividend_yield outside reasonable bounds")
        if self.asset_class is AssetClass.SUKUK:
            if self.sukuk_type is None:
                raise ValueError("sukuk entry requires sukuk_type")
            if not is_tradable_in_secondary(self.sukuk_type):
                raise ValueError(f"{self.sukuk_type.value} not tradable on secondary")
            if self.duration_years <= 0:
                raise ValueError("sukuk entry requires duration_years > 0")
        else:
            # EQUITY — duration must be 0 (no duration risk on stock).
            if self.duration_years != 0.0:
                raise ValueError("equity entry must have duration_years=0")


# AAOIFI Standard 21 derived prudential caps for equity sectors. The
# numbers are conservative — operators tune via constraints.
_DEFAULT_EQUITY_SECTOR_CAPS: dict[str, float] = {
    "financials": 0.33,
    "consumer_discretionary": 0.40,
    "technology": 0.40,
    "healthcare": 0.40,
    "energy": 0.30,
    "materials": 0.30,
    "industrials": 0.40,
    "utilities": 0.25,
    "communications": 0.30,
    "consumer_staples": 0.40,
    "real_estate": 0.20,
}


@dataclass(frozen=True)
class HybridConstraints:
    """Constraints layered on top of the hybrid Markowitz optimiser."""

    # (min, max) class weight bands. Each in [0, 1]; min ≤ max.
    equity_band: tuple[float, float] = (0.30, 0.70)
    sukuk_band: tuple[float, float] = (0.30, 0.70)
    # Equity sector caps (sector → max weight in [0, 1]). Defaults to
    # the AAOIFI Standard 21 prudential table.
    equity_sector_caps: dict[str, float] = field(default_factory=dict)
    # Sukuk-side caps.
    sukuk_sector_caps: dict[str, float] = field(default_factory=dict)
    sukuk_jurisdiction_caps: dict[str, float] = field(default_factory=dict)
    sukuk_type_caps: dict[SukukType, float] = field(default_factory=dict)
    # Per-asset cap (single-name concentration). Default is permissive
    # (0.50) so a 4-name universe is feasible by default; operators
    # tighten this in production with much larger universes.
    max_single_name: float = 0.50
    # Optional sukuk-sleeve duration target.
    sukuk_duration_target: float | None = None
    # Optional minimum dividend yield (e.g. for income strategies).
    min_dividend_yield: float = 0.0
    # Mean-variance risk-aversion λ.
    risk_aversion: float = 1.0

    def __post_init__(self) -> None:
        for band, name in (
            (self.equity_band, "equity_band"),
            (self.sukuk_band, "sukuk_band"),
        ):
            lo, hi = band
            if not 0.0 <= lo <= hi <= 1.0:
                raise ValueError(f"{name} must satisfy 0 ≤ min ≤ max ≤ 1")
        # Bands must overlap (sum of mins ≤ 1 ≤ sum of maxes).
        if self.equity_band[0] + self.sukuk_band[0] > 1.0 + 1e-9:
            raise ValueError("equity_band.min + sukuk_band.min must be ≤ 1")
        if self.equity_band[1] + self.sukuk_band[1] < 1.0 - 1e-9:
            raise ValueError("equity_band.max + sukuk_band.max must be ≥ 1")
        if not 0.0 < self.max_single_name <= 1.0:
            raise ValueError("max_single_name must be in (0, 1]")
        if self.risk_aversion < 0:
            raise ValueError("risk_aversion must be non-negative")
        if not 0.0 <= self.min_dividend_yield < 0.30:
            raise ValueError("min_dividend_yield outside reasonable bounds")
        if self.sukuk_duration_target is not None and self.sukuk_duration_target <= 0:
            raise ValueError("sukuk_duration_target must be positive")


@dataclass(frozen=True)
class HybridResult:
    """Output of `optimize_hybrid`."""

    weights: tuple[float, ...]
    assets: tuple[HybridAsset, ...]
    expected_return: float
    expected_variance: float
    equity_weight: float
    sukuk_weight: float
    sukuk_duration: float
    portfolio_dividend_yield: float

    def expected_volatility(self) -> float:
        return math.sqrt(max(0.0, self.expected_variance))


def _equity_sector_cap(constraints: HybridConstraints, sector: str) -> float:
    """Look up the equity sector cap with AAOIFI defaults."""
    if sector in constraints.equity_sector_caps:
        return constraints.equity_sector_caps[sector]
    return _DEFAULT_EQUITY_SECTOR_CAPS.get(sector, 0.40)


def _sukuk_sector_cap(constraints: HybridConstraints, sector: str) -> float:
    return constraints.sukuk_sector_caps.get(sector, 1.0)


def _sukuk_jur_cap(constraints: HybridConstraints, jurisdiction: str) -> float:
    return constraints.sukuk_jurisdiction_caps.get(jurisdiction, 1.0)


def _sukuk_type_cap(constraints: HybridConstraints, st: SukukType) -> float:
    return constraints.sukuk_type_caps.get(st, 1.0)


def _apply_class_band(
    out: list[float],
    eq_idx: list[int],
    sk_idx: list[int],
    band_eq: tuple[float, float],
    band_sk: tuple[float, float],
    *,
    single_name_cap: float = 1.0,
) -> list[float]:
    """Push per-class sums to the (eq, sk) target that satisfies both bands and sums to 1.

    The feasible interval for new_eq_sum is
    [max(eq_min, 1 - sk_max), min(eq_max, 1 - sk_min)].
    Pick the value in that interval closest to current eq_sum, then
    rebalance the within-class weights proportionally.
    """
    if not eq_idx and not sk_idx:
        return out
    if not eq_idx:
        # Pure-sukuk universe — push sukuk to 1, ignore eq band.
        sk_sum = sum(out[i] for i in sk_idx)
        if sk_sum > 1e-12:
            scale = 1.0 / sk_sum
            for i in sk_idx:
                out[i] *= scale
        else:
            for i in sk_idx:
                out[i] = 1.0 / len(sk_idx)
        return out
    if not sk_idx:
        eq_sum = sum(out[i] for i in eq_idx)
        if eq_sum > 1e-12:
            scale = 1.0 / eq_sum
            for i in eq_idx:
                out[i] *= scale
        else:
            for i in eq_idx:
                out[i] = 1.0 / len(eq_idx)
        return out

    eq_sum = sum(out[i] for i in eq_idx)
    # Single-name cap also bounds the within-class maximum.
    eq_max_by_single = single_name_cap * len(eq_idx)
    sk_max_by_single = single_name_cap * len(sk_idx)
    eq_band_hi = min(band_eq[1], eq_max_by_single)
    sk_band_hi = min(band_sk[1], sk_max_by_single)
    eq_lo = max(band_eq[0], 1.0 - sk_band_hi)
    eq_hi = min(eq_band_hi, 1.0 - band_sk[0])
    if eq_lo > eq_hi + 1e-9:
        raise InfeasibleBasketError("class bands + single-name cap do not admit a feasible point")
    new_eq_sum = min(max(eq_sum, eq_lo), eq_hi)
    new_sk_sum = 1.0 - new_eq_sum
    if eq_sum > 1e-12:
        scale_eq = new_eq_sum / eq_sum
        for i in eq_idx:
            out[i] *= scale_eq
    else:
        for i in eq_idx:
            out[i] = new_eq_sum / len(eq_idx)
    sk_sum = sum(out[i] for i in sk_idx)
    if sk_sum > 1e-12:
        scale_sk = new_sk_sum / sk_sum
        for i in sk_idx:
            out[i] *= scale_sk
    else:
        for i in sk_idx:
            out[i] = new_sk_sum / len(sk_idx)
    return out


def _apply_hybrid_caps(
    w: list[float],
    assets: Sequence[HybridAsset],
    constraints: HybridConstraints,
) -> list[float]:
    """Single-name + class band + sector + jur + type caps via water-filling."""
    out = list(w)
    cap_single = constraints.max_single_name
    eq_idx = [i for i, a in enumerate(assets) if a.asset_class is AssetClass.EQUITY]
    sk_idx = [i for i, a in enumerate(assets) if a.asset_class is AssetClass.SUKUK]
    by_eq_sector: dict[str, list[int]] = {}
    by_sk_sector: dict[str, list[int]] = {}
    by_sk_jur: dict[str, list[int]] = {}
    by_sk_type: dict[SukukType, list[int]] = {}
    for i in eq_idx:
        by_eq_sector.setdefault(assets[i].sector, []).append(i)
    for i in sk_idx:
        by_sk_sector.setdefault(assets[i].sector, []).append(i)
        by_sk_jur.setdefault(assets[i].jurisdiction, []).append(i)
        st = assets[i].sukuk_type
        if st is not None:
            by_sk_type.setdefault(st, []).append(i)

    for _ in range(50):
        # Class band first — push class sums into the feasible interval.
        out = _apply_class_band(
            out,
            eq_idx,
            sk_idx,
            constraints.equity_band,
            constraints.sukuk_band,
            single_name_cap=cap_single,
        )
        # Single-name clamp.
        out = [min(wi, cap_single) for wi in out]
        # Equity sector cap.
        for sector, idxs in by_eq_sector.items():
            cap = _equity_sector_cap(constraints, sector)
            s = sum(out[i] for i in idxs)
            if s > cap and s > 1e-12:
                scale = cap / s
                for i in idxs:
                    out[i] *= scale
        # Sukuk sector / jur / type caps.
        for sector, idxs in by_sk_sector.items():
            cap = _sukuk_sector_cap(constraints, sector)
            s = sum(out[i] for i in idxs)
            if s > cap and s > 1e-12:
                scale = cap / s
                for i in idxs:
                    out[i] *= scale
        for jur, idxs in by_sk_jur.items():
            cap = _sukuk_jur_cap(constraints, jur)
            s = sum(out[i] for i in idxs)
            if s > cap and s > 1e-12:
                scale = cap / s
                for i in idxs:
                    out[i] *= scale
        for st, idxs in by_sk_type.items():
            cap = _sukuk_type_cap(constraints, st)
            s = sum(out[i] for i in idxs)
            if s > cap and s > 1e-12:
                scale = cap / s
                for i in idxs:
                    out[i] *= scale
        s = sum(out)
        if s <= 1e-12:
            raise InfeasibleBasketError("constraints reduce all weights to zero")
        out = [wi / s for wi in out]
        # Convergence check.
        if max(out) > cap_single + 1e-9:
            continue
        eq_sum = sum(out[i] for i in eq_idx)
        sk_sum = sum(out[i] for i in sk_idx)
        if eq_sum > constraints.equity_band[1] + 1e-9 or eq_sum < constraints.equity_band[0] - 1e-9:
            continue
        if sk_sum > constraints.sukuk_band[1] + 1e-9 or sk_sum < constraints.sukuk_band[0] - 1e-9:
            continue
        violated = False
        for sector, idxs in by_eq_sector.items():
            if sum(out[i] for i in idxs) > _equity_sector_cap(constraints, sector) + 1e-9:
                violated = True
                break
        if not violated:
            break
    return out


def _portfolio_metrics(
    w: Sequence[float],
    assets: Sequence[HybridAsset],
    cov: Sequence[Sequence[float]],
) -> tuple[float, float, float, float, float, float]:
    """Compute (return, var, eq_w, sk_w, sk_duration, dividend_yield)."""
    n = len(w)
    pr = sum(wi * a.expected_return for wi, a in zip(w, assets, strict=True))
    pv = 0.0
    for i in range(n):
        for j in range(n):
            pv += w[i] * cov[i][j] * w[j]
    eq_w = sum(wi for wi, a in zip(w, assets, strict=True) if a.asset_class is AssetClass.EQUITY)
    sk_w = sum(wi for wi, a in zip(w, assets, strict=True) if a.asset_class is AssetClass.SUKUK)
    sk_dur = sum(
        wi * a.duration_years
        for wi, a in zip(w, assets, strict=True)
        if a.asset_class is AssetClass.SUKUK
    )
    div = sum(wi * a.dividend_yield for wi, a in zip(w, assets, strict=True))
    return pr, pv, eq_w, sk_w, sk_dur, div


def optimize_hybrid(
    assets: Sequence[HybridAsset],
    *,
    covariance: Sequence[Sequence[float]] | None = None,
    constraints: HybridConstraints | None = None,
    max_iter: int = 500,
    step_size: float = 0.05,
    tolerance: float = 1e-8,
) -> HybridResult:
    """Run the hybrid projected-gradient solver."""
    if not assets:
        raise ValueError("assets must be non-empty")
    if len(assets) > 200:
        raise ValueError("universe too large; pre-filter to ≤200")
    cstr = constraints if constraints is not None else HybridConstraints()
    n = len(assets)
    if covariance is None:
        cov: list[list[float]] = [[0.04 if i == j else 0.0 for j in range(n)] for i in range(n)]
    else:
        cov = [list(row) for row in covariance]
        if len(cov) != n or any(len(row) != n for row in cov):
            raise ValueError(f"covariance must be {n}×{n}")
        for i in range(n):
            for j in range(i + 1, n):
                if abs(cov[i][j] - cov[j][i]) > 1e-9:
                    raise ValueError("covariance must be symmetric")
            if cov[i][i] < 0:
                raise ValueError("covariance diagonal must be non-negative")

    mu = [a.expected_return for a in assets]

    # Initial weights: uniform, then push into class bands so the
    # iteration starts feasible.
    w = [1.0 / n] * n
    eq_idx = [i for i, a in enumerate(assets) if a.asset_class is AssetClass.EQUITY]
    sk_idx = [i for i, a in enumerate(assets) if a.asset_class is AssetClass.SUKUK]
    if eq_idx and sk_idx:
        # Distribute the equity_band midpoint across equity, etc.
        eq_target = (cstr.equity_band[0] + cstr.equity_band[1]) / 2
        sk_target = 1.0 - eq_target
        for i in eq_idx:
            w[i] = eq_target / len(eq_idx)
        for i in sk_idx:
            w[i] = sk_target / len(sk_idx)
    elif eq_idx:
        for i in eq_idx:
            w[i] = 1.0 / len(eq_idx)
    else:
        for i in sk_idx:
            w[i] = 1.0 / len(sk_idx)

    for _ in range(max_iter):
        # Mean-variance gradient.
        grad = [-mu[i] for i in range(n)]
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += cov[i][j] * w[j]
            grad[i] += 2 * cstr.risk_aversion * s

        # Sukuk duration target gradient.
        if cstr.sukuk_duration_target is not None and sk_idx:
            beta = max(1.0, cstr.risk_aversion)
            d_w = sum(w[i] * assets[i].duration_years for i in sk_idx)
            mismatch = d_w - cstr.sukuk_duration_target
            for i in sk_idx:
                grad[i] += 2 * beta * mismatch * assets[i].duration_years

        # Min-dividend-yield gradient (one-sided): if portfolio yield <
        # target, push grad to favour high-yield names.
        if cstr.min_dividend_yield > 0:
            beta = max(1.0, cstr.risk_aversion)
            yld = sum(w[i] * assets[i].dividend_yield for i in range(n))
            shortfall = cstr.min_dividend_yield - yld
            if shortfall > 0:
                for i in range(n):
                    grad[i] -= 2 * beta * shortfall * assets[i].dividend_yield

        # Adaptive step — same trick as sukuk_allocation.
        max_grad = max(abs(g) for g in grad) if grad else 0.0
        effective_step = step_size
        if max_grad * step_size > 0.1:
            effective_step = 0.1 / max_grad
        new_w = [w[i] - effective_step * grad[i] for i in range(n)]
        new_w = _project_to_simplex(new_w)
        new_w = _apply_hybrid_caps(new_w, assets, cstr)
        delta = sum(abs(new_w[i] - w[i]) for i in range(n))
        w = new_w
        if delta < tolerance:
            break

    pr, pv, eq_w, sk_w, sk_dur, div = _portfolio_metrics(w, assets, cov)
    return HybridResult(
        weights=tuple(w),
        assets=tuple(assets),
        expected_return=pr,
        expected_variance=pv,
        equity_weight=eq_w,
        sukuk_weight=sk_w,
        sukuk_duration=sk_dur,
        portfolio_dividend_yield=div,
    )


def render_hybrid(result: HybridResult, *, top_n: int = 10) -> str:
    """Operator-readable summary of the hybrid portfolio."""
    head = (
        f"⚖️ Hybrid portfolio: {len(result.assets)} assets — "
        f"equity {result.equity_weight * 100:.2f}% / "
        f"sukuk {result.sukuk_weight * 100:.2f}%\n"
        f"  return {result.expected_return * 100:.2f}%, "
        f"vol {result.expected_volatility() * 100:.2f}%, "
        f"sukuk-duration {result.sukuk_duration:.2f}y, "
        f"yield {result.portfolio_dividend_yield * 100:.2f}%"
    )
    pairs = sorted(
        zip(result.assets, result.weights, strict=True),
        key=lambda kv: kv[1],
        reverse=True,
    )[:top_n]
    lines = [head]
    for asset, w in pairs:
        if w <= 1e-9:
            continue
        cls = "🟢" if asset.asset_class is AssetClass.EQUITY else "🟦"
        lines.append(f"  • {cls} {asset.symbol} ({asset.sector}): {w * 100:.2f}%")
    return "\n".join(lines)
