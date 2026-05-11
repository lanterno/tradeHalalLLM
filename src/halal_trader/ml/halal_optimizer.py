"""Halal-native portfolio optimizer — Round-5 Wave 7.E.

Conventional Markowitz mean-variance optimisation maximises return for
a given variance budget across a *neutral* universe. The halal-native
version adds three structural pins:

1. **Mudarabah profit-share constraints** — the operator may pre-commit
   a slice of capital to a Mudarabah pool with profit-share semantics
   (see `halal/pls_strategy.py`); the optimiser must respect that
   capital is locked in the pool with a minimum weight.
2. **Sukuk integration** — sukuk are treated as a separate asset class
   with their own duration-risk profile; sector caps don't apply to
   sukuk the same way they apply to equities.
3. **AAOIFI Standard 21 sector caps** — financials capped at 33%,
   utilities 25%, real-estate 20%, etc. (the same defaults as the
   hybrid optimizer).

This module is a **focused mean-variance optimizer** with these halal
pins layered in. Where the hybrid optimizer (Wave 3.G) covers
arbitrary equity-sukuk universes, this one handles the case where the
operator pre-commits a Mudarabah slice and wants the optimiser to fill
the remainder.

Pinned semantics:

- **Closed-set HalalAssetClass** — EQUITY / SUKUK / MUDARABAH_POOL.
- **Mudarabah pool weight is a HARD floor.** If `mudarabah_pool_weight`
  is 0.20, the pool gets exactly 20% (water-fill the rest into
  equity + sukuk).
- **AAOIFI sector caps default-on for equity** (same table as 3.G).
- **Long-only sum-to-one** simplex projection.
- **Pure-Python deterministic.** Reuses the projected-gradient
  primitives from `markets/sukuk_allocation.py`.
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from halal_trader.markets.sukuk_allocation import (
    InfeasibleBasketError,
    _project_to_simplex,
)


class HalalAssetClass(str, Enum):
    """Closed-set halal asset class."""

    EQUITY = "equity"
    SUKUK = "sukuk"
    MUDARABAH_POOL = "mudarabah_pool"


@dataclass(frozen=True)
class HalalAsset:
    """A single asset in the halal-native universe."""

    symbol: str
    asset_class: HalalAssetClass
    sector: str
    expected_return: float
    duration_years: float = 0.0

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if not self.sector or not self.sector.strip():
            raise ValueError("sector must be non-empty")
        if not -0.20 < self.expected_return < 0.50:
            raise ValueError("expected_return outside reasonable bounds")
        if self.asset_class is HalalAssetClass.SUKUK:
            if self.duration_years <= 0:
                raise ValueError("sukuk requires duration_years > 0")
        else:
            if self.duration_years != 0.0:
                raise ValueError(f"{self.asset_class.value} must have duration_years=0")


_DEFAULT_SECTOR_CAPS: dict[str, float] = {
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
class HalalPolicy:
    """Operator-tunable halal-optimizer policy."""

    mudarabah_pool_weight: float = 0.0
    """Fraction of capital pre-committed to the Mudarabah pool. Floor."""
    sector_caps: dict[str, float] = field(default_factory=dict)
    """Equity sector caps (override defaults). Sukuk are not affected."""
    max_single_name: float = 0.30
    risk_aversion: float = 1.0
    sukuk_min_weight: float = 0.0
    """Optional floor on the sukuk sleeve (e.g. 0.20 for a fixed-income tilt)."""
    sukuk_max_weight: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.mudarabah_pool_weight < 1.0:
            raise ValueError("mudarabah_pool_weight must be in [0, 1)")
        if not 0.0 < self.max_single_name <= 1.0:
            raise ValueError("max_single_name must be in (0, 1]")
        if self.risk_aversion < 0:
            raise ValueError("risk_aversion must be non-negative")
        if not 0.0 <= self.sukuk_min_weight <= self.sukuk_max_weight <= 1.0:
            raise ValueError("sukuk_min_weight ≤ sukuk_max_weight, both in [0, 1]")
        for sector, cap in self.sector_caps.items():
            if not 0.0 < cap <= 1.0:
                raise ValueError(f"sector cap for {sector} must be in (0, 1]")
        if (
            self.mudarabah_pool_weight + self.sukuk_max_weight
        ) > 1.0 + 1e-9 and self.sukuk_min_weight > 0:
            # Can still be feasible with sukuk_max < 1 but pool + sukuk_min ≤ 1.
            if self.mudarabah_pool_weight + self.sukuk_min_weight > 1.0 + 1e-9:
                raise ValueError("mudarabah_pool_weight + sukuk_min_weight > 1")


@dataclass(frozen=True)
class HalalOptResult:
    """Output of `optimize`."""

    weights: tuple[float, ...]
    assets: tuple[HalalAsset, ...]
    expected_return: float
    expected_variance: float
    mudarabah_weight: float
    equity_weight: float
    sukuk_weight: float

    def expected_volatility(self) -> float:
        return math.sqrt(max(0.0, self.expected_variance))


def _sector_cap(policy: HalalPolicy, sector: str) -> float:
    if sector in policy.sector_caps:
        return policy.sector_caps[sector]
    return _DEFAULT_SECTOR_CAPS.get(sector, 0.40)


def _apply_caps(
    w: list[float],
    assets: Sequence[HalalAsset],
    policy: HalalPolicy,
) -> list[float]:
    """Pin Mudarabah pool weight, then enforce single-name + sector +
    sukuk band caps via water-filling."""
    out = list(w)
    cap_single = policy.max_single_name
    pool_idx = [i for i, a in enumerate(assets) if a.asset_class is HalalAssetClass.MUDARABAH_POOL]
    eq_idx = [i for i, a in enumerate(assets) if a.asset_class is HalalAssetClass.EQUITY]
    sk_idx = [i for i, a in enumerate(assets) if a.asset_class is HalalAssetClass.SUKUK]
    by_eq_sector: dict[str, list[int]] = {}
    for i in eq_idx:
        by_eq_sector.setdefault(assets[i].sector, []).append(i)

    pool_weight = policy.mudarabah_pool_weight
    remaining = 1.0 - pool_weight

    for _ in range(50):
        # 1. Pin Mudarabah pool weight.
        if pool_idx:
            current_pool = sum(out[i] for i in pool_idx)
            if current_pool > 1e-12:
                scale = pool_weight / current_pool
                for i in pool_idx:
                    out[i] *= scale
            else:
                for i in pool_idx:
                    out[i] = pool_weight / len(pool_idx)
        else:
            # No pool asset in universe — pool_weight must be 0.
            if pool_weight > 1e-9:
                raise InfeasibleBasketError(
                    "mudarabah_pool_weight > 0 but no MUDARABAH_POOL asset in universe"
                )

        # 2. Sukuk band on the non-pool sleeve.
        if sk_idx:
            non_pool_sum = sum(out[i] for i in eq_idx + sk_idx)
            if non_pool_sum > 1e-12:
                sk_sum = sum(out[i] for i in sk_idx)
                target_sk = max(
                    policy.sukuk_min_weight * remaining,
                    min(policy.sukuk_max_weight * remaining, sk_sum),
                )
                if sk_sum > 1e-12 and target_sk != sk_sum:
                    scale = target_sk / sk_sum
                    for i in sk_idx:
                        out[i] *= scale
                elif sk_sum <= 1e-12 and target_sk > 0:
                    for i in sk_idx:
                        out[i] = target_sk / len(sk_idx)

        # 3. Single-name clamp.
        out = [min(wi, cap_single) for wi in out]

        # 4. Equity sector caps.
        for sector, idxs in by_eq_sector.items():
            cap = _sector_cap(policy, sector)
            s = sum(out[i] for i in idxs)
            if s > cap and s > 1e-12:
                scale = cap / s
                for i in idxs:
                    out[i] *= scale

        # 5. Renormalise.
        s = sum(out)
        if s <= 1e-12:
            raise InfeasibleBasketError("constraints reduce all weights to zero")
        out = [wi / s for wi in out]

        # 6. Stop conditions.
        if pool_idx:
            cur_pool = sum(out[i] for i in pool_idx)
            if abs(cur_pool - pool_weight) > 1e-6:
                continue
        if max(out) > cap_single + 1e-9:
            continue
        violated = False
        for sector, idxs in by_eq_sector.items():
            if sum(out[i] for i in idxs) > _sector_cap(policy, sector) + 1e-9:
                violated = True
                break
        if violated:
            continue
        # Sukuk band check (against `remaining`, not the full simplex —
        # the sukuk_min/max are fractions of the non-pool sleeve).
        if sk_idx and remaining > 1e-9:
            sk_sum_now = sum(out[i] for i in sk_idx)
            sk_min_abs = policy.sukuk_min_weight * remaining
            sk_max_abs = policy.sukuk_max_weight * remaining
            if sk_sum_now < sk_min_abs - 1e-9 or sk_sum_now > sk_max_abs + 1e-9:
                continue
        if not violated:
            break
    return out


def optimize(
    assets: Sequence[HalalAsset],
    *,
    covariance: Sequence[Sequence[float]] | None = None,
    policy: HalalPolicy | None = None,
    max_iter: int = 500,
    step_size: float = 0.05,
    tolerance: float = 1e-8,
) -> HalalOptResult:
    """Run the halal-native projected-gradient optimiser."""
    if not assets:
        raise ValueError("assets must be non-empty")
    if len(assets) > 200:
        raise ValueError("universe too large; pre-filter to ≤200")
    pol = policy if policy is not None else HalalPolicy()
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
    w = [1.0 / n] * n

    for _ in range(max_iter):
        grad = [-mu[i] for i in range(n)]
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += cov[i][j] * w[j]
            grad[i] += 2 * pol.risk_aversion * s
        max_grad = max(abs(g) for g in grad) if grad else 0.0
        eff_step = step_size
        if max_grad * step_size > 0.1:
            eff_step = 0.1 / max_grad
        new_w = [w[i] - eff_step * grad[i] for i in range(n)]
        new_w = _project_to_simplex(new_w)
        new_w = _apply_caps(new_w, assets, pol)
        delta = sum(abs(new_w[i] - w[i]) for i in range(n))
        w = new_w
        if delta < tolerance:
            break

    pr = sum(wi * a.expected_return for wi, a in zip(w, assets, strict=True))
    pv = 0.0
    for i in range(n):
        for j in range(n):
            pv += w[i] * cov[i][j] * w[j]
    eq_w = sum(
        wi for wi, a in zip(w, assets, strict=True) if a.asset_class is HalalAssetClass.EQUITY
    )
    sk_w = sum(
        wi for wi, a in zip(w, assets, strict=True) if a.asset_class is HalalAssetClass.SUKUK
    )
    md_w = sum(
        wi
        for wi, a in zip(w, assets, strict=True)
        if a.asset_class is HalalAssetClass.MUDARABAH_POOL
    )
    return HalalOptResult(
        weights=tuple(w),
        assets=tuple(assets),
        expected_return=pr,
        expected_variance=pv,
        mudarabah_weight=md_w,
        equity_weight=eq_w,
        sukuk_weight=sk_w,
    )


def render_result(result: HalalOptResult, *, top_n: int = 10) -> str:
    """Operator-readable summary."""
    head = (
        f"🕌 Halal portfolio: {len(result.assets)} assets — "
        f"mudarabah {result.mudarabah_weight * 100:.2f}% / "
        f"equity {result.equity_weight * 100:.2f}% / "
        f"sukuk {result.sukuk_weight * 100:.2f}%\n"
        f"  return {result.expected_return * 100:.2f}%, "
        f"vol {result.expected_volatility() * 100:.2f}%"
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
        emoji = {
            HalalAssetClass.EQUITY: "🟢",
            HalalAssetClass.SUKUK: "🟦",
            HalalAssetClass.MUDARABAH_POOL: "🟡",
        }[asset.asset_class]
        lines.append(f"  • {emoji} {asset.symbol} ({asset.sector}): {w * 100:.2f}%")
    return "\n".join(lines)
