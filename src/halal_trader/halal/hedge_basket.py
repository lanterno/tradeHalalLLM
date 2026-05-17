"""Inverse-asset hedge basket — Round-5 Wave 13.C.

Composes a hedge basket from inversely-correlated halal assets (gold,
silver, sukuk, halal cash equivalents) sized to offset a target
downside in the operator's primary equity / crypto portfolio. Unlike
the Wa'd-based portfolio insurance (Wave 13.B), this module is a
**static-weighting hedge basket** — no contracts, just direct holdings
in halal-permitted defensive assets.

Pinned semantics:

- **Closed-set HedgeAsset ladder** — gold, silver, sukuk, gold-backed-
  stablecoin, halal cash.
- **Closed-set BasketWeighting ladder** (EQUAL / RISK_PARITY /
  CUSTOM).
- **Hedge ratio cap** — operator-tunable; default 30% of portfolio
  notional in defensive basket.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum


class HedgeAsset(str, Enum):
    """Closed-set defensive halal hedge assets."""

    GOLD = "gold"
    SILVER = "silver"
    SUKUK = "sukuk"
    GOLD_BACKED_STABLECOIN = "gold_backed_stablecoin"
    HALAL_CASH = "halal_cash"


class BasketWeighting(str, Enum):
    """Closed-set basket-weighting strategies."""

    EQUAL = "equal"
    RISK_PARITY = "risk_parity"
    CUSTOM = "custom"


# Default per-asset volatility estimate (annualised) used for
# risk-parity weighting. Operators override via custom weights.
_DEFAULT_VOL: dict[HedgeAsset, float] = {
    HedgeAsset.GOLD: 0.18,
    HedgeAsset.SILVER: 0.30,
    HedgeAsset.SUKUK: 0.06,
    HedgeAsset.GOLD_BACKED_STABLECOIN: 0.18,
    HedgeAsset.HALAL_CASH: 0.005,
}


@dataclass(frozen=True)
class BasketPolicy:
    """Operator-tunable basket policy."""

    weighting: BasketWeighting = BasketWeighting.RISK_PARITY
    hedge_ratio: float = 0.30  # 30% of portfolio in defensive basket
    custom_weights: Mapping[HedgeAsset, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.hedge_ratio <= 1.0:
            raise ValueError("hedge_ratio must be in (0, 1]")
        if self.weighting is BasketWeighting.CUSTOM:
            if not self.custom_weights:
                raise ValueError("CUSTOM weighting requires custom_weights")
            total = sum(self.custom_weights.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError("custom_weights must sum to 1.0")
            for asset, w in self.custom_weights.items():
                if w < 0:
                    raise ValueError(f"weight for {asset.value} must be non-negative")


@dataclass(frozen=True)
class BasketAllocation:
    """A single asset's allocation in the basket."""

    asset: HedgeAsset
    weight: float
    notional: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        if self.notional < 0:
            raise ValueError("notional must be non-negative")


@dataclass(frozen=True)
class HedgeBasket:
    """The complete hedge basket."""

    portfolio_value: float
    hedge_notional: float
    allocations: tuple[BasketAllocation, ...]

    def __post_init__(self) -> None:
        if self.portfolio_value < 0:
            raise ValueError("portfolio_value must be non-negative")
        if self.hedge_notional < 0:
            raise ValueError("hedge_notional must be non-negative")


def _equal_weights(assets: Iterable[HedgeAsset]) -> dict[HedgeAsset, float]:
    asset_list = list(assets)
    if not asset_list:
        return {}
    w = 1.0 / len(asset_list)
    return {a: w for a in asset_list}


def _risk_parity_weights(
    assets: Iterable[HedgeAsset], vol_estimates: Mapping[HedgeAsset, float]
) -> dict[HedgeAsset, float]:
    asset_list = list(assets)
    if not asset_list:
        return {}
    inverse_vols = {
        a: 1.0 / max(vol_estimates.get(a, 0.10), 1e-6) for a in asset_list
    }
    total = sum(inverse_vols.values())
    if total == 0:
        return _equal_weights(asset_list)
    return {a: inv / total for a, inv in inverse_vols.items()}


def compose(
    portfolio_value: float,
    *,
    assets: Iterable[HedgeAsset],
    policy: BasketPolicy | None = None,
    vol_estimates: Mapping[HedgeAsset, float] | None = None,
) -> HedgeBasket:
    """Compose a hedge basket targeting ``hedge_ratio`` of portfolio_value."""
    if portfolio_value < 0:
        raise ValueError("portfolio_value must be non-negative")
    pol = policy if policy is not None else BasketPolicy()
    asset_list = list(assets)
    if not asset_list:
        return HedgeBasket(
            portfolio_value=portfolio_value, hedge_notional=0.0, allocations=()
        )

    vols = vol_estimates if vol_estimates is not None else _DEFAULT_VOL

    if pol.weighting is BasketWeighting.EQUAL:
        weights = _equal_weights(asset_list)
    elif pol.weighting is BasketWeighting.RISK_PARITY:
        weights = _risk_parity_weights(asset_list, vols)
    else:  # CUSTOM
        weights = dict(pol.custom_weights)

    hedge_notional = portfolio_value * pol.hedge_ratio
    allocations = tuple(
        BasketAllocation(
            asset=a,
            weight=weights.get(a, 0.0),
            notional=hedge_notional * weights.get(a, 0.0),
        )
        for a in asset_list
    )
    return HedgeBasket(
        portfolio_value=portfolio_value,
        hedge_notional=hedge_notional,
        allocations=allocations,
    )


def render_basket(basket: HedgeBasket) -> str:
    head = (
        f"🛡️ Hedge basket: ${basket.hedge_notional:.2f} "
        f"(of ${basket.portfolio_value:.2f} portfolio)"
    )
    lines = [head]
    for a in basket.allocations:
        lines.append(
            f"  • {a.asset.value:24s} weight={a.weight * 100:.1f}% "
            f"notional=${a.notional:.2f}"
        )
    return "\n".join(lines)
