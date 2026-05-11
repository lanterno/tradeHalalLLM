"""Currency hedging via halal forwards — Round-5 Wave 13.E.

Conventional FX forwards are forbidden under classical fiqh because
they involve a future exchange of two currencies (riba al-fadl /
deferred-currency exchange) and leverage / margin (riba al-nasi'ah).

The halal alternative substitutes a **Salam-style currency contract**
where one currency is delivered at spot (immediate exchange satisfies
the bay' al-sarf rule of "hand-to-hand" currency trade) while the
hedger holds a **commitment to convert** at a pre-agreed schedule. The
construct relies on:

1. Two parallel Murabaha-spot exchanges (no deferred exchange of
   currencies — both legs settle on initiation), and
2. A Wa'd-based commitment to roll the resulting balance into the
   target currency on a specified date.

This module is the **hedge planner**. It:
- Takes a multi-currency portfolio + target base currency,
- Computes net exposure per non-base currency,
- Produces a hedge plan with one Wa'd-commitment per currency leg,
- Validates AAOIFI Standard 1 (bay' al-sarf) compliance pins.

Pinned semantics:

- **Base currency must be pre-declared** — operator picks once, hedge
  plans are computed against it.
- **Spot leg settles immediately.** Pin: `spot_settlement_days = 0` or
  `1` per Standard 1's "tilbasti tilbasti" (hand-to-hand) requirement;
  T+2 is rejected by default.
- **No interest / forward points.** Forward points are riba; the
  hedge price is the *spot* with an explicit *Wa'd-roll fee* (a
  fixed Wakalah service fee — not a basis-point spread on time).
- **Closed-set HedgeHorizon ladder.** SHORT (≤30d), MEDIUM (≤90d),
  LONG (≤180d). Beyond LONG requires explicit scholar review.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render — counterparty IDs masked.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum


class HedgeHorizon(str, Enum):
    """Closed-set hedge tenor ladder."""

    SHORT = "short"  # ≤30 days
    MEDIUM = "medium"  # ≤90 days
    LONG = "long"  # ≤180 days


_HORIZON_MAX_DAYS: dict[HedgeHorizon, int] = {
    HedgeHorizon.SHORT: 30,
    HedgeHorizon.MEDIUM: 90,
    HedgeHorizon.LONG: 180,
}


@dataclass(frozen=True)
class CurrencyExposure:
    """A single currency leg in the operator's portfolio."""

    currency: str  # ISO 4217 (e.g. USD, SAR, AED, MYR, IDR)
    amount: float  # positive = long, negative = short

    def __post_init__(self) -> None:
        if not self.currency or len(self.currency) != 3:
            raise ValueError("currency must be a 3-letter ISO code")
        if self.currency != self.currency.upper():
            raise ValueError("currency must be uppercase")
        if self.amount == 0:
            raise ValueError("amount must be non-zero (no zero exposures)")


@dataclass(frozen=True)
class FXSpotRate:
    """A spot rate quote — one base unit costs `rate` of the quote currency."""

    base: str
    quote: str
    rate: float

    def __post_init__(self) -> None:
        if not self.base or len(self.base) != 3:
            raise ValueError("base must be a 3-letter ISO code")
        if not self.quote or len(self.quote) != 3:
            raise ValueError("quote must be a 3-letter ISO code")
        if self.base == self.quote:
            raise ValueError("base and quote must differ")
        if self.rate <= 0:
            raise ValueError("rate must be positive")


@dataclass(frozen=True)
class FXHedgePolicy:
    """Operator-tunable hedge policy."""

    base_currency: str
    horizon: HedgeHorizon = HedgeHorizon.MEDIUM
    spot_settlement_days: int = 1
    """T+0 (=0) or T+1 (=1) per Standard 1 bay' al-sarf. T+2 is
    rejected — operators must change the broker if their CSD demands
    T+2. Pinned in __post_init__."""
    wakalah_fee_bps: float = 5.0
    """Fixed-fee service charge for the Wa'd roll, in basis points of
    the notional. NOT interest — a flat fee for the agency service."""
    min_notional: float = 1000.0
    """Below this notional, the hedge is not worth the fee — skip."""

    def __post_init__(self) -> None:
        if not self.base_currency or len(self.base_currency) != 3:
            raise ValueError("base_currency must be a 3-letter ISO code")
        if self.spot_settlement_days not in (0, 1):
            raise ValueError("spot_settlement_days must be 0 or 1 per AAOIFI Standard 1")
        if not 0.0 <= self.wakalah_fee_bps < 100:
            raise ValueError("wakalah_fee_bps must be in [0, 100)")
        if self.min_notional < 0:
            raise ValueError("min_notional must be non-negative")


@dataclass(frozen=True)
class FXHedgeLeg:
    """A single currency leg of the hedge plan."""

    currency: str
    notional_in_base: float
    """Amount converted to base currency at spot."""
    spot_rate_to_base: float
    spot_settlement_date: date
    waad_roll_date: date
    wakalah_fee: float
    """Fixed fee on this leg (in base currency)."""

    def __post_init__(self) -> None:
        if self.notional_in_base <= 0:
            raise ValueError("notional_in_base must be positive")
        if self.spot_rate_to_base <= 0:
            raise ValueError("spot_rate_to_base must be positive")
        if self.spot_settlement_date >= self.waad_roll_date:
            raise ValueError("waad_roll_date must be after spot_settlement_date")


@dataclass(frozen=True)
class FXHedgePlan:
    """Output of `plan_fx_hedge`."""

    base_currency: str
    plan_date: date
    legs: tuple[FXHedgeLeg, ...]
    skipped_currencies: tuple[str, ...]
    """Exposures below `min_notional` after spot conversion — not hedged."""
    total_notional_in_base: float
    total_wakalah_fee: float

    def hedge_count(self) -> int:
        return len(self.legs)


def _spot_rate_to_base(currency: str, base: str, rates: Sequence[FXSpotRate]) -> float:
    """Find the spot rate from `currency` to `base`.

    Looks up direct or inverse quotes. Raises if no match.
    """
    if currency == base:
        return 1.0
    for r in rates:
        if r.base == currency and r.quote == base:
            return r.rate
        if r.base == base and r.quote == currency:
            return 1.0 / r.rate
    raise ValueError(f"no spot rate for {currency}/{base}")


def plan_fx_hedge(
    exposures: Iterable[CurrencyExposure],
    rates: Sequence[FXSpotRate],
    policy: FXHedgePolicy,
    *,
    plan_date: date,
) -> FXHedgePlan:
    """Build a halal FX hedge plan against the policy's base currency.

    Each non-base exposure produces one FXHedgeLeg with:
    - spot_settlement_date = plan_date + spot_settlement_days
    - waad_roll_date = spot_settlement_date + horizon_days
    - notional_in_base = abs(exposure.amount) × spot_to_base
    - wakalah_fee = notional_in_base × (wakalah_fee_bps / 1e4)

    Below-min-notional currencies are listed in `skipped_currencies`.
    """
    horizon_days = _HORIZON_MAX_DAYS[policy.horizon]
    spot_settle = plan_date + timedelta(days=policy.spot_settlement_days)
    roll_date = spot_settle + timedelta(days=horizon_days)
    legs: list[FXHedgeLeg] = []
    skipped: list[str] = []
    total_notional = 0.0
    total_fee = 0.0
    for exp in exposures:
        if exp.currency == policy.base_currency:
            continue
        spot_to_base = _spot_rate_to_base(exp.currency, policy.base_currency, rates)
        notional = abs(exp.amount) * spot_to_base
        if notional < policy.min_notional:
            skipped.append(exp.currency)
            continue
        fee = notional * (policy.wakalah_fee_bps / 1e4)
        legs.append(
            FXHedgeLeg(
                currency=exp.currency,
                notional_in_base=notional,
                spot_rate_to_base=spot_to_base,
                spot_settlement_date=spot_settle,
                waad_roll_date=roll_date,
                wakalah_fee=fee,
            )
        )
        total_notional += notional
        total_fee += fee
    return FXHedgePlan(
        base_currency=policy.base_currency,
        plan_date=plan_date,
        legs=tuple(legs),
        skipped_currencies=tuple(skipped),
        total_notional_in_base=total_notional,
        total_wakalah_fee=total_fee,
    )


def net_exposure_by_currency(
    exposures: Iterable[CurrencyExposure],
) -> tuple[CurrencyExposure, ...]:
    """Sum all exposures per currency. Long-short cancellation included.

    Returns one CurrencyExposure per currency with non-zero net amount.
    Used as a pre-filter before plan_fx_hedge.
    """
    net: dict[str, float] = {}
    for exp in exposures:
        net[exp.currency] = net.get(exp.currency, 0.0) + exp.amount
    out: list[CurrencyExposure] = []
    for currency, amt in net.items():
        if amt != 0:
            out.append(CurrencyExposure(currency=currency, amount=amt))
    out.sort(key=lambda e: e.currency)
    return tuple(out)


def render_plan(plan: FXHedgePlan) -> str:
    """Operator-readable summary of the hedge plan."""
    head = (
        f"💱 FX hedge plan ({plan.base_currency}, {plan.plan_date.isoformat()}): "
        f"{plan.hedge_count()} legs, "
        f"notional={plan.total_notional_in_base:.2f}, "
        f"fee={plan.total_wakalah_fee:.2f}"
    )
    lines = [head]
    for leg in plan.legs:
        lines.append(
            f"  • {leg.currency}: {leg.notional_in_base:.2f} {plan.base_currency} "
            f"@ {leg.spot_rate_to_base:.4f}, "
            f"spot {leg.spot_settlement_date.isoformat()}, "
            f"roll {leg.waad_roll_date.isoformat()}, "
            f"fee {leg.wakalah_fee:.2f}"
        )
    if plan.skipped_currencies:
        lines.append(f"  • Skipped (below min): {', '.join(plan.skipped_currencies)}")
    return "\n".join(lines)
