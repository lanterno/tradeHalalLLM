"""Multi-currency Zakat calculator.

Round-5 Wave 1.J primitive. Zakat is the annual wealth tax owed by
Muslims whose net assets exceed nisab (the threshold) at the end
of one lunar (Hijri) year (hawl). The standard rate is 2.5%
(2.5/100 = 1/40th) on net wealth above nisab. The platform's
existing `halal/purification.py` covers a different concept:
purification of incidental riba income from screened holdings;
zakat is the broader wealth-tax obligation on the user's full
zakatable assets.

Picked a pure-functional calculator with operator-supplied FX
rates over an internal FX feed because (a) the calculator must
be testable without network calls; (b) the FX rates used for
zakat calculation are a documented decision the user makes once
per year (typically the spot rate at hawl-end), so making them
operator-supplied is the right user experience; (c) the lunar-
year arithmetic is operator-tunable (some users use the actual
moon-sighting date; others use the algorithmic Hijri calendar);
encoding hawl_start_date as input keeps the engine policy-free.

Pinned semantics:
- **Closed-set NisabBasis enum.** GOLD or SILVER. Silver basis is
  more conservative (lower threshold → more zakat); some scholars
  recommend silver in modern times because silver-nisab in
  current prices is below gold-nisab. Operator picks based on
  their school + scholar's preference.
- **2.5% standard rate, operator-tunable.** The 2.5% rate is the
  Sunni majority position (1/40); some Shia methodologies use
  different rates (Khums = 20% on certain categories). The
  policy lets operators override.
- **Net assets = cash + investments + monetized gold/silver +
  loans-owed-to-user - debts-owed-by-user.** All converted to
  the reporting currency via supplied FX rates.
- **Hawl is 354 lunar days (Hijri year).** The due_date computation
  uses 354 days from hawl_start_date — this is the standard
  approximation; operators wanting moon-sighting precision feed
  the actual hawl_start_date themselves.
- **Render output never includes per-account balances.** The
  render shows only nisab status + total + zakat owed; the
  detailed asset breakdown is operator-private.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

# Module-level historic nisab thresholds. These are documented
# constants from classical fiqh references; operators can override
# via ZakatPolicy if they follow a scholar with different reading.
DEFAULT_GOLD_NISAB_GRAMS: float = 87.48  # 20 mithqal × 4.374g
DEFAULT_SILVER_NISAB_GRAMS: float = 612.36  # 200 dirham × 3.0618g
DEFAULT_ZAKAT_RATE: float = 0.025  # 2.5% (Sunni majority)
LUNAR_YEAR_DAYS: int = 354  # Hijri year approximation


class NisabBasis(str, Enum):
    """Which nisab threshold to apply.

    Pinned string values for JSON / DB persistence. GOLD historical
    nisab is 87.48g; SILVER is 612.36g — at modern gold/silver
    prices, silver-nisab is the lower (more conservative) threshold,
    which is why some contemporary scholars recommend it.
    """

    GOLD = "gold"
    SILVER = "silver"


@dataclass(frozen=True)
class ZakatPolicy:
    """Operator-tunable Zakat policy.

    Defaults are the Sunni majority position. Shia operators using
    Khums methodology should pass their own rates.
    """

    zakat_rate: float = DEFAULT_ZAKAT_RATE
    gold_nisab_grams: float = DEFAULT_GOLD_NISAB_GRAMS
    silver_nisab_grams: float = DEFAULT_SILVER_NISAB_GRAMS
    lunar_year_days: int = LUNAR_YEAR_DAYS

    def __post_init__(self) -> None:
        if not (0 < self.zakat_rate <= 1.0):
            raise ValueError("zakat_rate must be in (0, 1]")
        if self.gold_nisab_grams <= 0:
            raise ValueError("gold_nisab_grams must be > 0")
        if self.silver_nisab_grams <= 0:
            raise ValueError("silver_nisab_grams must be > 0")
        if self.lunar_year_days < 1:
            raise ValueError("lunar_year_days must be >= 1")


@dataclass(frozen=True)
class FxRates:
    """FX rates + precious-metal prices in the reporting currency.

    `rates` maps each non-base currency code to its multiplier
    against the reporting currency (e.g., if reporting in USD and
    holding 1000 SAR, FxRates.rates["SAR"] should be the USD-per-SAR
    rate so that USD value = 1000 * rates["SAR"]). The reporting
    currency itself implicitly has rate 1.0.
    """

    base_currency: str
    rates: Mapping[str, float]
    gold_price_per_gram: float
    silver_price_per_gram: float

    def __post_init__(self) -> None:
        if not self.base_currency or not self.base_currency.strip():
            raise ValueError("base_currency must be non-empty")
        if self.gold_price_per_gram <= 0:
            raise ValueError("gold_price_per_gram must be > 0")
        if self.silver_price_per_gram <= 0:
            raise ValueError("silver_price_per_gram must be > 0")
        for code, rate in self.rates.items():
            if not code or not code.strip():
                raise ValueError("currency code must be non-empty")
            if rate <= 0:
                raise ValueError(f"rate for {code} must be > 0 (got {rate})")

    def to_base(self, amount: float, currency: str) -> float:
        """Convert `amount` in `currency` to base currency.

        Base currency itself returns the amount unchanged.
        Unknown currency raises KeyError.
        """

        if currency == self.base_currency:
            return amount
        if currency not in self.rates:
            raise KeyError(f"no FX rate for currency {currency!r}")
        return amount * self.rates[currency]


@dataclass(frozen=True)
class ZakatInputs:
    """User's zakatable position at hawl-end.

    All amounts are in the per-currency native units; the
    calculator uses FxRates to convert. Gold + silver are in grams
    of the metal itself (priced via FxRates).
    """

    cash_by_currency: Mapping[str, float] = field(default_factory=dict)
    investments_by_currency: Mapping[str, float] = field(default_factory=dict)
    gold_grams: float = 0.0
    silver_grams: float = 0.0
    debts_owed_to_user_by_currency: Mapping[str, float] = field(default_factory=dict)
    debts_owed_by_user_by_currency: Mapping[str, float] = field(default_factory=dict)
    reporting_currency: str = "USD"
    hawl_start_date: date | None = None

    def __post_init__(self) -> None:
        if not self.reporting_currency or not self.reporting_currency.strip():
            raise ValueError("reporting_currency must be non-empty")
        if self.gold_grams < 0:
            raise ValueError("gold_grams must be >= 0")
        if self.silver_grams < 0:
            raise ValueError("silver_grams must be >= 0")
        for label, m in [
            ("cash_by_currency", self.cash_by_currency),
            ("investments_by_currency", self.investments_by_currency),
            ("debts_owed_to_user_by_currency", self.debts_owed_to_user_by_currency),
            ("debts_owed_by_user_by_currency", self.debts_owed_by_user_by_currency),
        ]:
            for code, amount in m.items():
                if not code or not code.strip():
                    raise ValueError(f"{label}: currency code must be non-empty")
                if amount < 0:
                    raise ValueError(f"{label}: amount for {code} must be >= 0 (got {amount})")


@dataclass(frozen=True)
class ZakatCalculation:
    """Output of the calculator."""

    net_assets: float
    nisab_value: float
    meets_nisab: bool
    zakat_owed: float
    basis_used: NisabBasis
    reporting_currency: str
    hawl_due_date: date | None

    def __post_init__(self) -> None:
        if self.net_assets < 0:
            # Net can be negative (more debt than assets); zakat is 0
            # but the net-assets value is still informational.
            pass
        if self.nisab_value <= 0:
            raise ValueError("nisab_value must be > 0")
        if self.zakat_owed < 0:
            raise ValueError("zakat_owed must be >= 0")
        if not self.reporting_currency or not self.reporting_currency.strip():
            raise ValueError("reporting_currency must be non-empty")
        if self.meets_nisab and self.zakat_owed == 0 and self.net_assets > 0:
            raise ValueError("meets_nisab=True with zakat_owed=0 is inconsistent")
        if not self.meets_nisab and self.zakat_owed > 0:
            raise ValueError("meets_nisab=False with zakat_owed>0 is inconsistent")


def _sum_in_base(amounts: Mapping[str, float], fx: FxRates) -> float:
    """Sum a per-currency amount mapping in fx.base_currency."""

    return sum(fx.to_base(amount, code) for code, amount in amounts.items())


def calculate_zakat(
    inputs: ZakatInputs,
    fx: FxRates,
    *,
    policy: ZakatPolicy = ZakatPolicy(),
    basis: NisabBasis = NisabBasis.SILVER,
) -> ZakatCalculation:
    """Compute zakat owed.

    Net assets = cash + investments + (gold * gold_price) +
    (silver * silver_price) + loans-owed-to-user - debts-owed.
    All converted to inputs.reporting_currency.

    Default basis is SILVER (more conservative — typically lower
    threshold in modern prices). Operator can pass GOLD if their
    scholar prefers.

    Default policy uses the 2.5% Sunni majority rate. Khums users
    pass their own ZakatPolicy.

    If hawl_start_date is provided, hawl_due_date is computed as
    start + lunar_year_days. Otherwise the due date is None.
    """

    if inputs.reporting_currency != fx.base_currency:
        raise ValueError(
            f"FxRates.base_currency {fx.base_currency!r} must match "
            f"inputs.reporting_currency {inputs.reporting_currency!r}"
        )

    cash = _sum_in_base(inputs.cash_by_currency, fx)
    investments = _sum_in_base(inputs.investments_by_currency, fx)
    gold_value = inputs.gold_grams * fx.gold_price_per_gram
    silver_value = inputs.silver_grams * fx.silver_price_per_gram
    loans_to_user = _sum_in_base(inputs.debts_owed_to_user_by_currency, fx)
    debts_owed = _sum_in_base(inputs.debts_owed_by_user_by_currency, fx)

    gross = cash + investments + gold_value + silver_value + loans_to_user
    net = gross - debts_owed

    if basis is NisabBasis.GOLD:
        nisab_value = policy.gold_nisab_grams * fx.gold_price_per_gram
    else:
        nisab_value = policy.silver_nisab_grams * fx.silver_price_per_gram

    meets_nisab = net >= nisab_value
    zakat_owed = net * policy.zakat_rate if meets_nisab else 0.0

    hawl_due: date | None = None
    if inputs.hawl_start_date is not None:
        hawl_due = inputs.hawl_start_date + timedelta(days=policy.lunar_year_days)

    return ZakatCalculation(
        net_assets=net,
        nisab_value=nisab_value,
        meets_nisab=meets_nisab,
        zakat_owed=zakat_owed,
        basis_used=basis,
        reporting_currency=inputs.reporting_currency,
        hawl_due_date=hawl_due,
    )


def days_until_hawl(
    calculation: ZakatCalculation,
    *,
    today: date,
) -> int | None:
    """Days remaining until hawl due date (negative if past).

    Returns None if hawl_due_date isn't set on the calculation.
    Operators surface in the dashboard "your zakat is due in X
    days" tile.
    """

    if calculation.hawl_due_date is None:
        return None
    return (calculation.hawl_due_date - today).days


def render_calculation(calculation: ZakatCalculation) -> str:
    """Format the calculation for ops display.

    No-secret-leak: shows only summary numbers + nisab status; the
    detailed per-currency / per-asset breakdown is operator-private.
    """

    if calculation.meets_nisab:
        emoji = "💰"
        verdict = f"OWED {calculation.zakat_owed:,.2f} {calculation.reporting_currency}"
    else:
        emoji = "✅"
        verdict = "BELOW NISAB — no zakat owed"
    lines = [
        f"{emoji} Zakat: {verdict}",
        f"  net assets: {calculation.net_assets:,.2f} {calculation.reporting_currency}",
        f"  nisab ({calculation.basis_used.value}): "
        f"{calculation.nisab_value:,.2f} {calculation.reporting_currency}",
    ]
    if calculation.hawl_due_date is not None:
        lines.append(f"  due: {calculation.hawl_due_date.isoformat()}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_GOLD_NISAB_GRAMS",
    "DEFAULT_SILVER_NISAB_GRAMS",
    "DEFAULT_ZAKAT_RATE",
    "FxRates",
    "LUNAR_YEAR_DAYS",
    "NisabBasis",
    "ZakatCalculation",
    "ZakatInputs",
    "ZakatPolicy",
    "calculate_zakat",
    "days_until_hawl",
    "render_calculation",
]
