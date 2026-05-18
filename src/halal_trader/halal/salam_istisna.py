"""Halal short alternatives — Salam (forward-sale) + Istisna (manufacturing-order).

Round-5 Wave 1.L primitive. Conventional shorting (borrowing the asset
to sell + later buy-back to cover) is impermissible: it sells what one
does not own + pays interest on the borrow. The classical fiqh
alternatives are:

- **Salam** — pre-paid forward sale of fungible commodity. The buyer
  (the operator's counterparty) pays now; the seller (the operator)
  delivers later. The operator receives cash up-front, the
  obligation is to deliver a fungible quantity. Used for hedging
  declining-price views on grain, metals, currencies that can be
  defined fungibly.

- **Istisna** — manufacturing / construction order. The buyer
  commissions a non-fungible asset; the seller manufactures it.
  Payment terms are flexible. Used for declining-view exposure to
  custom-built / non-fungible items.

This module ships the **structuring engine** that validates a proposed
contract against the AAOIFI rules (Standards 10 + 11) and produces an
operator-readable contract. Persistence + broker dispatch live one
layer up.

Pinned semantics:

- **Closed-set ContractKind ladder.** SALAM / ISTISNA only.
- **Salam requires fungible asset + full prepayment.** Both pinned
  in tests.
- **Istisna allows non-fungible + staged payment.**
- **Delivery date must be in future, capacity must match standard
  practice** — pinned in validation.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum


class ContractKind(str, Enum):
    """Two AAOIFI-recognised forward / order constructs."""

    SALAM = "salam"
    ISTISNA = "istisna"


class AssetClass(str, Enum):
    """The asset classes the contract delivers — pinned closed set."""

    AGRICULTURAL_COMMODITY = "agricultural_commodity"
    PRECIOUS_METAL = "precious_metal"
    INDUSTRIAL_METAL = "industrial_metal"
    CURRENCY = "currency"
    ENERGY_FUNGIBLE = "energy_fungible"
    MANUFACTURED_GOOD = "manufactured_good"
    CONSTRUCTED_PROPERTY = "constructed_property"


# Asset classes considered fungible for Salam under classical fiqh.
FUNGIBLE_ASSET_CLASSES: frozenset[AssetClass] = frozenset(
    {
        AssetClass.AGRICULTURAL_COMMODITY,
        AssetClass.PRECIOUS_METAL,
        AssetClass.INDUSTRIAL_METAL,
        AssetClass.CURRENCY,
        AssetClass.ENERGY_FUNGIBLE,
    }
)
"""Frozenset of asset classes Salam accepts. Manufactured / constructed go to Istisna."""


class ContractIssue(str, Enum):
    """Closed-set catalogue of issues the structurer can flag."""

    NON_FUNGIBLE_FOR_SALAM = "non_fungible_for_salam"
    INCOMPLETE_PREPAYMENT_FOR_SALAM = "incomplete_prepayment_for_salam"
    DELIVERY_NOT_IN_FUTURE = "delivery_not_in_future"
    DELIVERY_TOO_FAR = "delivery_too_far"
    QUANTITY_NON_POSITIVE = "quantity_non_positive"
    PRICE_NON_POSITIVE = "price_non_positive"
    DESCRIPTION_TOO_VAGUE = "description_too_vague"
    DELIVERY_LOCATION_MISSING = "delivery_location_missing"


@dataclass(frozen=True)
class StructuringPolicy:
    """Operator-tunable thresholds — defaults pin reasonable AAOIFI ranges."""

    salam_max_term_days: int = 365  # 1 year — classical jurists' ceiling
    istisna_max_term_days: int = 1825  # 5 years — common construction span
    min_description_chars: int = 20
    salam_prepayment_tolerance: float = 0.001  # within 0.1% of price = "full"

    def __post_init__(self) -> None:
        if self.salam_max_term_days <= 0 or self.istisna_max_term_days <= 0:
            raise ValueError("term days must be positive")
        if self.salam_max_term_days >= self.istisna_max_term_days:
            raise ValueError("salam term must be < istisna term")
        if self.min_description_chars <= 0:
            raise ValueError("min_description_chars must be positive")
        if not 0.0 <= self.salam_prepayment_tolerance <= 0.1:
            raise ValueError("salam_prepayment_tolerance must be in [0, 0.1]")


@dataclass(frozen=True)
class ContractInputs:
    """Inputs for a proposed Salam or Istisna contract."""

    contract_id: str
    kind: ContractKind
    asset_class: AssetClass
    description: str
    quantity: float
    quantity_unit: str
    delivery_location: str
    contracted_price: float
    prepayment_amount: float
    contract_date: date
    delivery_date: date

    def __post_init__(self) -> None:
        if not self.contract_id or not self.contract_id.strip():
            raise ValueError("contract_id must be non-empty")
        if not self.quantity_unit or not self.quantity_unit.strip():
            raise ValueError("quantity_unit must be non-empty")


@dataclass(frozen=True)
class StructuringResult:
    """Result of running a contract through the structurer."""

    contract_id: str
    issues: frozenset[ContractIssue]
    is_valid: bool

    def __post_init__(self) -> None:
        if self.is_valid and self.issues:
            raise ValueError("is_valid=True but issues non-empty")
        if (not self.is_valid) and not self.issues:
            raise ValueError("is_valid=False but issues empty")


def _is_fully_prepaid(price: float, prepayment: float, tolerance: float) -> bool:
    if price <= 0:
        return False
    return prepayment >= price * (1.0 - tolerance)


def structure_contract(
    inputs: ContractInputs,
    *,
    policy: StructuringPolicy | None = None,
) -> StructuringResult:
    """Validate a proposed Salam or Istisna contract; return a structuring result."""
    pol = policy if policy is not None else StructuringPolicy()
    issues: set[ContractIssue] = set()

    if inputs.quantity <= 0:
        issues.add(ContractIssue.QUANTITY_NON_POSITIVE)
    if inputs.contracted_price <= 0:
        issues.add(ContractIssue.PRICE_NON_POSITIVE)
    if not inputs.description or len(inputs.description) < pol.min_description_chars:
        issues.add(ContractIssue.DESCRIPTION_TOO_VAGUE)
    if not inputs.delivery_location or not inputs.delivery_location.strip():
        issues.add(ContractIssue.DELIVERY_LOCATION_MISSING)

    delta = inputs.delivery_date - inputs.contract_date
    if delta <= timedelta(0):
        issues.add(ContractIssue.DELIVERY_NOT_IN_FUTURE)

    if inputs.kind is ContractKind.SALAM:
        if delta > timedelta(days=pol.salam_max_term_days):
            issues.add(ContractIssue.DELIVERY_TOO_FAR)
        if inputs.asset_class not in FUNGIBLE_ASSET_CLASSES:
            issues.add(ContractIssue.NON_FUNGIBLE_FOR_SALAM)
        if not _is_fully_prepaid(
            inputs.contracted_price,
            inputs.prepayment_amount,
            pol.salam_prepayment_tolerance,
        ):
            issues.add(ContractIssue.INCOMPLETE_PREPAYMENT_FOR_SALAM)
    else:  # ISTISNA
        if delta > timedelta(days=pol.istisna_max_term_days):
            issues.add(ContractIssue.DELIVERY_TOO_FAR)
        # Istisna allows partial prepayment — no INCOMPLETE_PREPAYMENT issue.

    return StructuringResult(
        contract_id=inputs.contract_id,
        issues=frozenset(issues),
        is_valid=len(issues) == 0,
    )


# --- Render ------------------------------------------------------------------


_FORBIDDEN_RENDER_TOKENS: tuple[str, ...] = (
    "@",
    "zoom.us",
    "meet.google",
    "private_email",
    "+1-",
    "Authorization",
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_contract(inputs: ContractInputs, result: StructuringResult) -> str:
    emoji = "✅" if result.is_valid else "❌"
    head = (
        f"{emoji} {inputs.contract_id} — {inputs.kind.value} "
        f"{inputs.quantity:.2f} {inputs.quantity_unit} of {inputs.asset_class.value} "
        f"@ {inputs.contracted_price:.2f}"
    )
    lines = [
        head,
        f"  delivery: {inputs.delivery_date.isoformat()} at {inputs.delivery_location}",
    ]
    for issue in sorted(result.issues, key=lambda x: x.value):
        lines.append(f"  • {issue.value}")
    return _scrub("\n".join(lines))
