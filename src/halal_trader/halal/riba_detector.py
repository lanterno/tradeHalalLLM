"""Riba detection across derivative instrument classes.

Round-5 Wave 1.F primitive. Standard halal screening covers
equity ratios + sector. The maysir screen catches gambling-pattern
equities; the gharar detector catches structural uncertainty.
This module catches the fourth and oldest Shariah prohibition:
riba (interest / usury) embedded in instrument structure.

Riba comes in two classical forms:
- **Riba al-nasiyah**: interest on delayed payment / loans
  (the canonical form — every conventional bond pays this)
- **Riba al-fadl**: excess in same-genus exchange (swapping 1kg
  gold-now for 1.1kg gold-later; trading a face-value bond at
  discount is bay' al-dayn, a related concern)

Modern instrument structures contain riba in subtler forms:
- **Embedded financing rate** in conventional futures (cost-of-
  carry includes the risk-free rate)
- **Leverage interest** in CFDs, margin trading, leveraged ETFs
- **Debt sale** when bonds trade at non-face value (bay' al-dayn)

Picked an instrument-class-keyed classifier with additive signal
flags because (a) most riba determinations are deterministic from
the instrument type — every conventional bond pays riba al-
nasiyah, period — so a lookup table is the natural representation;
(b) overlay flags (uses_margin, has_embedded_financing) catch the
operator-supplied edge cases on top of the base class; (c) the
mapping is well-settled fiqh and matches AAOIFI Standard 21
Section 3 + Standard 30 (commodity Murabaha) — operators don't
need to interpret; they look up.

Pinned semantics:
- **Closed-set InstrumentClass catalogue.** 14 documented classes
  span the canonical halal + non-halal instrument types. Adding a
  class is a code review change.
- **Closed-set RibaType catalogue.** NONE / NASIYAH / FADL /
  EMBEDDED_FINANCING / DEBT_SALE / LEVERAGE_INTEREST. Each maps
  to a documented fiqh reference.
- **Base verdict from instrument class; signal flags add on top.**
  A SPOT_EQUITY base is clean unless margin/borrowed flags fire;
  a CONVENTIONAL_BOND base always carries NASIYAH regardless of
  flags.
- **`is_clean(assessment)` is the load-bearing gate.** Returns
  True only when riba_types is empty.
- **Render output never includes broker-internal funding rate
  details or counterparty-specific terms.** Only the instrument
  class + detected riba types; the per-trade financing schedule
  goes to the operator-side DB.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class RibaType(str, Enum):
    """Catalogue of riba (interest / usury) violation types.

    Pinned string values for JSON / DB persistence stability.
    Each maps to a documented classical-fiqh reference.
    """

    NASIYAH = "nasiyah"  # Interest on delayed payment / loans
    FADL = "fadl"  # Excess in same-genus exchange
    EMBEDDED_FINANCING = "embedded_financing"  # Time-value-of-money baked in
    DEBT_SALE = "debt_sale"  # Bay' al-dayn (debt traded off-face)
    LEVERAGE_INTEREST = "leverage_interest"  # Margin / borrow / financing


class InstrumentClass(str, Enum):
    """Catalogue of derivative + instrument classes.

    Pinned string values. Covers the canonical halal-tradable
    set (spot equity, sukuk, wa'd-forward, salam-forward,
    arboun-option, physical commodity) AND the canonical non-
    halal set (conventional bond, conventional future,
    conventional option, swap, CFD, leveraged/inverse ETF).
    """

    SPOT_EQUITY = "spot_equity"
    PHYSICAL_COMMODITY = "physical_commodity"
    SUKUK = "sukuk"
    WAAD_FORWARD = "waad_forward"
    SALAM_FORWARD = "salam_forward"
    ARBOUN_OPTION = "arboun_option"
    CONVENTIONAL_BOND = "conventional_bond"
    CONVENTIONAL_FUTURE = "conventional_future"
    CONVENTIONAL_OPTION = "conventional_option"
    INTEREST_RATE_SWAP = "interest_rate_swap"
    CURRENCY_SWAP = "currency_swap"
    CFD = "cfd"
    LEVERAGED_ETF = "leveraged_etf"
    INVERSE_ETF = "inverse_etf"


# Module-level base verdict mapping. Keyed by InstrumentClass;
# value is the frozenset of riba types the class always carries
# (regardless of operator flags). Operator flags can ADD types
# but cannot REMOVE — once an instrument's structure embeds riba,
# no operator-side configuration changes that.
_BASE_VERDICT: dict[InstrumentClass, frozenset[RibaType]] = {
    # Halal-by-default classes (clean unless flags fire)
    InstrumentClass.SPOT_EQUITY: frozenset(),
    InstrumentClass.PHYSICAL_COMMODITY: frozenset(),
    InstrumentClass.SUKUK: frozenset(),
    InstrumentClass.WAAD_FORWARD: frozenset(),
    InstrumentClass.SALAM_FORWARD: frozenset(),
    InstrumentClass.ARBOUN_OPTION: frozenset(),
    # Riba-bearing classes (always flagged)
    InstrumentClass.CONVENTIONAL_BOND: frozenset({RibaType.NASIYAH}),
    InstrumentClass.CONVENTIONAL_FUTURE: frozenset({RibaType.EMBEDDED_FINANCING}),
    InstrumentClass.CONVENTIONAL_OPTION: frozenset({RibaType.EMBEDDED_FINANCING}),
    InstrumentClass.INTEREST_RATE_SWAP: frozenset({RibaType.NASIYAH}),
    InstrumentClass.CURRENCY_SWAP: frozenset({RibaType.EMBEDDED_FINANCING}),
    InstrumentClass.CFD: frozenset({RibaType.EMBEDDED_FINANCING, RibaType.LEVERAGE_INTEREST}),
    InstrumentClass.LEVERAGED_ETF: frozenset({RibaType.LEVERAGE_INTEREST}),
    InstrumentClass.INVERSE_ETF: frozenset({RibaType.EMBEDDED_FINANCING}),
}


# Module-level set of "halal-by-default" instrument classes — the
# classes whose base verdict is empty (no riba unless operator
# flags fire). Used by the operator dashboard to filter the
# tradable-universe view.
HALAL_BY_DEFAULT_CLASSES: frozenset[InstrumentClass] = frozenset(
    cls for cls, types in _BASE_VERDICT.items() if not types
)


@dataclass(frozen=True)
class RibaPolicy:
    """Operator-tunable riba-detector policy.

    Defaults are conservative: margin and borrowed securities both
    flag LEVERAGE_INTEREST / NASIYAH respectively. Some operators
    (rare) follow a scholar permitting margin under specific
    Murabaha-like structures; the policy exposes the override.
    """

    flag_margin_as_riba: bool = True
    flag_borrow_as_riba: bool = True

    # Note: there is no `flag_X = False` knob to remove the base
    # verdict for non-halal instrument classes — that's a fiqh
    # determination, not a policy. Operators wanting different
    # base verdicts for an instrument class should petition for
    # a code review change.


@dataclass(frozen=True)
class RibaInputs:
    """Per-instrument inputs for the riba detector.

    `instrument_class` drives the base verdict; the boolean flags
    add on top for cases where the operator's specific use of an
    otherwise-clean class introduces riba (margin, borrowed
    securities, embedded financing, fixed-interest payment,
    debt sold off-face).
    """

    instrument_id: str
    instrument_class: InstrumentClass
    uses_margin: bool = False
    uses_borrowed_securities: bool = False
    has_embedded_financing_rate: bool = False
    pays_or_receives_fixed_interest: bool = False
    is_debt_traded_off_face: bool = False

    def __post_init__(self) -> None:
        if not self.instrument_id or not self.instrument_id.strip():
            raise ValueError("instrument_id must be non-empty")


@dataclass(frozen=True)
class RibaAssessment:
    """Output of the riba detector for one instrument."""

    instrument_id: str
    instrument_class: InstrumentClass
    riba_types: frozenset[RibaType]

    def __post_init__(self) -> None:
        if not self.instrument_id or not self.instrument_id.strip():
            raise ValueError("instrument_id must be non-empty")

    @property
    def is_clean(self) -> bool:
        """True when no riba types fire."""

        return not self.riba_types


def _detect_extra_signals(
    inputs: RibaInputs,
    *,
    policy: RibaPolicy,
) -> frozenset[RibaType]:
    """Run the operator-flag-driven signal detectors.

    Returns the additional riba types beyond the base verdict.
    """

    extra: set[RibaType] = set()
    if inputs.uses_margin and policy.flag_margin_as_riba:
        extra.add(RibaType.LEVERAGE_INTEREST)
    if inputs.uses_borrowed_securities and policy.flag_borrow_as_riba:
        extra.add(RibaType.NASIYAH)
    if inputs.has_embedded_financing_rate:
        extra.add(RibaType.EMBEDDED_FINANCING)
    if inputs.pays_or_receives_fixed_interest:
        extra.add(RibaType.NASIYAH)
    if inputs.is_debt_traded_off_face:
        extra.add(RibaType.DEBT_SALE)
    return frozenset(extra)


def assess_riba(
    inputs: RibaInputs,
    *,
    policy: RibaPolicy = RibaPolicy(),
) -> RibaAssessment:
    """Run the riba detector for one instrument.

    Returns the assessment with combined base + flag-driven riba
    types. Operators consult `assessment.is_clean` as the
    load-bearing gate.
    """

    base = _BASE_VERDICT[inputs.instrument_class]
    extra = _detect_extra_signals(inputs, policy=policy)
    return RibaAssessment(
        instrument_id=inputs.instrument_id,
        instrument_class=inputs.instrument_class,
        riba_types=base | extra,
    )


def assess_batch(
    inputs_list: Iterable[RibaInputs],
    *,
    policy: RibaPolicy = RibaPolicy(),
) -> tuple[RibaAssessment, ...]:
    """Run the detector across many instruments; sorted by id.

    Deterministic ordering for the dashboard tile + email summary.
    """

    assessments = [assess_riba(i, policy=policy) for i in inputs_list]
    assessments.sort(key=lambda a: a.instrument_id)
    return tuple(assessments)


def filter_blocked(
    assessments: Iterable[RibaAssessment],
) -> tuple[RibaAssessment, ...]:
    """Return only the assessments with at least one riba type."""

    return tuple(a for a in assessments if not a.is_clean)


_RIBA_TYPE_LABEL: dict[RibaType, str] = {
    RibaType.NASIYAH: "riba al-nasiyah",
    RibaType.FADL: "riba al-fadl",
    RibaType.EMBEDDED_FINANCING: "embedded financing",
    RibaType.DEBT_SALE: "debt sale (bay' al-dayn)",
    RibaType.LEVERAGE_INTEREST: "leverage interest",
}


_CLASS_LABEL: dict[InstrumentClass, str] = {
    InstrumentClass.SPOT_EQUITY: "spot equity",
    InstrumentClass.PHYSICAL_COMMODITY: "physical commodity",
    InstrumentClass.SUKUK: "sukuk",
    InstrumentClass.WAAD_FORWARD: "wa'd forward",
    InstrumentClass.SALAM_FORWARD: "salam forward",
    InstrumentClass.ARBOUN_OPTION: "arboun option",
    InstrumentClass.CONVENTIONAL_BOND: "conventional bond",
    InstrumentClass.CONVENTIONAL_FUTURE: "conventional future",
    InstrumentClass.CONVENTIONAL_OPTION: "conventional option",
    InstrumentClass.INTEREST_RATE_SWAP: "interest rate swap",
    InstrumentClass.CURRENCY_SWAP: "currency swap",
    InstrumentClass.CFD: "CFD",
    InstrumentClass.LEVERAGED_ETF: "leveraged ETF",
    InstrumentClass.INVERSE_ETF: "inverse ETF",
}


def render_assessment(assessment: RibaAssessment) -> str:
    """Format one assessment for ops display.

    No-secret-leak: shows only instrument id + class + riba type
    labels. Per-trade financing schedules / counterparty terms
    live in the operator-side DB.
    """

    if assessment.is_clean:
        emoji = "✅"
        suffix = "no riba detected"
    else:
        emoji = "❌"
        labels = sorted(_RIBA_TYPE_LABEL[t] for t in assessment.riba_types)
        suffix = f"riba: {', '.join(labels)}"
    class_label = _CLASS_LABEL[assessment.instrument_class]
    return f"{emoji} {assessment.instrument_id} ({class_label}): {suffix}"


__all__ = [
    "HALAL_BY_DEFAULT_CLASSES",
    "InstrumentClass",
    "RibaAssessment",
    "RibaInputs",
    "RibaPolicy",
    "RibaType",
    "assess_batch",
    "assess_riba",
    "filter_blocked",
    "render_assessment",
]
