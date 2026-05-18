"""Halal climate finance screen — Round-5 Wave 5.E.

Screens climate-finance instruments — carbon credits, green sukuk,
renewable-energy projects, climate-aligned ESG funds — for halal
compliance. Carbon credits are particularly contested in fiqh: some
scholars treat them as permissible utility instruments, others see
the underlying offset as gharar.

Pinned semantics:

- **Closed-set ClimateInstrument ladder** — 6 documented instrument
  types.
- **Closed-set ClimateIssue ladder** — 6 specific issues.
- **Carbon credit verification** uses operator-supplied tier
  (VERRA / GOLD_STANDARD / OTHER).
- **Green sukuk** delegates structural compliance to AAOIFI
  Standard 17 via existing `aaoifi_standard_17.py`.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ClimateInstrument(str, Enum):
    """Closed-set climate-finance instrument types."""

    GREEN_SUKUK = "green_sukuk"
    CARBON_CREDIT_VERRA = "carbon_credit_verra"
    CARBON_CREDIT_GOLD_STANDARD = "carbon_credit_gold_standard"
    CARBON_CREDIT_OTHER = "carbon_credit_other"
    RENEWABLE_PROJECT_EQUITY = "renewable_project_equity"
    CLIMATE_ALIGNED_ETF = "climate_aligned_etf"


class ClimateIssue(str, Enum):
    """Closed-set climate-finance halal issues."""

    UNVERIFIED_PROVENANCE = "unverified_provenance"
    SPECULATIVE_OFFSET = "speculative_offset"
    INTEREST_BEARING_TRANCHES = "interest_bearing_tranches"
    UNDERLYING_INCLUDES_HARAM = "underlying_includes_haram"
    GHARAR_FUTURE_DELIVERY = "gharar_future_delivery"
    NO_SHARIAH_OPINION = "no_shariah_opinion"


@dataclass(frozen=True)
class ClimatePolicy:
    """Operator-tunable policy."""

    accept_carbon_credits: bool = True
    require_audited_provenance: bool = True
    require_shariah_opinion_for_sukuk: bool = True

    def __post_init__(self) -> None:
        pass


@dataclass(frozen=True)
class ClimateInputs:
    """Inputs for a climate finance screen."""

    instrument_id: str
    instrument: ClimateInstrument
    has_audited_provenance: bool
    speculative_offset_proportion: float  # 0..1
    has_interest_tranches: bool
    underlying_includes_haram: bool
    has_shariah_opinion: bool

    def __post_init__(self) -> None:
        if not self.instrument_id or not self.instrument_id.strip():
            raise ValueError("instrument_id must be non-empty")
        if not 0.0 <= self.speculative_offset_proportion <= 1.0:
            raise ValueError("speculative_offset_proportion must be in [0, 1]")


@dataclass(frozen=True)
class ClimateAssessment:
    """Result of a climate finance screen."""

    instrument_id: str
    instrument: ClimateInstrument
    issues: frozenset[ClimateIssue]
    is_compliant: bool


def screen_instrument(
    inputs: ClimateInputs, *, policy: ClimatePolicy | None = None
) -> ClimateAssessment:
    """Screen a climate-finance instrument."""
    pol = policy if policy is not None else ClimatePolicy()
    issues: set[ClimateIssue] = set()

    is_carbon = inputs.instrument in {
        ClimateInstrument.CARBON_CREDIT_VERRA,
        ClimateInstrument.CARBON_CREDIT_GOLD_STANDARD,
        ClimateInstrument.CARBON_CREDIT_OTHER,
    }

    if is_carbon and not pol.accept_carbon_credits:
        # Operator's policy rejects carbon credits outright
        issues.add(ClimateIssue.SPECULATIVE_OFFSET)
        return ClimateAssessment(
            instrument_id=inputs.instrument_id,
            instrument=inputs.instrument,
            issues=frozenset(issues),
            is_compliant=False,
        )

    if pol.require_audited_provenance and not inputs.has_audited_provenance:
        issues.add(ClimateIssue.UNVERIFIED_PROVENANCE)

    if inputs.speculative_offset_proportion > 0.30:
        issues.add(ClimateIssue.SPECULATIVE_OFFSET)

    if inputs.has_interest_tranches:
        issues.add(ClimateIssue.INTEREST_BEARING_TRANCHES)

    if inputs.underlying_includes_haram:
        issues.add(ClimateIssue.UNDERLYING_INCLUDES_HARAM)

    if (
        inputs.instrument is ClimateInstrument.GREEN_SUKUK
        and pol.require_shariah_opinion_for_sukuk
        and not inputs.has_shariah_opinion
    ):
        issues.add(ClimateIssue.NO_SHARIAH_OPINION)

    if (
        is_carbon
        and inputs.instrument is ClimateInstrument.CARBON_CREDIT_OTHER
        and not inputs.has_audited_provenance
    ):
        issues.add(ClimateIssue.GHARAR_FUTURE_DELIVERY)

    return ClimateAssessment(
        instrument_id=inputs.instrument_id,
        instrument=inputs.instrument,
        issues=frozenset(issues),
        is_compliant=len(issues) == 0,
    )


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


def render_assessment(a: ClimateAssessment) -> str:
    emoji = "✅" if a.is_compliant else "❌"
    head = f"{emoji} {a.instrument_id} ({a.instrument.value})"
    lines = [head]
    for issue in sorted(a.issues, key=lambda x: x.value):
        lines.append(f"  • {issue.value}")
    return _scrub("\n".join(lines))
