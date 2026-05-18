"""Gharar (excessive uncertainty) detector.

Round-5 Wave 1.E primitive. Standard halal screening checks debt
ratios + revenue purity + sector. The maysir screen
(`halal/maysir_screen.py`) catches gambling-pattern equities. This
module catches the third independently-prohibited Shariah issue:
gharar — excessive uncertainty in the contract or instrument
itself. Examples: a structured note where you can't see the
underlying basket; an asset-backed security where the asset
quality is opaque; a forward-sale where the delivery date keeps
slipping; dual-class shares where one class has voting rights
the other doesn't.

Where maysir catches "this stock trades like gambling" (a
behavioral / market-structure signal), gharar catches "this
contract has hidden uncertainty" (a structural / disclosure
signal). The two compose: a name passes the standard halal screen
AND the maysir screen AND the gharar detector → tradable.

Picked a closed-set signal catalogue + structural boolean inputs
because (a) gharar signals are mostly disclosure-quality flags,
not continuous metrics — a counterparty is either disclosed or
not, an underlying basket is either transparent or opaque; (b)
the catalogue is documentation that scholars + operators read
+ AAOIFI Standard 21 references; (c) operators across schools
agree more on gharar than on subtler issues (Hanafi / Shafi'i /
Maliki / Hanbali all classify undisclosed-underlying as severe
gharar) — the engine is policy-light because the underlying fiqh
is well-settled.

Pinned semantics:
- **Closed-set GhararLevel ladder.** NONE < MINOR < MODERATE <
  SEVERE. SEVERE is non-tradable by default.
- **Closed-set GhararSignal catalogue.** Adding a signal is a
  code review change.
- **Each signal has documented severity weight.** Most-severe
  signals (UNDISCLOSED_UNDERLYING, COUNTERPARTY_UNDISCLOSED) are
  weight 3 because either alone is enough to make a contract
  void per AAOIFI Standard 21. Lighter signals (DUAL_CLASS_
  UNEQUAL_RIGHTS, OPAQUE_FEE_STRUCTURE) are weight 1.
- **`is_tradable(assessment)` is the load-bearing gate.** Returns
  True for NONE / MINOR / MODERATE; False for SEVERE.
- **Render output never includes the underlying instrument's
  prospectus / disclosure-document text.** Only the signals +
  level + brief reasoning; the full disclosure document lives in
  the operator-side compliance archive.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class GhararLevel(str, Enum):
    """Gharar (excessive uncertainty) risk ladder.

    Pinned string values for JSON / DB persistence stability.
    NONE < MINOR < MODERATE < SEVERE. SEVERE is non-tradable.
    """

    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class GhararSignal(str, Enum):
    """Catalogue of gharar-pattern detector signals.

    Pinned string values. Adding a signal is a code review change.
    """

    UNDISCLOSED_UNDERLYING = "undisclosed_underlying"
    COUNTERPARTY_UNDISCLOSED = "counterparty_undisclosed"
    ASSET_BACKING_OPAQUE = "asset_backing_opaque"
    CONTINGENT_PAYOFF = "contingent_payoff"
    FUTURE_DELIVERY_INDETERMINATE = "future_delivery_indeterminate"
    NESTED_DERIVATIVE_LAYERS = "nested_derivative_layers"
    DUAL_CLASS_UNEQUAL_RIGHTS = "dual_class_unequal_rights"
    OPAQUE_FEE_STRUCTURE = "opaque_fee_structure"


# Weight-3 signals are the AAOIFI Standard 21 "void contract"
# triggers — either alone is enough to make a contract impermissible.
# Weight-2 signals are "significant gharar" — concerning but
# potentially mitigable. Weight-1 signals are "minor" — operator-
# tunable cumulative concerns.
_SIGNAL_WEIGHT: dict[GhararSignal, int] = {
    GhararSignal.UNDISCLOSED_UNDERLYING: 3,  # Don't know what you're buying
    GhararSignal.COUNTERPARTY_UNDISCLOSED: 3,  # Don't know who you're trading with
    GhararSignal.ASSET_BACKING_OPAQUE: 2,
    GhararSignal.CONTINGENT_PAYOFF: 2,
    GhararSignal.FUTURE_DELIVERY_INDETERMINATE: 2,
    GhararSignal.NESTED_DERIVATIVE_LAYERS: 2,
    GhararSignal.DUAL_CLASS_UNEQUAL_RIGHTS: 1,
    GhararSignal.OPAQUE_FEE_STRUCTURE: 1,
}


@dataclass(frozen=True)
class GhararPolicy:
    """Operator-tunable gharar-detector policy.

    Defaults align with AAOIFI Standard 21: any single weight-3
    signal pushes to MODERATE+; two or more push to SEVERE. The
    nested_layers_threshold reflects the documented scholar
    position that 2+ wrappers around a single underlying obscures
    economic substance.
    """

    nested_layers_threshold: int = 2  # >= this many nested layers fires signal
    minor_score_threshold: int = 1  # score >= 1 → MINOR
    moderate_score_threshold: int = 3  # score >= 3 → MODERATE
    severe_score_threshold: int = 5  # score >= 5 → SEVERE

    def __post_init__(self) -> None:
        if self.nested_layers_threshold < 2:
            raise ValueError("nested_layers_threshold must be >= 2")
        if not (
            self.minor_score_threshold < self.moderate_score_threshold < self.severe_score_threshold
        ):
            raise ValueError("score thresholds must satisfy minor < moderate < severe")
        if self.minor_score_threshold < 1:
            raise ValueError("minor_score_threshold must be >= 1")


@dataclass(frozen=True)
class GhararInputs:
    """Per-instrument inputs for the gharar detector.

    All flags default to the "clean" / "transparent" position so
    operators only set the ones that flag concerns. Booleans
    invert per signal — `underlying_disclosed=False` fires
    UNDISCLOSED_UNDERLYING, etc.
    """

    instrument_id: str
    underlying_disclosed: bool = True
    counterparty_disclosed: bool = True
    asset_backing_transparent: bool = True
    payoff_contingent_on_event: bool = False
    delivery_date_specified: bool = True
    derivative_layers: int = 0
    has_dual_class_unequal_rights: bool = False
    fees_fully_disclosed: bool = True

    def __post_init__(self) -> None:
        if not self.instrument_id or not self.instrument_id.strip():
            raise ValueError("instrument_id must be non-empty")
        if self.derivative_layers < 0:
            raise ValueError("derivative_layers must be >= 0")


@dataclass(frozen=True)
class GhararAssessment:
    """Output of the gharar detector for one instrument."""

    instrument_id: str
    signals: frozenset[GhararSignal]
    level: GhararLevel
    score: int

    def __post_init__(self) -> None:
        if not self.instrument_id or not self.instrument_id.strip():
            raise ValueError("instrument_id must be non-empty")
        if self.score < 0:
            raise ValueError("score must be non-negative")
        if self.level is GhararLevel.NONE and self.signals:
            raise ValueError("NONE level must have empty signals")
        if self.level is not GhararLevel.NONE and not self.signals:
            raise ValueError("non-NONE level requires at least one signal")


def _detect_signals(
    inputs: GhararInputs,
    *,
    policy: GhararPolicy,
) -> frozenset[GhararSignal]:
    """Run each gharar detector against the inputs."""

    signals: set[GhararSignal] = set()
    if not inputs.underlying_disclosed:
        signals.add(GhararSignal.UNDISCLOSED_UNDERLYING)
    if not inputs.counterparty_disclosed:
        signals.add(GhararSignal.COUNTERPARTY_UNDISCLOSED)
    if not inputs.asset_backing_transparent:
        signals.add(GhararSignal.ASSET_BACKING_OPAQUE)
    if inputs.payoff_contingent_on_event:
        signals.add(GhararSignal.CONTINGENT_PAYOFF)
    if not inputs.delivery_date_specified:
        signals.add(GhararSignal.FUTURE_DELIVERY_INDETERMINATE)
    if inputs.derivative_layers >= policy.nested_layers_threshold:
        signals.add(GhararSignal.NESTED_DERIVATIVE_LAYERS)
    if inputs.has_dual_class_unequal_rights:
        signals.add(GhararSignal.DUAL_CLASS_UNEQUAL_RIGHTS)
    if not inputs.fees_fully_disclosed:
        signals.add(GhararSignal.OPAQUE_FEE_STRUCTURE)
    return frozenset(signals)


def _score_to_level(score: int, *, policy: GhararPolicy) -> GhararLevel:
    """Map fired-signal weight sum to GhararLevel via policy cutoffs."""

    if score == 0:
        return GhararLevel.NONE
    if score >= policy.severe_score_threshold:
        return GhararLevel.SEVERE
    if score >= policy.moderate_score_threshold:
        return GhararLevel.MODERATE
    if score >= policy.minor_score_threshold:
        return GhararLevel.MINOR
    return GhararLevel.NONE


def assess_gharar(
    inputs: GhararInputs,
    *,
    policy: GhararPolicy = GhararPolicy(),
) -> GhararAssessment:
    """Run the gharar detector for one instrument.

    Returns the assessment with fired signals + computed level +
    weighted score. Operators consult `is_tradable(assessment)`
    as the load-bearing gate.
    """

    signals = _detect_signals(inputs, policy=policy)
    score = sum(_SIGNAL_WEIGHT[s] for s in signals)
    level = _score_to_level(score, policy=policy)
    return GhararAssessment(
        instrument_id=inputs.instrument_id,
        signals=signals,
        level=level,
        score=score,
    )


def assess_batch(
    inputs_list: Iterable[GhararInputs],
    *,
    policy: GhararPolicy = GhararPolicy(),
) -> tuple[GhararAssessment, ...]:
    """Run the detector across many instruments; sorted by id.

    Deterministic ordering for the dashboard tile + email summary.
    """

    assessments = [assess_gharar(i, policy=policy) for i in inputs_list]
    assessments.sort(key=lambda a: a.instrument_id)
    return tuple(assessments)


def is_tradable(assessment: GhararAssessment) -> bool:
    """Whether the instrument passes the gharar detector.

    Load-bearing gate. True for NONE / MINOR / MODERATE; False for
    SEVERE. The MODERATE-still-tradable design matches operator
    expectations of "block only the most severe; surface the rest
    to operator with awareness."
    """

    return assessment.level is not GhararLevel.SEVERE


def filter_blocked(
    assessments: Iterable[GhararAssessment],
) -> tuple[GhararAssessment, ...]:
    """Return only the assessments blocked (SEVERE) by the detector."""

    return tuple(a for a in assessments if not is_tradable(a))


_LEVEL_EMOJI: dict[GhararLevel, str] = {
    GhararLevel.NONE: "✅",
    GhararLevel.MINOR: "🟢",
    GhararLevel.MODERATE: "🟡",
    GhararLevel.SEVERE: "🔴",
}


_SIGNAL_LABEL: dict[GhararSignal, str] = {
    GhararSignal.UNDISCLOSED_UNDERLYING: "undisclosed underlying",
    GhararSignal.COUNTERPARTY_UNDISCLOSED: "counterparty undisclosed",
    GhararSignal.ASSET_BACKING_OPAQUE: "asset-backing opaque",
    GhararSignal.CONTINGENT_PAYOFF: "contingent payoff",
    GhararSignal.FUTURE_DELIVERY_INDETERMINATE: "future-delivery indeterminate",
    GhararSignal.NESTED_DERIVATIVE_LAYERS: "nested derivative layers",
    GhararSignal.DUAL_CLASS_UNEQUAL_RIGHTS: "dual-class unequal rights",
    GhararSignal.OPAQUE_FEE_STRUCTURE: "opaque fee structure",
}


def render_assessment(assessment: GhararAssessment) -> str:
    """Format one assessment for ops display.

    No-secret-leak: shows only instrument id + level + signal
    labels. The full prospectus / disclosure-document text lives
    in the operator-side compliance archive.
    """

    emoji = _LEVEL_EMOJI[assessment.level]
    parts = [f"{emoji} {assessment.instrument_id}: {assessment.level.value}"]
    if assessment.signals:
        labels = sorted(_SIGNAL_LABEL[s] for s in assessment.signals)
        parts.append(f"({', '.join(labels)})")
    return " ".join(parts)


__all__ = [
    "GhararAssessment",
    "GhararInputs",
    "GhararLevel",
    "GhararPolicy",
    "GhararSignal",
    "assess_batch",
    "assess_gharar",
    "filter_blocked",
    "is_tradable",
    "render_assessment",
]
