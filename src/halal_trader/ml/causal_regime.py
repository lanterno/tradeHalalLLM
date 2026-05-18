"""Causal regime detector.

The roadmap calls for replacing the rule-based regime detector
(`crypto/regime.py`) with a *causal* model: given the current macro
state (rates, VIX, dollar index, sector breadth), which trading
regime are we in? Wave 4.D landed a correlation-based fusion engine
(`core/cross_asset_signal.py`); this wave ships the **causal**
counterpart — a discrete Bayesian network that supports
**do-calculus interventions** ("what if I intervened to set VIX=40?")
distinct from observations ("VIX is currently 40").

The distinction matters operationally: an observation `vix=CRISIS`
gets propagated through the graph via Bayes' rule (and updates
upstream beliefs about rates_change because rates_change → vix);
an intervention `do(vix=CRISIS)` does NOT update upstream beliefs
because we *forced* vix to that state regardless of its parents.
The intervention is the right primitive for "stress test" /
"what-if" scenarios; the observation is the right primitive for
inference from real-time data.

Picked discrete Bayesian network with brute-force enumeration over
sample-based methods because (a) the graph has 5-6 variables with
3 states each (3^5 = 243 cells) so brute force is fast and exact,
(b) operators can read the CPTs and debug a misfire at the source,
(c) sample-based methods (Gibbs, importance) require burn-in and
convergence checks which are operationally tedious for a per-cycle
hot path. The full pgmpy-style symbolic engine would be over-kill;
we ship the small focused engine the bot actually needs.

Pinned semantics:
- **Closed variable + state set.** `MacroVariable` lists the five
  inputs + the regime output; each has a closed `*State` enum.
  Adding a variable / state is a code-review change.
- **CPTs module-level frozen.** Runtime config drift can't change
  the conditional probabilities — operators tune via code review
  with regression-tested before/after distributions.
- **Intervention overrides observation.** When both are provided
  for the same variable, intervention wins — pinned via test.
- **Partial evidence supported.** Operator can supply observations
  for any subset of variables; the engine marginalises out the
  rest. Empty evidence returns the prior P(regime).
- **Probability distributions sum to 1.0 within tolerance.**
  Inference outputs are renormalised at the end so float drift
  doesn't accumulate.
- **Render output never includes raw probability values for
  individual cells.** Shows the regime marginal as percentages —
  operator audits the CPTs in the source if they want raw numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RatesLevel(str, Enum):
    """Federal funds rate level state."""

    LOW = "low"  # < 2%
    NORMAL = "normal"  # 2-5%
    HIGH = "high"  # > 5%


class RatesChange(str, Enum):
    """Rate-of-change in monetary policy."""

    EASING = "easing"  # cuts in last 6 months
    HOLDING = "holding"  # no change
    TIGHTENING = "tightening"  # hikes in last 6 months


class VIXState(str, Enum):
    """VIX volatility regime."""

    CALM = "calm"  # < 15
    ELEVATED = "elevated"  # 15-25
    CRISIS = "crisis"  # > 25


class DXYState(str, Enum):
    """Dollar-index regime."""

    WEAK = "weak"  # falling DXY
    NORMAL = "normal"  # range-bound
    STRONG = "strong"  # rising DXY


class BreadthState(str, Enum):
    """Sector-breadth regime (% sectors above 50-day MA)."""

    NEGATIVE = "negative"  # < 30%
    MIXED = "mixed"  # 30-70%
    POSITIVE = "positive"  # > 70%


class Regime(str, Enum):
    """Trading regime — the inference target."""

    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"
    CRISIS = "crisis"


# Module-level frozen CPTs.
# P(rates_level) — prior
_PRIOR_RATES_LEVEL: dict[RatesLevel, float] = {
    RatesLevel.LOW: 0.30,
    RatesLevel.NORMAL: 0.50,
    RatesLevel.HIGH: 0.20,
}

# P(rates_change) — prior
_PRIOR_RATES_CHANGE: dict[RatesChange, float] = {
    RatesChange.EASING: 0.30,
    RatesChange.HOLDING: 0.40,
    RatesChange.TIGHTENING: 0.30,
}

# P(vix | rates_change)
# Tightening cycles → more crisis risk; easing → calm
_CPT_VIX_GIVEN_RATES_CHANGE: dict[RatesChange, dict[VIXState, float]] = {
    RatesChange.EASING: {
        VIXState.CALM: 0.50,
        VIXState.ELEVATED: 0.35,
        VIXState.CRISIS: 0.15,
    },
    RatesChange.HOLDING: {
        VIXState.CALM: 0.45,
        VIXState.ELEVATED: 0.40,
        VIXState.CRISIS: 0.15,
    },
    RatesChange.TIGHTENING: {
        VIXState.CALM: 0.20,
        VIXState.ELEVATED: 0.45,
        VIXState.CRISIS: 0.35,
    },
}

# P(dxy | rates_level)
# Higher rates → stronger dollar
_CPT_DXY_GIVEN_RATES_LEVEL: dict[RatesLevel, dict[DXYState, float]] = {
    RatesLevel.LOW: {
        DXYState.WEAK: 0.50,
        DXYState.NORMAL: 0.40,
        DXYState.STRONG: 0.10,
    },
    RatesLevel.NORMAL: {
        DXYState.WEAK: 0.25,
        DXYState.NORMAL: 0.50,
        DXYState.STRONG: 0.25,
    },
    RatesLevel.HIGH: {
        DXYState.WEAK: 0.10,
        DXYState.NORMAL: 0.40,
        DXYState.STRONG: 0.50,
    },
}

# P(breadth | vix)
# High VIX → negative breadth
_CPT_BREADTH_GIVEN_VIX: dict[VIXState, dict[BreadthState, float]] = {
    VIXState.CALM: {
        BreadthState.NEGATIVE: 0.10,
        BreadthState.MIXED: 0.40,
        BreadthState.POSITIVE: 0.50,
    },
    VIXState.ELEVATED: {
        BreadthState.NEGATIVE: 0.30,
        BreadthState.MIXED: 0.50,
        BreadthState.POSITIVE: 0.20,
    },
    VIXState.CRISIS: {
        BreadthState.NEGATIVE: 0.65,
        BreadthState.MIXED: 0.30,
        BreadthState.POSITIVE: 0.05,
    },
}

# P(regime | vix, dxy, breadth)
# Hand-tuned to reflect operator's macro mental model:
# - CRISIS: VIX=CRISIS + STRONG-DXY (flight to safety) + NEGATIVE breadth
# - RISK_OFF: ELEVATED VIX + STRONG / NORMAL DXY + MIXED / NEGATIVE breadth
# - NEUTRAL: middle states
# - RISK_ON: CALM VIX + WEAK / NORMAL DXY + POSITIVE breadth
_CPT_REGIME_GIVEN_VIX_DXY_BREADTH: dict[
    tuple[VIXState, DXYState, BreadthState], dict[Regime, float]
] = {
    # CALM VIX
    (VIXState.CALM, DXYState.WEAK, BreadthState.POSITIVE): {
        Regime.RISK_ON: 0.85,
        Regime.NEUTRAL: 0.13,
        Regime.RISK_OFF: 0.02,
        Regime.CRISIS: 0.00,
    },
    (VIXState.CALM, DXYState.WEAK, BreadthState.MIXED): {
        Regime.RISK_ON: 0.50,
        Regime.NEUTRAL: 0.40,
        Regime.RISK_OFF: 0.09,
        Regime.CRISIS: 0.01,
    },
    (VIXState.CALM, DXYState.WEAK, BreadthState.NEGATIVE): {
        Regime.RISK_ON: 0.15,
        Regime.NEUTRAL: 0.55,
        Regime.RISK_OFF: 0.27,
        Regime.CRISIS: 0.03,
    },
    (VIXState.CALM, DXYState.NORMAL, BreadthState.POSITIVE): {
        Regime.RISK_ON: 0.70,
        Regime.NEUTRAL: 0.25,
        Regime.RISK_OFF: 0.05,
        Regime.CRISIS: 0.00,
    },
    (VIXState.CALM, DXYState.NORMAL, BreadthState.MIXED): {
        Regime.RISK_ON: 0.35,
        Regime.NEUTRAL: 0.50,
        Regime.RISK_OFF: 0.14,
        Regime.CRISIS: 0.01,
    },
    (VIXState.CALM, DXYState.NORMAL, BreadthState.NEGATIVE): {
        Regime.RISK_ON: 0.10,
        Regime.NEUTRAL: 0.50,
        Regime.RISK_OFF: 0.35,
        Regime.CRISIS: 0.05,
    },
    (VIXState.CALM, DXYState.STRONG, BreadthState.POSITIVE): {
        Regime.RISK_ON: 0.45,
        Regime.NEUTRAL: 0.45,
        Regime.RISK_OFF: 0.09,
        Regime.CRISIS: 0.01,
    },
    (VIXState.CALM, DXYState.STRONG, BreadthState.MIXED): {
        Regime.RISK_ON: 0.20,
        Regime.NEUTRAL: 0.55,
        Regime.RISK_OFF: 0.23,
        Regime.CRISIS: 0.02,
    },
    (VIXState.CALM, DXYState.STRONG, BreadthState.NEGATIVE): {
        Regime.RISK_ON: 0.05,
        Regime.NEUTRAL: 0.40,
        Regime.RISK_OFF: 0.50,
        Regime.CRISIS: 0.05,
    },
    # ELEVATED VIX
    (VIXState.ELEVATED, DXYState.WEAK, BreadthState.POSITIVE): {
        Regime.RISK_ON: 0.45,
        Regime.NEUTRAL: 0.45,
        Regime.RISK_OFF: 0.09,
        Regime.CRISIS: 0.01,
    },
    (VIXState.ELEVATED, DXYState.WEAK, BreadthState.MIXED): {
        Regime.RISK_ON: 0.25,
        Regime.NEUTRAL: 0.50,
        Regime.RISK_OFF: 0.22,
        Regime.CRISIS: 0.03,
    },
    (VIXState.ELEVATED, DXYState.WEAK, BreadthState.NEGATIVE): {
        Regime.RISK_ON: 0.05,
        Regime.NEUTRAL: 0.30,
        Regime.RISK_OFF: 0.55,
        Regime.CRISIS: 0.10,
    },
    (VIXState.ELEVATED, DXYState.NORMAL, BreadthState.POSITIVE): {
        Regime.RISK_ON: 0.35,
        Regime.NEUTRAL: 0.50,
        Regime.RISK_OFF: 0.14,
        Regime.CRISIS: 0.01,
    },
    (VIXState.ELEVATED, DXYState.NORMAL, BreadthState.MIXED): {
        Regime.RISK_ON: 0.15,
        Regime.NEUTRAL: 0.50,
        Regime.RISK_OFF: 0.30,
        Regime.CRISIS: 0.05,
    },
    (VIXState.ELEVATED, DXYState.NORMAL, BreadthState.NEGATIVE): {
        Regime.RISK_ON: 0.03,
        Regime.NEUTRAL: 0.20,
        Regime.RISK_OFF: 0.62,
        Regime.CRISIS: 0.15,
    },
    (VIXState.ELEVATED, DXYState.STRONG, BreadthState.POSITIVE): {
        Regime.RISK_ON: 0.20,
        Regime.NEUTRAL: 0.55,
        Regime.RISK_OFF: 0.23,
        Regime.CRISIS: 0.02,
    },
    (VIXState.ELEVATED, DXYState.STRONG, BreadthState.MIXED): {
        Regime.RISK_ON: 0.07,
        Regime.NEUTRAL: 0.40,
        Regime.RISK_OFF: 0.45,
        Regime.CRISIS: 0.08,
    },
    (VIXState.ELEVATED, DXYState.STRONG, BreadthState.NEGATIVE): {
        Regime.RISK_ON: 0.02,
        Regime.NEUTRAL: 0.15,
        Regime.RISK_OFF: 0.58,
        Regime.CRISIS: 0.25,
    },
    # CRISIS VIX
    (VIXState.CRISIS, DXYState.WEAK, BreadthState.POSITIVE): {
        Regime.RISK_ON: 0.10,
        Regime.NEUTRAL: 0.30,
        Regime.RISK_OFF: 0.45,
        Regime.CRISIS: 0.15,
    },
    (VIXState.CRISIS, DXYState.WEAK, BreadthState.MIXED): {
        Regime.RISK_ON: 0.05,
        Regime.NEUTRAL: 0.20,
        Regime.RISK_OFF: 0.55,
        Regime.CRISIS: 0.20,
    },
    (VIXState.CRISIS, DXYState.WEAK, BreadthState.NEGATIVE): {
        Regime.RISK_ON: 0.02,
        Regime.NEUTRAL: 0.10,
        Regime.RISK_OFF: 0.58,
        Regime.CRISIS: 0.30,
    },
    (VIXState.CRISIS, DXYState.NORMAL, BreadthState.POSITIVE): {
        Regime.RISK_ON: 0.07,
        Regime.NEUTRAL: 0.25,
        Regime.RISK_OFF: 0.50,
        Regime.CRISIS: 0.18,
    },
    (VIXState.CRISIS, DXYState.NORMAL, BreadthState.MIXED): {
        Regime.RISK_ON: 0.03,
        Regime.NEUTRAL: 0.15,
        Regime.RISK_OFF: 0.55,
        Regime.CRISIS: 0.27,
    },
    (VIXState.CRISIS, DXYState.NORMAL, BreadthState.NEGATIVE): {
        Regime.RISK_ON: 0.01,
        Regime.NEUTRAL: 0.05,
        Regime.RISK_OFF: 0.54,
        Regime.CRISIS: 0.40,
    },
    (VIXState.CRISIS, DXYState.STRONG, BreadthState.POSITIVE): {
        Regime.RISK_ON: 0.03,
        Regime.NEUTRAL: 0.20,
        Regime.RISK_OFF: 0.55,
        Regime.CRISIS: 0.22,
    },
    (VIXState.CRISIS, DXYState.STRONG, BreadthState.MIXED): {
        Regime.RISK_ON: 0.01,
        Regime.NEUTRAL: 0.10,
        Regime.RISK_OFF: 0.55,
        Regime.CRISIS: 0.34,
    },
    (VIXState.CRISIS, DXYState.STRONG, BreadthState.NEGATIVE): {
        Regime.RISK_ON: 0.00,
        Regime.NEUTRAL: 0.04,
        Regime.RISK_OFF: 0.46,
        Regime.CRISIS: 0.50,
    },
}


@dataclass(frozen=True)
class MacroEvidence:
    """Operator's observed macro state.

    Each field is optional — missing observations are marginalised
    out via the brute-force inference. Empty MacroEvidence returns
    the prior P(regime).
    """

    rates_level: RatesLevel | None = None
    rates_change: RatesChange | None = None
    vix: VIXState | None = None
    dxy: DXYState | None = None
    breadth: BreadthState | None = None


@dataclass(frozen=True)
class MacroIntervention:
    """Do-calculus intervention.

    When set, the engine treats the variable as if forced to the
    given state regardless of its parents — does NOT propagate
    upstream like an observation would.
    """

    rates_level: RatesLevel | None = None
    rates_change: RatesChange | None = None
    vix: VIXState | None = None
    dxy: DXYState | None = None
    breadth: BreadthState | None = None


@dataclass(frozen=True)
class RegimeInferenceResult:
    """The inferred regime distribution."""

    distribution: dict[Regime, float]
    most_likely: Regime
    confidence: float  # P(most-likely regime)
    evidence_used: MacroEvidence
    intervention_used: MacroIntervention | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _joint_probability(
    rates_level: RatesLevel,
    rates_change: RatesChange,
    vix: VIXState,
    dxy: DXYState,
    breadth: BreadthState,
    regime: Regime,
    *,
    intervention: MacroIntervention | None,
) -> float:
    """Compute the joint probability of one full state assignment.

    Under intervention, the intervened variable's CPT is replaced
    by a delta function (probability 1 for the forced state, 0
    otherwise) — implemented by skipping the variable's normal CPT
    contribution and including the joint only when the value
    matches the forced state.
    """

    # Apply intervention to the relevant variable's CPT contribution.
    p = 1.0

    # rates_level: prior unless intervened
    if intervention and intervention.rates_level is not None:
        if rates_level is not intervention.rates_level:
            return 0.0
        # forced — no CPT contribution, multiplier is 1
    else:
        p *= _PRIOR_RATES_LEVEL[rates_level]

    # rates_change: prior unless intervened
    if intervention and intervention.rates_change is not None:
        if rates_change is not intervention.rates_change:
            return 0.0
    else:
        p *= _PRIOR_RATES_CHANGE[rates_change]

    # vix: depends on rates_change (if intervened, edge from rates_change is severed)
    if intervention and intervention.vix is not None:
        if vix is not intervention.vix:
            return 0.0
    else:
        p *= _CPT_VIX_GIVEN_RATES_CHANGE[rates_change][vix]

    # dxy: depends on rates_level (if intervened, edge from rates_level is severed)
    if intervention and intervention.dxy is not None:
        if dxy is not intervention.dxy:
            return 0.0
    else:
        p *= _CPT_DXY_GIVEN_RATES_LEVEL[rates_level][dxy]

    # breadth: depends on vix (if intervened, edge from vix is severed)
    if intervention and intervention.breadth is not None:
        if breadth is not intervention.breadth:
            return 0.0
    else:
        p *= _CPT_BREADTH_GIVEN_VIX[vix][breadth]

    # regime: depends on (vix, dxy, breadth) — never intervened (it's the target)
    p *= _CPT_REGIME_GIVEN_VIX_DXY_BREADTH[(vix, dxy, breadth)][regime]

    return p


def _matches_evidence(
    rates_level: RatesLevel,
    rates_change: RatesChange,
    vix: VIXState,
    dxy: DXYState,
    breadth: BreadthState,
    *,
    evidence: MacroEvidence,
) -> bool:
    """True if the assignment is consistent with the evidence."""

    if evidence.rates_level is not None and rates_level is not evidence.rates_level:
        return False
    if evidence.rates_change is not None and rates_change is not evidence.rates_change:
        return False
    if evidence.vix is not None and vix is not evidence.vix:
        return False
    if evidence.dxy is not None and dxy is not evidence.dxy:
        return False
    if evidence.breadth is not None and breadth is not evidence.breadth:
        return False
    return True


def infer_regime(
    *,
    evidence: MacroEvidence = MacroEvidence(),
    intervention: MacroIntervention | None = None,
) -> RegimeInferenceResult:
    """Compute P(regime | evidence, do(intervention)).

    The intervention takes precedence over evidence for the same
    variable; the engine surfaces a warning when both are
    specified for the same field. Empty evidence returns the prior
    P(regime).
    """

    warnings: list[str] = []

    if intervention is not None:
        for field_name in ("rates_level", "rates_change", "vix", "dxy", "breadth"):
            ev_val = getattr(evidence, field_name)
            iv_val = getattr(intervention, field_name)
            if ev_val is not None and iv_val is not None:
                warnings.append(
                    f"both observation and intervention provided for "
                    f"{field_name!r}: intervention wins"
                )

    # Brute-force enumeration over the 3^5 joint state space.
    regime_marginal: dict[Regime, float] = {r: 0.0 for r in Regime}

    for rates_level in RatesLevel:
        for rates_change in RatesChange:
            for vix in VIXState:
                for dxy in DXYState:
                    for breadth in BreadthState:
                        # Evidence is observation; intervention overrides for that field.
                        # Build the effective evidence for matching:
                        # under intervention, the intervened field's evidence is dropped
                        # (we don't filter by it; the intervention's delta in the joint
                        # already restricts the assignment).
                        if not _matches_evidence(
                            rates_level=rates_level,
                            rates_change=rates_change,
                            vix=vix,
                            dxy=dxy,
                            breadth=breadth,
                            evidence=_evidence_minus_intervention(evidence, intervention),
                        ):
                            continue
                        for regime in Regime:
                            p = _joint_probability(
                                rates_level=rates_level,
                                rates_change=rates_change,
                                vix=vix,
                                dxy=dxy,
                                breadth=breadth,
                                regime=regime,
                                intervention=intervention,
                            )
                            regime_marginal[regime] += p

    # Normalise.
    total = sum(regime_marginal.values())
    if total <= 0:
        # Should not happen with valid CPTs + non-impossible evidence, but guard.
        warnings.append(
            "joint probability summed to zero — evidence + intervention may be inconsistent"
        )
        # fallback to uniform
        regime_marginal = {r: 1.0 / len(Regime) for r in Regime}
    else:
        regime_marginal = {r: p / total for r, p in regime_marginal.items()}

    most_likely = max(regime_marginal, key=lambda r: regime_marginal[r])
    confidence = regime_marginal[most_likely]

    return RegimeInferenceResult(
        distribution=regime_marginal,
        most_likely=most_likely,
        confidence=confidence,
        evidence_used=evidence,
        intervention_used=intervention,
        warnings=tuple(warnings),
    )


def _evidence_minus_intervention(
    evidence: MacroEvidence, intervention: MacroIntervention | None
) -> MacroEvidence:
    """Return a copy of evidence with intervened fields cleared.

    Intervention semantically overrides observation; we strip the
    overridden field from evidence so the matching pass doesn't
    incorrectly filter on the observed (pre-intervention) value.
    """

    if intervention is None:
        return evidence
    return MacroEvidence(
        rates_level=None if intervention.rates_level is not None else evidence.rates_level,
        rates_change=None if intervention.rates_change is not None else evidence.rates_change,
        vix=None if intervention.vix is not None else evidence.vix,
        dxy=None if intervention.dxy is not None else evidence.dxy,
        breadth=None if intervention.breadth is not None else evidence.breadth,
    )


_REGIME_EMOJI: dict[Regime, str] = {
    Regime.RISK_ON: "🟢",
    Regime.NEUTRAL: "⚪",
    Regime.RISK_OFF: "🟠",
    Regime.CRISIS: "🔴",
}


def render_inference(result: RegimeInferenceResult) -> str:
    """Format the regime inference for ops display.

    Pinned no-raw-CPT contract: shows the regime marginal as
    percentages plus the most-likely + confidence; doesn't dump
    the full CPT cells. Operators audit the source for those.
    """

    emoji = _REGIME_EMOJI[result.most_likely]
    lines = [
        f"{emoji} most likely: {result.most_likely.value} "
        f"({result.confidence * 100:.1f}% confidence)",
        "  distribution:",
    ]
    for regime in Regime:
        prob = result.distribution[regime]
        bar = "█" * int(prob * 20)
        lines.append(f"    {regime.value:9s} {prob * 100:5.1f}%  {bar}")
    if result.intervention_used is not None:
        non_none = [
            f"{k}={getattr(result.intervention_used, k).value}"
            for k in ("rates_level", "rates_change", "vix", "dxy", "breadth")
            if getattr(result.intervention_used, k) is not None
        ]
        if non_none:
            lines.append(f"  intervention: do({', '.join(non_none)})")
    if result.warnings:
        lines.append("  warnings:")
        for w in result.warnings:
            lines.append(f"    · {w}")
    return "\n".join(lines)


__all__ = [
    "BreadthState",
    "DXYState",
    "MacroEvidence",
    "MacroIntervention",
    "RatesChange",
    "RatesLevel",
    "Regime",
    "RegimeInferenceResult",
    "VIXState",
    "infer_regime",
    "render_inference",
]
