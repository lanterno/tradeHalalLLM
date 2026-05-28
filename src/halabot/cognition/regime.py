"""Deterministic regime classifier (REARCHITECTURE L2).

Derives a market regime from the current evidence vector — no LLM, no network
(INV-1). A stand-in for the richer ``ml/regime_memory`` + ``RegimeDetector``
that fold in later; this gets the belief loop running with a sensible regime.
"""

from __future__ import annotations

from halabot.belief.evidence import fraction_same_sign, has_flag, weighted_sum
from halabot.belief.schema import EvidenceItem, Regime

# net |signed| at/above which we call it a trend (vs ranging)
_TREND_THRESHOLD = 0.25


class EvidenceRegimeClassifier:
    """Regime from net signed evidence, agreement, and volatility flags."""

    def classify(self, evidence: list[EvidenceItem]) -> tuple[Regime, float]:
        if not evidence:
            return Regime.RANGING, 0.0
        # An anomaly/drift flag dominates: the tape is unstable → VOLATILE.
        if has_flag(evidence, "anomaly") or has_flag(evidence, "drift"):
            return Regime.VOLATILE, 0.5
        signed = weighted_sum(evidence)
        agreement = fraction_same_sign(evidence)
        strength = abs(signed)
        confidence = max(0.0, min(1.0, strength * (0.5 + 0.5 * agreement)))
        if strength < _TREND_THRESHOLD:
            return Regime.RANGING, max(0.1, 1.0 - strength)  # confident it's NOT trending
        if signed > 0:
            return Regime.TRENDING_UP, confidence
        return Regime.TRENDING_DOWN, confidence
