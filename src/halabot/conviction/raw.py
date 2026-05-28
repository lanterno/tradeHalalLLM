"""Raw conviction + the calibrator surface (REARCHITECTURE B.2, L4).

``conviction_raw`` is deterministic and LLM-free (INV-1): conviction keeps
updating even when the LLM is down. It is the product of the net signed
evidence, an agreement (dispersion) factor, and regime confidence, then
down-weighted by anomaly/drift flags — so it is structurally ≤ the net signal,
which is why the policy's entry band is tuned from the observed raw-score
distribution rather than set aspirationally (fix R, cold-start ceiling).
"""

from __future__ import annotations

from typing import Any, Protocol

from halabot.belief.evidence import FLAG_SOURCES, fraction_same_sign, weighted_sum
from halabot.belief.schema import EvidenceItem

_DRIFT_PENALTY = 0.7  # concept drift → widen uncertainty
_ANOMALY_PENALTY = 0.6  # anomalous tape → trust the signal less
# Directional-evidence mass at which conviction saturates. Because the signed
# vector is normalized (scale-invariant), conviction must ALSO depend on the
# absolute mass of *live* evidence — otherwise proportional decay wouldn't fade
# conviction and a stale belief would hold full size forever (R-08). With
# full_mass=2, one unit-weight signal gives ~half conviction (corroboration
# wanted), and decay shrinks mass → conviction de-risks on the passage of time.
_FULL_MASS = 2.0


def conviction_raw(
    evidence: list[EvidenceItem],
    regime_confidence: float,
    *,
    drift_flag: bool = False,
    anomaly_flag: bool = False,
    full_mass: float = _FULL_MASS,
) -> float:
    """Pre-calibration conviction ∈ [0, 1] (long-only: 0 when not net-bullish)."""
    if not evidence:
        return 0.0
    signed = weighted_sum(evidence)  # normalized direction — same source as `direction`
    if signed <= 0.0:
        return 0.0  # long-only: no bullish net → no conviction
    agreement = fraction_same_sign(evidence)
    # Freshness: total live directional weight, saturating at full_mass. Decays
    # as evidence decays, so conviction fades when no fresh evidence arrives.
    mass = sum(e.weight for e in evidence if e.directional and e.source not in FLAG_SOURCES)
    freshness = min(1.0, mass / full_mass) if full_mass > 0 else 1.0
    raw = signed * (0.5 + 0.5 * agreement) * max(0.0, min(1.0, regime_confidence)) * freshness
    if drift_flag:
        raw *= _DRIFT_PENALTY
    if anomaly_flag:
        raw *= _ANOMALY_PENALTY
    return max(0.0, min(1.0, raw))


class Calibrator(Protocol):
    """Maps a raw conviction to a calibrated probability of a favorable move.

    Fit on closed-position outcomes (L8). Entry-time features only — no
    mid-trade leakage (fix R, leakage)."""

    async def calibrate(self, asset: str, raw: float, *, features: dict[str, Any]) -> float: ...


class IdentityCalibrator:
    """Cold-start fallback: calibrated == raw, clamped to [0, 1].

    Used until ``CONVICTION_min_samples_to_calibrate`` closed outcomes exist,
    and as the degradation path when a fitted calibrator fails to load (INV-1).
    """

    async def calibrate(self, asset: str, raw: float, *, features: dict[str, Any]) -> float:
        return max(0.0, min(1.0, raw))
