"""Cross-asset macro-regime signal fusion.

Round-4 wave 4.D: today the cycle treats each pair independently
(modulo correlation in `core/risk.py`). This module composes a
**single macro-regime signal** from several reference instruments
the operator already pulls or can wire in cheaply:

* VIX (S&P 500 implied vol) — fear gauge
* DXY (US dollar index) — risk-on / risk-off proxy
* US 10-year yield (and 2y for the curve) — duration stance
* Gold spot — safe-haven demand
* Sector breadth (% S&P 500 sectors above 50-day MA) — internal
  market health

The fusion rule encoded here is the textbook practitioner heuristic
(see e.g. *Tactical Asset Allocation* by Faber, *Quantitative
Investing* by Hilpisch): regime is **risk-off** when fear gauges
are elevated *and* curve is inverting *and* breadth is weak; it's
**risk-on** when the dollar is weakening *and* gold is calm *and*
breadth is broad. Anything in between is **neutral**, which the
strategy should treat as "no macro tilt — go on the per-pair
signal alone".

Why a hand-rolled rule rather than an LLM call: the macro signal
must be *fast* (≤1ms in the cycle hot path) and *deterministic*
(operators want a regression test that a known macro snapshot →
a known signal). An LLM in this slot adds latency, cost, and
non-determinism for no edge — the published practitioner rules
are clear enough.

Halal alignment: the signal is informational. It can downsize a
high-conviction LLM buy into a risk-off regime, or raise the bar
for entries during a neutral-but-weakening period. It must NEVER
open a new position by itself or short anything; that's the
strategy's domain.

Pure-Python; no NumPy / scipy / DB / async. Operates on a single
`MacroContextSnapshot` dataclass — the caller is responsible for
fetching the underlying data (FRED for yields, Yahoo / Alpaca
for VIX / DXY / gold, computed sector breadth from holdings).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ── Vocabulary ────────────────────────────────────────────


class MacroRegime(str, Enum):
    """The three regime states the strategy gates on.

    Ordering: more-permissive → more-restrictive.
    """

    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"


# ── Inputs ────────────────────────────────────────────────


@dataclass(frozen=True)
class MacroContextSnapshot:
    """One point-in-time snapshot of every macro instrument.

    Each field is nullable because the caller may have a partial
    feed — a missing VIX shouldn't crash the engine, just lower
    the confidence on the resulting signal. ``*_change_pct`` is
    the recent percent-change (caller picks 1d / 5d / 1w; the
    fusion rules are scale-invariant) over the same lookback for
    every field.
    """

    vix: float | None = None
    vix_change_pct: float | None = None
    dxy: float | None = None
    dxy_change_pct: float | None = None
    us10y_yield: float | None = None
    us10y_change_bps: float | None = None  # change in basis points
    us2y_yield: float | None = None
    gold_change_pct: float | None = None
    sector_breadth_pct: float | None = None  # 0..1, e.g. 0.65 = 65% of sectors


# ── Outputs ───────────────────────────────────────────────


@dataclass(frozen=True)
class MacroSignalReason:
    """One contributing factor in the regime decision.

    ``score`` is the bias the factor exerts on the regime: positive
    pushes toward RISK_OFF, negative toward RISK_ON. Tracking each
    factor separately lets the dashboard explain the decision in
    plain language ("VIX 32 contributes +1.0 risk-off; DXY
    weakening contributes -0.5 risk-on").
    """

    name: str
    detail: str
    score: float


@dataclass(frozen=True)
class MacroRegimeSignal:
    """Final regime call with provenance.

    ``confidence`` in [0, 1] captures how much agreement the
    factors showed *and* how complete the input was — a snapshot
    with only 2 of 5 fields populated lands at low confidence
    even if those 2 agree.

    ``risk_bias`` in [-1, 1]: positive = lean risk-off (downsize),
    negative = lean risk-on (size up modestly). Strategy hooks
    multiply candidate buy quantities by ``1 - max(0, risk_bias)``
    in the simplest gating shape.
    """

    regime: MacroRegime
    confidence: float
    risk_bias: float
    reasons: list[MacroSignalReason] = field(default_factory=list)
    measured_factor_count: int = 0
    summary: str = ""


# ── Thresholds ────────────────────────────────────────────


# Practitioner defaults. Operators can tweak via the `thresholds`
# arg on `fuse` if their reference universe differs.
@dataclass(frozen=True)
class MacroThresholds:
    """Tunable cut-offs for each factor.

    All defaults are conservative — meant to *flag* a regime shift,
    not to chase every wiggle. Tightening these tightens how often
    the engine emits a non-NEUTRAL signal.
    """

    vix_high: float = 25.0  # above → fear regime
    vix_extreme: float = 35.0  # above → strong risk-off
    vix_change_pct_alarming: float = 0.20  # +20% spike day
    dxy_change_pct_strong: float = 0.005  # ±0.5%
    yield_curve_inversion_bps: float = 0.0  # 10y - 2y < this → inversion
    us10y_change_bps_large: float = 25.0  # 25bps move = significant
    gold_change_pct_strong: float = 0.015  # 1.5%
    breadth_strong: float = 0.65  # >65% sectors above MA
    breadth_weak: float = 0.35  # <35% sectors above MA
    risk_off_score_threshold: float = 1.0  # net score → RISK_OFF
    risk_on_score_threshold: float = -1.0  # net score → RISK_ON


# ── Per-factor scorers ────────────────────────────────────


def _score_vix(snap: MacroContextSnapshot, t: MacroThresholds) -> MacroSignalReason | None:
    """VIX scorer: extreme → +2 risk-off; high → +1; spiking → +0.5
    on top of either."""
    if snap.vix is None:
        return None
    score = 0.0
    parts = []
    if snap.vix >= t.vix_extreme:
        score += 2.0
        parts.append(f"VIX {snap.vix:.1f} ≥ extreme {t.vix_extreme}")
    elif snap.vix >= t.vix_high:
        score += 1.0
        parts.append(f"VIX {snap.vix:.1f} ≥ high {t.vix_high}")
    if snap.vix_change_pct is not None and snap.vix_change_pct >= t.vix_change_pct_alarming:
        score += 0.5
        parts.append(f"VIX up {snap.vix_change_pct:.1%}")
    if score == 0.0:
        # VIX measured but calm — that's a mild risk-on signal.
        if snap.vix < t.vix_high * 0.7:
            score = -0.5
            parts.append(f"VIX calm at {snap.vix:.1f}")
    if not parts:
        parts.append(f"VIX {snap.vix:.1f} (between bands)")
    return MacroSignalReason(name="vix", detail="; ".join(parts), score=score)


def _score_dxy(snap: MacroContextSnapshot, t: MacroThresholds) -> MacroSignalReason | None:
    """DXY: rising sharply → risk-off; falling sharply → risk-on."""
    if snap.dxy_change_pct is None:
        return None
    score = 0.0
    parts = []
    if snap.dxy_change_pct >= t.dxy_change_pct_strong:
        score = 0.5
        parts.append(f"DXY +{snap.dxy_change_pct:.2%} (strengthening)")
    elif snap.dxy_change_pct <= -t.dxy_change_pct_strong:
        score = -0.5
        parts.append(f"DXY {snap.dxy_change_pct:.2%} (weakening)")
    else:
        parts.append(f"DXY {snap.dxy_change_pct:+.2%} (range)")
    return MacroSignalReason(name="dxy", detail="; ".join(parts), score=score)


def _score_yield_curve(snap: MacroContextSnapshot, t: MacroThresholds) -> MacroSignalReason | None:
    """Yield curve: inverted → risk-off; large 10y move → both
    directions, scaled by sign."""
    score = 0.0
    parts = []
    if snap.us10y_yield is not None and snap.us2y_yield is not None:
        spread = snap.us10y_yield - snap.us2y_yield
        if spread <= t.yield_curve_inversion_bps / 100.0:
            score += 1.0
            parts.append(f"curve inverted ({spread * 100:.1f}bps)")
        else:
            parts.append(f"curve {spread * 100:+.1f}bps")
    if snap.us10y_change_bps is not None:
        if snap.us10y_change_bps >= t.us10y_change_bps_large:
            score += 0.5
            parts.append(f"10y +{snap.us10y_change_bps:.0f}bps")
        elif snap.us10y_change_bps <= -t.us10y_change_bps_large:
            score -= 0.5
            parts.append(f"10y {snap.us10y_change_bps:.0f}bps")
    if not parts:
        return None
    return MacroSignalReason(name="yields", detail="; ".join(parts), score=score)


def _score_gold(snap: MacroContextSnapshot, t: MacroThresholds) -> MacroSignalReason | None:
    """Gold rallying = safe-haven demand → risk-off."""
    if snap.gold_change_pct is None:
        return None
    score = 0.0
    parts = []
    if snap.gold_change_pct >= t.gold_change_pct_strong:
        score = 0.5
        parts.append(f"gold +{snap.gold_change_pct:.2%}")
    elif snap.gold_change_pct <= -t.gold_change_pct_strong:
        # Gold dump usually risk-on (unless dollar surge — but
        # DXY scorer captures that).
        score = -0.25
        parts.append(f"gold {snap.gold_change_pct:.2%}")
    else:
        parts.append(f"gold {snap.gold_change_pct:+.2%} (range)")
    return MacroSignalReason(name="gold", detail="; ".join(parts), score=score)


def _score_breadth(snap: MacroContextSnapshot, t: MacroThresholds) -> MacroSignalReason | None:
    """Sector breadth: % of sectors above their 50-day MA. Strong
    → risk-on; weak → risk-off."""
    if snap.sector_breadth_pct is None:
        return None
    score = 0.0
    parts = []
    if snap.sector_breadth_pct >= t.breadth_strong:
        score = -0.75
        parts.append(f"breadth strong ({snap.sector_breadth_pct:.0%})")
    elif snap.sector_breadth_pct <= t.breadth_weak:
        score = 0.75
        parts.append(f"breadth weak ({snap.sector_breadth_pct:.0%})")
    else:
        parts.append(f"breadth {snap.sector_breadth_pct:.0%} (mid)")
    return MacroSignalReason(name="breadth", detail="; ".join(parts), score=score)


# ── Fusion ────────────────────────────────────────────────


_SCORERS = (
    _score_vix,
    _score_dxy,
    _score_yield_curve,
    _score_gold,
    _score_breadth,
)


def _confidence_from(measured: int, max_factors: int, agreement: float) -> float:
    """Confidence is the geometric mean of two terms:

    * **Coverage** — fraction of factors actually measured.
    * **Agreement** — how much the measured factors all point the
      same way (0 = perfect split, 1 = unanimous direction).

    Geometric mean penalises *either* term being low. A 2-of-5
    coverage with perfect agreement still lands around 0.45 —
    suspiciously confident from too little data.
    """
    if max_factors == 0:
        return 0.0
    coverage = measured / max_factors
    return (coverage * agreement) ** 0.5


def fuse(
    snapshot: MacroContextSnapshot,
    *,
    thresholds: MacroThresholds | None = None,
) -> MacroRegimeSignal:
    """Compose every available factor's score into one regime call.

    Empty / all-None snapshot returns NEUTRAL with zero confidence
    so a cold-start cycle (data feeds not yet wired) doesn't tilt
    the strategy in either direction. Pin: NEUTRAL is the safe
    default — strategies that interpret a neutral signal as "no
    macro tilt" stay correct.
    """
    t = thresholds or MacroThresholds()
    reasons: list[MacroSignalReason] = []
    for scorer in _SCORERS:
        r = scorer(snapshot, t)
        if r is not None:
            reasons.append(r)

    measured = len(reasons)
    if measured == 0:
        return MacroRegimeSignal(
            regime=MacroRegime.NEUTRAL,
            confidence=0.0,
            risk_bias=0.0,
            reasons=[],
            measured_factor_count=0,
            summary="no macro data available — defaulting to neutral",
        )

    net_score = sum(r.score for r in reasons)
    if net_score >= t.risk_off_score_threshold:
        regime = MacroRegime.RISK_OFF
    elif net_score <= t.risk_on_score_threshold:
        regime = MacroRegime.RISK_ON
    else:
        regime = MacroRegime.NEUTRAL

    # Agreement: 1 if every factor agrees with the net direction;
    # 0 if half push each way. Use absolute scores so a |0.0|
    # neutral factor doesn't punish agreement.
    abs_total = sum(abs(r.score) for r in reasons)
    if abs_total == 0:
        agreement = 1.0
    elif net_score == 0:
        agreement = 0.5
    else:
        sign = 1 if net_score > 0 else -1
        with_dir = sum(abs(r.score) for r in reasons if (r.score * sign) > 0)
        agreement = with_dir / abs_total

    confidence = _confidence_from(measured, len(_SCORERS), agreement)

    # Risk bias: clamp net score to [-2, 2] then scale to [-1, 1].
    raw_bias = max(-2.0, min(2.0, net_score)) / 2.0
    # Multiply by confidence so a low-coverage signal exerts less
    # pull on the strategy.
    risk_bias = raw_bias * confidence

    summary = _build_summary(regime, net_score, measured, len(_SCORERS))

    return MacroRegimeSignal(
        regime=regime,
        confidence=float(confidence),
        risk_bias=float(risk_bias),
        reasons=reasons,
        measured_factor_count=measured,
        summary=summary,
    )


def _build_summary(regime: MacroRegime, net_score: float, measured: int, total: int) -> str:
    """Single-line operator-readable summary."""
    coverage = f"{measured}/{total} factors"
    if regime == MacroRegime.RISK_OFF:
        return f"Macro risk-off (score {net_score:+.2f}; {coverage})"
    if regime == MacroRegime.RISK_ON:
        return f"Macro risk-on (score {net_score:+.2f}; {coverage})"
    return f"Macro neutral (score {net_score:+.2f}; {coverage})"


# ── Render helper ─────────────────────────────────────────


def render_signal(signal: MacroRegimeSignal) -> str:
    """Pretty multi-line summary for CLI / Slack / Telegram.

    Format mirrors `core/promotion_gate.render_verdict` for visual
    consistency across operator-facing reports.
    """
    lines = ["=== Macro regime signal ==="]
    emoji = {
        MacroRegime.RISK_OFF: "🔴",
        MacroRegime.NEUTRAL: "🟡",
        MacroRegime.RISK_ON: "🟢",
    }[signal.regime]
    lines.append(
        f"Regime: {emoji} {signal.regime.value} "
        f"(confidence {signal.confidence:.0%}, risk_bias {signal.risk_bias:+.2f})"
    )
    lines.append(signal.summary)
    if signal.reasons:
        lines.append("")
        lines.append("Contributing factors:")
        for r in signal.reasons:
            sign = "+" if r.score > 0 else ""
            lines.append(f"  · {r.name:<8} score={sign}{r.score:.2f}  {r.detail}")
    return "\n".join(lines)


__all__ = [
    "MacroContextSnapshot",
    "MacroRegime",
    "MacroRegimeSignal",
    "MacroSignalReason",
    "MacroThresholds",
    "fuse",
    "render_signal",
]
