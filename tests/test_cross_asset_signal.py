"""Tests for `core/cross_asset_signal.py` (macro-regime fusion).

Pins each per-factor scorer's threshold logic, the fusion's
risk-on / neutral / risk-off bands, the confidence formula
(coverage × agreement, geometric mean), the cold-start zero-data
contract, and the render helper output.
"""

from __future__ import annotations

import pytest

from halal_trader.core.cross_asset_signal import (
    MacroContextSnapshot,
    MacroRegime,
    MacroSignalReason,
    MacroThresholds,
    fuse,
    render_signal,
)

# ── cold-start ───────────────────────────────────────────


def test_empty_snapshot_returns_neutral_with_zero_confidence():
    """Pin the safe default: a cold-start cycle (no feeds wired)
    must NOT tilt the strategy in either direction."""
    sig = fuse(MacroContextSnapshot())
    assert sig.regime == MacroRegime.NEUTRAL
    assert sig.confidence == 0.0
    assert sig.risk_bias == 0.0
    assert sig.measured_factor_count == 0
    assert "no macro data" in sig.summary.lower()


# ── VIX scorer ───────────────────────────────────────────


def test_vix_extreme_pushes_risk_off():
    sig = fuse(MacroContextSnapshot(vix=40.0))
    assert sig.regime == MacroRegime.RISK_OFF
    assert any(r.name == "vix" and r.score >= 2.0 for r in sig.reasons)


def test_vix_high_but_not_extreme_contributes_one_point():
    sig = fuse(MacroContextSnapshot(vix=30.0))
    vix = next(r for r in sig.reasons if r.name == "vix")
    assert 0.5 < vix.score < 2.0


def test_vix_calm_contributes_mild_risk_on():
    """VIX well below the high threshold pushes the signal a touch
    toward risk-on. Pin so the calm-market case isn't a no-op."""
    sig = fuse(MacroContextSnapshot(vix=15.0))
    vix = next(r for r in sig.reasons if r.name == "vix")
    assert vix.score < 0


def test_vix_spike_adds_extra_score_on_top_of_band():
    sig = fuse(MacroContextSnapshot(vix=30.0, vix_change_pct=0.30))
    vix = next(r for r in sig.reasons if r.name == "vix")
    # Should be band score (1.0) + spike (0.5).
    assert vix.score >= 1.4


def test_vix_missing_yields_no_factor():
    sig = fuse(MacroContextSnapshot(dxy_change_pct=0.0))
    names = {r.name for r in sig.reasons}
    assert "vix" not in names


# ── DXY scorer ───────────────────────────────────────────


def test_dxy_strong_strengthening_is_risk_off():
    sig = fuse(MacroContextSnapshot(dxy_change_pct=0.01))
    dxy = next(r for r in sig.reasons if r.name == "dxy")
    assert dxy.score > 0


def test_dxy_strong_weakening_is_risk_on():
    sig = fuse(MacroContextSnapshot(dxy_change_pct=-0.01))
    dxy = next(r for r in sig.reasons if r.name == "dxy")
    assert dxy.score < 0


def test_dxy_within_range_contributes_zero():
    sig = fuse(MacroContextSnapshot(dxy_change_pct=0.001))
    dxy = next(r for r in sig.reasons if r.name == "dxy")
    assert dxy.score == 0.0


# ── yield curve scorer ───────────────────────────────────


def test_inverted_curve_pushes_risk_off():
    sig = fuse(MacroContextSnapshot(us10y_yield=4.0, us2y_yield=4.5))
    yields = next(r for r in sig.reasons if r.name == "yields")
    assert yields.score >= 1.0
    assert "inverted" in yields.detail.lower()


def test_normal_curve_yields_zero_score_factor():
    """A normal positive-spread curve contributes 0 from the curve
    rule; the factor is still recorded for transparency."""
    sig = fuse(MacroContextSnapshot(us10y_yield=4.5, us2y_yield=4.0))
    yields = next(r for r in sig.reasons if r.name == "yields")
    assert yields.score == 0.0


def test_large_10y_jump_adds_to_score():
    sig = fuse(MacroContextSnapshot(us10y_change_bps=30.0))
    yields = next(r for r in sig.reasons if r.name == "yields")
    assert yields.score > 0
    assert "10y" in yields.detail


def test_large_10y_drop_subtracts_from_score():
    sig = fuse(MacroContextSnapshot(us10y_change_bps=-30.0))
    yields = next(r for r in sig.reasons if r.name == "yields")
    assert yields.score < 0


def test_yield_curve_only_one_side_supplied_skips_factor():
    """If only us10y_yield is set (no us2y), the curve check skips
    but the move-magnitude check still runs."""
    sig = fuse(MacroContextSnapshot(us10y_yield=4.0))
    names = {r.name for r in sig.reasons}
    assert "yields" not in names  # neither curve nor move magnitude applicable


# ── gold scorer ──────────────────────────────────────────


def test_gold_rally_is_risk_off():
    sig = fuse(MacroContextSnapshot(gold_change_pct=0.02))
    gold = next(r for r in sig.reasons if r.name == "gold")
    assert gold.score > 0


def test_gold_dump_is_mildly_risk_on():
    sig = fuse(MacroContextSnapshot(gold_change_pct=-0.02))
    gold = next(r for r in sig.reasons if r.name == "gold")
    assert gold.score < 0
    # Asymmetric: dump signal weaker than rally signal.
    assert abs(gold.score) < 0.5


# ── breadth scorer ───────────────────────────────────────


def test_strong_breadth_is_risk_on():
    sig = fuse(MacroContextSnapshot(sector_breadth_pct=0.80))
    breadth = next(r for r in sig.reasons if r.name == "breadth")
    assert breadth.score < 0


def test_weak_breadth_is_risk_off():
    sig = fuse(MacroContextSnapshot(sector_breadth_pct=0.20))
    breadth = next(r for r in sig.reasons if r.name == "breadth")
    assert breadth.score > 0


def test_mid_breadth_is_neutral():
    sig = fuse(MacroContextSnapshot(sector_breadth_pct=0.50))
    breadth = next(r for r in sig.reasons if r.name == "breadth")
    assert breadth.score == 0.0


# ── fusion bands ─────────────────────────────────────────


def test_fusion_classic_risk_off_combo_lands_in_risk_off():
    """The textbook risk-off scenario: VIX > 30 + curve inverted
    + breadth weak. Must trip the risk-off threshold."""
    snap = MacroContextSnapshot(
        vix=32.0,
        us10y_yield=4.0,
        us2y_yield=4.4,
        sector_breadth_pct=0.30,
    )
    sig = fuse(snap)
    assert sig.regime == MacroRegime.RISK_OFF
    assert sig.risk_bias > 0


def test_fusion_classic_risk_on_combo_lands_in_risk_on():
    """The textbook risk-on scenario: low VIX + DXY weakening +
    breadth strong + gold calm."""
    snap = MacroContextSnapshot(
        vix=14.0,
        dxy_change_pct=-0.008,
        sector_breadth_pct=0.75,
        gold_change_pct=0.0,
    )
    sig = fuse(snap)
    assert sig.regime == MacroRegime.RISK_ON
    assert sig.risk_bias < 0


def test_fusion_mixed_signals_land_in_neutral():
    """When factors fight each other, the engine should err
    neutral rather than picking a side."""
    snap = MacroContextSnapshot(
        vix=27.0,  # mildly risk-off
        sector_breadth_pct=0.70,  # mildly risk-on
        dxy_change_pct=0.0,
    )
    sig = fuse(snap)
    assert sig.regime == MacroRegime.NEUTRAL


def test_fusion_threshold_can_be_tightened():
    """Operators can change risk_off_score_threshold; pin it
    flows through to the regime decision."""
    snap = MacroContextSnapshot(vix=30.0)  # ~+1.0 score
    default = fuse(snap)
    assert default.regime == MacroRegime.RISK_OFF
    strict = fuse(snap, thresholds=MacroThresholds(risk_off_score_threshold=2.0))
    assert strict.regime == MacroRegime.NEUTRAL


# ── confidence formula ───────────────────────────────────


def test_confidence_uses_coverage_and_agreement():
    """Pin: 1-of-5 coverage even with full agreement → low
    confidence (coverage × agreement, geometric mean)."""
    sig = fuse(MacroContextSnapshot(vix=14.0))  # 1 of 5
    # geometric mean of (1/5) × 1.0 = √0.2 ≈ 0.45
    assert 0.4 < sig.confidence < 0.5


def test_confidence_high_with_full_coverage_and_agreement():
    """All five factors measured + all pointing the same way →
    confidence near 1.0."""
    snap = MacroContextSnapshot(
        vix=40.0,
        vix_change_pct=0.30,
        dxy_change_pct=0.01,
        us10y_yield=4.0,
        us2y_yield=4.5,
        us10y_change_bps=30.0,
        gold_change_pct=0.02,
        sector_breadth_pct=0.20,
    )
    sig = fuse(snap)
    assert sig.confidence > 0.9


def test_confidence_low_when_factors_split():
    """Half pulling each way → agreement near 0 → low confidence
    even with full coverage."""
    snap = MacroContextSnapshot(
        vix=40.0,  # +2 risk-off
        sector_breadth_pct=0.85,  # -0.75 risk-on
        dxy_change_pct=-0.01,  # -0.5 risk-on
        us10y_yield=4.5,
        us2y_yield=4.0,  # 0
        gold_change_pct=-0.02,  # -0.25 risk-on
    )
    sig = fuse(snap)
    # Coverage high but agreement low; net score still positive but
    # confidence shouldn't be wild.
    assert sig.confidence < 0.85


def test_confidence_is_zero_with_no_measured_factors():
    sig = fuse(MacroContextSnapshot())
    assert sig.confidence == 0.0


# ── risk bias ─────────────────────────────────────────────


def test_risk_bias_bounded_by_minus_one_and_one():
    """Pin: even on extreme scores, risk_bias clamps so a strategy
    multiplying by `1 - max(0, risk_bias)` never goes negative."""
    snap = MacroContextSnapshot(
        vix=60.0,
        vix_change_pct=1.0,
        dxy_change_pct=0.05,
        us10y_yield=4.0,
        us2y_yield=4.5,
        us10y_change_bps=100.0,
        gold_change_pct=0.10,
        sector_breadth_pct=0.05,
    )
    sig = fuse(snap)
    assert -1.0 <= sig.risk_bias <= 1.0


def test_risk_bias_scales_with_confidence():
    """A signal with low confidence should exert less pull. Pin so
    a partial-feed cycle doesn't yank the strategy around."""
    full_data = MacroContextSnapshot(
        vix=40.0,
        vix_change_pct=0.30,
        dxy_change_pct=0.01,
        us10y_yield=4.0,
        us2y_yield=4.5,
        us10y_change_bps=30.0,
        gold_change_pct=0.02,
        sector_breadth_pct=0.20,
    )
    partial = MacroContextSnapshot(vix=40.0)
    full_bias = abs(fuse(full_data).risk_bias)
    partial_bias = abs(fuse(partial).risk_bias)
    assert full_bias > partial_bias


# ── reason structure ─────────────────────────────────────


def test_reasons_list_is_typed():
    sig = fuse(MacroContextSnapshot(vix=20.0))
    assert all(isinstance(r, MacroSignalReason) for r in sig.reasons)


def test_signal_includes_summary_and_factor_count():
    sig = fuse(MacroContextSnapshot(vix=40.0))
    assert sig.measured_factor_count == 1
    assert "1/5 factors" in sig.summary


def test_signal_is_immutable():
    sig = fuse(MacroContextSnapshot())
    with pytest.raises(Exception):
        sig.regime = MacroRegime.RISK_ON  # type: ignore[misc]


# ── render_signal ────────────────────────────────────────


def test_render_includes_regime_emoji_for_risk_off():
    snap = MacroContextSnapshot(vix=40.0)
    text = render_signal(fuse(snap))
    assert "🔴" in text
    assert "risk_off" in text


def test_render_includes_emoji_for_each_regime():
    risk_on = render_signal(fuse(MacroContextSnapshot(vix=14.0, sector_breadth_pct=0.85)))
    neutral = render_signal(fuse(MacroContextSnapshot()))
    risk_off = render_signal(fuse(MacroContextSnapshot(vix=40.0)))
    assert "🟢" in risk_on
    assert "🟡" in neutral
    assert "🔴" in risk_off


def test_render_lists_contributing_factors():
    text = render_signal(fuse(MacroContextSnapshot(vix=40.0)))
    assert "Contributing factors" in text
    assert "vix" in text


def test_render_handles_no_factors():
    text = render_signal(fuse(MacroContextSnapshot()))
    assert "neutral" in text
    assert "Contributing factors" not in text


def test_render_includes_confidence_and_risk_bias():
    text = render_signal(fuse(MacroContextSnapshot(vix=40.0)))
    assert "confidence" in text
    assert "risk_bias" in text
