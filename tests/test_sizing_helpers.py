"""Tests for the private helpers in :mod:`core.sizing`.

The full Kelly sizer (`size_position`) is integration-tested in
`test_sizing.py` against representative scenarios; this file pins
the small numeric helpers underneath: `_clamp`, `_confidence_to_edge`,
and `_vol_scale`.
"""

from __future__ import annotations

from decimal import Decimal

from halal_trader.core.sizing import (
    VOL_SCALE_MAX,
    VOL_SCALE_MIN,
    _clamp,
    _confidence_to_edge,
    _vol_scale,
)

# ── _clamp ──────────────────────────────────────────────────


def test_clamp_within_range_passes_through():
    assert _clamp(Decimal("0.5"), Decimal("0.0"), Decimal("1.0")) == Decimal("0.5")


def test_clamp_below_lo_returns_lo():
    assert _clamp(Decimal("-0.5"), Decimal("0.0"), Decimal("1.0")) == Decimal("0.0")


def test_clamp_above_hi_returns_hi():
    assert _clamp(Decimal("2.0"), Decimal("0.0"), Decimal("1.0")) == Decimal("1.0")


def test_clamp_at_boundaries_passes_through():
    """Inclusive at both ends — exactly-lo and exactly-hi return their value."""
    assert _clamp(Decimal("0.0"), Decimal("0.0"), Decimal("1.0")) == Decimal("0.0")
    assert _clamp(Decimal("1.0"), Decimal("0.0"), Decimal("1.0")) == Decimal("1.0")


# ── _confidence_to_edge ────────────────────────────────────


def test_confidence_to_edge_below_half_returns_zero():
    """We never bet on coin flips — p ≤ 0.5 yields zero size."""
    assert _confidence_to_edge(0.5) == Decimal("0")
    assert _confidence_to_edge(0.3) == Decimal("0")
    assert _confidence_to_edge(0.0) == Decimal("0")


def test_confidence_to_edge_at_one_returns_one():
    """Maximum conviction → edge=1 (full Kelly fraction kicks in)."""
    assert _confidence_to_edge(1.0) == Decimal("1")


def test_confidence_to_edge_basic_2p_minus_1():
    """0.7 confidence → 2*0.7 - 1 = 0.4 edge."""
    assert _confidence_to_edge(0.7) == Decimal("0.4")


def test_confidence_to_edge_clamps_above_one():
    """Defensive: an LLM hallucinating confidence=1.5 still gives a
    valid edge (clamped to 1.0 → edge 1.0)."""
    assert _confidence_to_edge(1.5) == Decimal("1")


def test_confidence_to_edge_clamps_below_zero():
    """Negative confidence (also a hallucination) → no bet."""
    assert _confidence_to_edge(-0.2) == Decimal("0")


def test_confidence_to_edge_avoids_binary_float_drift():
    """The Decimal(str(p)) routing keeps 0.7 → 0.4 clean instead of
    0.39999…99…. Important for the test's exact Decimal compare."""
    edge = _confidence_to_edge(0.7)
    assert str(edge) == "0.4"


# ── _vol_scale ──────────────────────────────────────────────


def test_vol_scale_returns_one_when_inputs_invalid():
    """Zero/negative ATR or baseline → fall back to neutral 1.0
    (sizer reverts to confidence × Kelly only)."""
    assert _vol_scale(atr_pct=0.0, baseline=0.02) == Decimal("1")
    assert _vol_scale(atr_pct=0.02, baseline=0.0) == Decimal("1")
    assert _vol_scale(atr_pct=-0.01, baseline=0.02) == Decimal("1")


def test_vol_scale_at_baseline_returns_one():
    """Right at baseline → no scaling."""
    assert _vol_scale(atr_pct=0.02, baseline=0.02) == Decimal("1")


def test_vol_scale_high_vol_shrinks_below_one():
    """ATR > baseline → scale < 1 (smaller size)."""
    out = _vol_scale(atr_pct=0.04, baseline=0.02)
    assert out < Decimal("1")
    # Specifically: baseline / atr = 0.02/0.04 = 0.5
    assert out == Decimal("0.5")


def test_vol_scale_low_vol_boosts_above_one():
    """ATR < baseline → scale > 1 (larger size)."""
    out = _vol_scale(atr_pct=0.01, baseline=0.02)
    assert out > Decimal("1")
    assert out == Decimal("2")


def test_vol_scale_clamped_at_minimum():
    """An extreme high-vol regime still leaves *some* size on the table."""
    out = _vol_scale(atr_pct=1.0, baseline=0.02)  # 50× baseline
    assert out == VOL_SCALE_MIN


def test_vol_scale_clamped_at_maximum():
    """An ultra-quiet regime can't push the boost arbitrarily high."""
    out = _vol_scale(atr_pct=0.0001, baseline=0.02)  # 0.5% of baseline
    assert out == VOL_SCALE_MAX
