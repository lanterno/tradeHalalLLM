"""Tests for the minimum-sample guard."""

from __future__ import annotations

from halal_trader.core.sample_guard import (
    DEFAULT_MIN_SAMPLES,
    SampleGate,
    gate_stat,
)


def test_gate_sufficient_and_shortfall():
    g = SampleGate(n=25, min_n=20)
    assert g.sufficient
    assert g.shortfall == 0

    g2 = SampleGate(n=12, min_n=20)
    assert not g2.sufficient
    assert g2.shortfall == 8


def test_gate_default_threshold():
    assert SampleGate(n=DEFAULT_MIN_SAMPLES).sufficient
    assert not SampleGate(n=DEFAULT_MIN_SAMPLES - 1).sufficient


def test_gate_stat_returns_value_when_sufficient():
    assert gate_stat(0.42, n=30, min_n=20, fallback=0.0) == 0.42


def test_gate_stat_returns_fallback_when_insufficient():
    # A Kelly fraction off thin data falls back to no bet.
    assert gate_stat(0.42, n=5, min_n=20, fallback=0.0) == 0.0
    # A calibrated confidence falls back to the raw confidence.
    assert gate_stat(0.9, n=3, min_n=20, fallback=0.55) == 0.55
    # An IC falls back to None.
    assert gate_stat(0.1, n=1, min_n=30, fallback=None) is None


def test_gate_stat_boundary_inclusive():
    assert gate_stat("x", n=20, min_n=20, fallback="fb") == "x"  # >= is sufficient
    assert gate_stat("x", n=19, min_n=20, fallback="fb") == "fb"
