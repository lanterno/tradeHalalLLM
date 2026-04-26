"""Tests for spot-perp basis features."""

from __future__ import annotations

import pytest

from halal_trader.crypto.basis import (
    BasisFeatures,
    BasisRiskPolicy,
    BasisTracker,
    compute_basis,
    format_basis_for_prompt,
)


# ── compute_basis ────────────────────────────────────────────────


def test_basis_bps_positive_for_contango() -> None:
    f = compute_basis(
        pair="BTCUSDT",
        spot_price=100.0,
        perp_price=100.5,
        funding_rate_pct=0.0001,
    )
    assert f.basis_bps == pytest.approx(50.0, abs=0.1)
    assert f.regime == "contango"


def test_basis_bps_negative_for_backwardation() -> None:
    f = compute_basis(
        pair="BTCUSDT",
        spot_price=100.0,
        perp_price=99.5,
        funding_rate_pct=-0.0001,
    )
    assert f.basis_bps == pytest.approx(-50.0, abs=0.1)
    assert f.regime == "backwardation"


def test_basis_neutral_when_funding_disagrees() -> None:
    # Positive basis but negative funding (rare divergence) -> neutral
    f = compute_basis(
        pair="BTCUSDT",
        spot_price=100.0,
        perp_price=100.5,
        funding_rate_pct=-0.0001,
    )
    assert f.regime == "neutral"


def test_basis_zero_when_spot_zero() -> None:
    f = compute_basis(
        pair="BTCUSDT",
        spot_price=0.0,
        perp_price=100.0,
        funding_rate_pct=0.0001,
    )
    assert f.basis_bps == 0.0


def test_basis_zscore_zero_with_short_history() -> None:
    f = compute_basis(
        pair="BTCUSDT",
        spot_price=100.0,
        perp_price=100.5,
        funding_rate_pct=0.0,
        basis_history=[40.0, 50.0],
    )
    assert f.basis_zscore == 0.0


def test_basis_zscore_nonzero_with_history() -> None:
    history = [10.0, 12.0, 11.0, 13.0, 9.0, 10.0]  # mean ~ 10.8, std ~ 1.5
    f = compute_basis(
        pair="BTCUSDT",
        spot_price=100.0,
        perp_price=101.0,  # basis 100 bps
        funding_rate_pct=0.0,
        basis_history=history,
    )
    assert f.basis_zscore > 5  # 100 bps is way above the trailing 10ish


# ── BasisTracker ─────────────────────────────────────────────────


def test_tracker_accumulates_and_zscores() -> None:
    tr = BasisTracker(window=10)
    # warmup with slightly varying low basis (need non-zero variance for z-score)
    for i in range(8):
        perp = 100.05 + (0.001 * (i % 2))
        tr.observe(pair="BTCUSDT", spot_price=100.0, perp_price=perp, funding_rate_pct=0.0)
    # spike
    f = tr.observe(pair="BTCUSDT", spot_price=100.0, perp_price=101.0, funding_rate_pct=0.0)
    assert f.basis_zscore > 2


def test_tracker_separates_pairs() -> None:
    tr = BasisTracker(window=20)
    tr.observe(pair="BTCUSDT", spot_price=100.0, perp_price=100.5, funding_rate_pct=0.0)
    tr.observe(pair="ETHUSDT", spot_price=200.0, perp_price=199.0, funding_rate_pct=0.0)
    assert "BTCUSDT" in tr.history_by_pair
    assert "ETHUSDT" in tr.history_by_pair
    assert len(tr.history_by_pair["BTCUSDT"]) == 1


def test_tracker_window_caps_history() -> None:
    tr = BasisTracker(window=5)
    for _ in range(10):
        tr.observe(pair="X", spot_price=100.0, perp_price=100.0, funding_rate_pct=0.0)
    assert len(tr.history_by_pair["X"]) == 5


# ── Prompt formatting ────────────────────────────────────────────


def test_format_empty_features_returns_empty_string() -> None:
    assert format_basis_for_prompt({}) == ""


def test_format_writes_each_pair() -> None:
    f1 = compute_basis(
        pair="BTCUSDT",
        spot_price=100.0,
        perp_price=100.5,
        funding_rate_pct=0.0001,
    )
    f2 = compute_basis(
        pair="ETHUSDT",
        spot_price=200.0,
        perp_price=199.0,
        funding_rate_pct=-0.0001,
    )
    text = format_basis_for_prompt({"BTCUSDT": f1, "ETHUSDT": f2})
    assert "BTCUSDT" in text
    assert "ETHUSDT" in text
    assert "contango" in text
    assert "backwardation" in text


# ── Risk policy ──────────────────────────────────────────────────


def test_risk_policy_normal_full_size() -> None:
    pol = BasisRiskPolicy()
    f = compute_basis(
        pair="BTCUSDT", spot_price=100.0, perp_price=100.5, funding_rate_pct=0.0001
    )
    assert pol.buy_size_multiplier(f) == 1.0


def test_risk_policy_extreme_contango_shrinks() -> None:
    pol = BasisRiskPolicy(extreme_contango_bps=100.0, extreme_size_multiplier=0.4)
    f = compute_basis(
        pair="BTCUSDT", spot_price=100.0, perp_price=102.0, funding_rate_pct=0.001
    )
    assert pol.buy_size_multiplier(f) == 0.4


def test_risk_policy_extreme_backwardation_shrinks() -> None:
    pol = BasisRiskPolicy(extreme_backwardation_bps=-100.0, extreme_size_multiplier=0.3)
    f = compute_basis(
        pair="BTCUSDT", spot_price=100.0, perp_price=98.0, funding_rate_pct=-0.001
    )
    assert pol.buy_size_multiplier(f) == 0.3


def test_risk_policy_neutral_full_size() -> None:
    pol = BasisRiskPolicy()
    f = BasisFeatures(
        pair="BTCUSDT",
        spot_price=100.0,
        perp_price=100.0,
        funding_rate_pct=0.0,
        basis_bps=0.0,
        regime="neutral",
    )
    assert pol.buy_size_multiplier(f) == 1.0
