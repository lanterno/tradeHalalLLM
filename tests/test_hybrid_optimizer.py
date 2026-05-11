"""Tests for markets/hybrid_optimizer.py — Round-5 Wave 3.G."""

from __future__ import annotations

import pytest

from halal_trader.halal.aaoifi_standard_17 import SukukType
from halal_trader.markets.hybrid_optimizer import (
    AssetClass,
    HybridAsset,
    HybridConstraints,
    HybridResult,
    optimize_hybrid,
    render_hybrid,
)


def _eq(
    symbol: str = "AAPL",
    sector: str = "technology",
    jurisdiction: str = "US",
    er: float = 0.08,
    div: float = 0.02,
) -> HybridAsset:
    return HybridAsset(
        symbol=symbol,
        asset_class=AssetClass.EQUITY,
        sector=sector,
        jurisdiction=jurisdiction,
        expected_return=er,
        dividend_yield=div,
    )


def _sk(
    symbol: str = "MY-IJARA-2030",
    sector: str = "sovereign",
    jurisdiction: str = "MY",
    er: float = 0.04,
    duration: float = 5.0,
    sukuk_type: SukukType = SukukType.IJARA,
) -> HybridAsset:
    return HybridAsset(
        symbol=symbol,
        asset_class=AssetClass.SUKUK,
        sector=sector,
        jurisdiction=jurisdiction,
        expected_return=er,
        duration_years=duration,
        sukuk_type=sukuk_type,
    )


# --- HybridAsset validation ---------------------------------------------


def test_equity_asset_valid():
    a = _eq()
    assert a.asset_class is AssetClass.EQUITY
    assert a.duration_years == 0.0


def test_sukuk_asset_valid():
    a = _sk()
    assert a.asset_class is AssetClass.SUKUK
    assert a.sukuk_type is SukukType.IJARA


def test_equity_with_duration_rejected():
    with pytest.raises(ValueError):
        HybridAsset(
            symbol="X",
            asset_class=AssetClass.EQUITY,
            sector="tech",
            jurisdiction="US",
            expected_return=0.08,
            duration_years=5.0,
        )


def test_sukuk_without_type_rejected():
    with pytest.raises(ValueError):
        HybridAsset(
            symbol="X",
            asset_class=AssetClass.SUKUK,
            sector="sovereign",
            jurisdiction="MY",
            expected_return=0.04,
            duration_years=5.0,
        )


def test_sukuk_non_tradable_type_rejected():
    with pytest.raises(ValueError):
        HybridAsset(
            symbol="X",
            asset_class=AssetClass.SUKUK,
            sector="sovereign",
            jurisdiction="MY",
            expected_return=0.04,
            duration_years=5.0,
            sukuk_type=SukukType.MURABAHA,
        )


def test_sukuk_zero_duration_rejected():
    with pytest.raises(ValueError):
        HybridAsset(
            symbol="X",
            asset_class=AssetClass.SUKUK,
            sector="sovereign",
            jurisdiction="MY",
            expected_return=0.04,
            duration_years=0.0,
            sukuk_type=SukukType.IJARA,
        )


def test_equity_invalid_return():
    with pytest.raises(ValueError):
        _eq(er=0.99)


def test_equity_invalid_div():
    with pytest.raises(ValueError):
        _eq(div=0.50)


def test_asset_immutable():
    a = _eq()
    with pytest.raises(AttributeError):
        a.expected_return = 0.10  # type: ignore[misc]


# --- HybridConstraints validation ---------------------------------------


def test_constraints_default():
    c = HybridConstraints()
    assert c.equity_band == (0.30, 0.70)


def test_constraints_invalid_band():
    with pytest.raises(ValueError):
        HybridConstraints(equity_band=(0.7, 0.3))
    with pytest.raises(ValueError):
        HybridConstraints(equity_band=(-0.1, 0.5))


def test_constraints_bands_must_overlap_top():
    with pytest.raises(ValueError):
        HybridConstraints(
            equity_band=(0.0, 0.4),
            sukuk_band=(0.0, 0.4),
        )


def test_constraints_bands_must_overlap_bottom():
    with pytest.raises(ValueError):
        HybridConstraints(
            equity_band=(0.6, 1.0),
            sukuk_band=(0.6, 1.0),
        )


def test_constraints_invalid_max_single_name():
    with pytest.raises(ValueError):
        HybridConstraints(max_single_name=0.0)


# --- optimize_hybrid basic shape ----------------------------------------


def test_optimize_basic():
    assets = [_eq(symbol="A"), _eq(symbol="B"), _sk(symbol="C"), _sk(symbol="D")]
    res = optimize_hybrid(assets)
    assert isinstance(res, HybridResult)
    assert abs(sum(res.weights) - 1.0) < 1e-6
    for w in res.weights:
        assert w >= -1e-9


def test_optimize_empty_rejected():
    with pytest.raises(ValueError):
        optimize_hybrid([])


def test_optimize_single_class_only_equity():
    """A pure-equity universe still optimises but the sukuk band is
    relaxed to 0 (no sukuk to fill the band)."""
    assets = [_eq(symbol="A"), _eq(symbol="B")]
    cstr = HybridConstraints(
        equity_band=(0.0, 1.0),
        sukuk_band=(0.0, 1.0),
        max_single_name=0.99,
    )
    res = optimize_hybrid(assets, constraints=cstr)
    assert res.equity_weight > 0.99
    assert res.sukuk_weight < 0.01


def test_optimize_single_class_only_sukuk():
    assets = [_sk(symbol="A"), _sk(symbol="B")]
    cstr = HybridConstraints(
        equity_band=(0.0, 1.0),
        sukuk_band=(0.0, 1.0),
        max_single_name=0.99,
    )
    res = optimize_hybrid(assets, constraints=cstr)
    assert res.sukuk_weight > 0.99


# --- Class band enforcement ---------------------------------------------


def test_equity_band_enforced():
    assets = [
        _eq(symbol="A", er=0.20),
        _eq(symbol="B", er=0.20),
        _sk(symbol="C", er=0.04),
        _sk(symbol="D", er=0.04),
    ]
    cstr = HybridConstraints(
        equity_band=(0.30, 0.50),
        sukuk_band=(0.50, 0.70),
        max_single_name=0.99,
    )
    res = optimize_hybrid(assets, constraints=cstr)
    assert 0.30 - 1e-6 <= res.equity_weight <= 0.50 + 1e-6
    assert 0.50 - 1e-6 <= res.sukuk_weight <= 0.70 + 1e-6


def test_sukuk_band_enforced():
    assets = [
        _eq(symbol="A", sector="industrials", er=0.04),
        _eq(symbol="B", sector="industrials", er=0.04),
        _sk(symbol="C", er=0.20),
        _sk(symbol="D", er=0.20),
    ]
    cstr = HybridConstraints(
        equity_band=(0.50, 0.70),
        sukuk_band=(0.30, 0.50),
        max_single_name=0.99,
        # Loosen the AAOIFI default for industrials so the class
        # band — not the sector cap — is the binding constraint.
        equity_sector_caps={"industrials": 0.99},
    )
    res = optimize_hybrid(assets, constraints=cstr)
    assert res.sukuk_weight <= 0.50 + 1e-6
    assert res.equity_weight >= 0.50 - 1e-6


# --- Single-name cap ----------------------------------------------------


def test_single_name_cap_enforced():
    """Diversify across sectors so AAOIFI sector caps do not clash
    with the class-band feasibility check; the test then isolates
    whether single-name caps are enforced."""
    assets = [
        _eq(symbol="A", sector="technology", er=0.20),
        _eq(symbol="B", sector="healthcare", er=0.20),
        _eq(symbol="C", sector="industrials", er=0.20),
        _sk(symbol="D"),
        _sk(symbol="E"),
        _sk(symbol="F"),
    ]
    cstr = HybridConstraints(
        equity_band=(0.30, 0.70),
        sukuk_band=(0.30, 0.70),
        max_single_name=0.30,
    )
    res = optimize_hybrid(assets, constraints=cstr)
    for w in res.weights:
        assert w <= 0.30 + 1e-6


# --- AAOIFI sector caps -------------------------------------------------


def test_default_aaoifi_financials_cap():
    """Pinned: default cap on financials = 33%."""
    assets = [
        _eq(symbol="A", sector="financials", er=0.20),
        _eq(symbol="B", sector="financials", er=0.20),
        _sk(symbol="C"),
    ]
    cstr = HybridConstraints(equity_band=(0.0, 1.0), sukuk_band=(0.0, 1.0), max_single_name=0.99)
    res = optimize_hybrid(assets, constraints=cstr)
    fin_w = res.weights[0] + res.weights[1]
    assert fin_w <= 0.33 + 1e-6


def test_operator_can_override_sector_cap():
    assets = [
        _eq(symbol="A", sector="financials", er=0.20),
        _eq(symbol="B", sector="financials", er=0.20),
        _sk(symbol="C"),
    ]
    cstr = HybridConstraints(
        equity_band=(0.0, 1.0),
        sukuk_band=(0.0, 1.0),
        max_single_name=0.99,
        equity_sector_caps={"financials": 0.10},
    )
    res = optimize_hybrid(assets, constraints=cstr)
    fin_w = res.weights[0] + res.weights[1]
    assert fin_w <= 0.10 + 1e-6


# --- Portfolio metrics --------------------------------------------------


def test_metrics_match_weights():
    """Portfolio expected_return = sum(w_i * er_i)."""
    assets = [_eq(symbol="A", er=0.10), _sk(symbol="B", er=0.04, duration=5.0)]
    cstr = HybridConstraints(
        equity_band=(0.4, 0.6),
        sukuk_band=(0.4, 0.6),
        max_single_name=0.99,
    )
    res = optimize_hybrid(assets, constraints=cstr)
    expected = res.weights[0] * 0.10 + res.weights[1] * 0.04
    assert abs(res.expected_return - expected) < 1e-9


def test_sukuk_duration_only_counts_sukuk():
    """Equity duration_years=0 → only sukuk contribute to sukuk_duration."""
    assets = [_eq(symbol="A", er=0.10), _sk(symbol="B", er=0.04, duration=8.0)]
    cstr = HybridConstraints(
        equity_band=(0.4, 0.6),
        sukuk_band=(0.4, 0.6),
        max_single_name=0.99,
    )
    res = optimize_hybrid(assets, constraints=cstr)
    expected_dur = res.weights[1] * 8.0
    assert abs(res.sukuk_duration - expected_dur) < 1e-9


# --- Sukuk duration target ----------------------------------------------


def test_sukuk_duration_target_pulls_basket():
    assets = [
        _eq(symbol="EQ"),
        _sk(symbol="SK1y", duration=1.0, er=0.03),
        _sk(symbol="SK10y", duration=10.0, er=0.05),
    ]
    cstr_short = HybridConstraints(
        equity_band=(0.3, 0.5),
        sukuk_band=(0.5, 0.7),
        max_single_name=0.99,
        sukuk_duration_target=2.0,
        risk_aversion=2.0,
    )
    cstr_long = HybridConstraints(
        equity_band=(0.3, 0.5),
        sukuk_band=(0.5, 0.7),
        max_single_name=0.99,
        sukuk_duration_target=8.0,
        risk_aversion=2.0,
    )
    res_short = optimize_hybrid(assets, constraints=cstr_short)
    res_long = optimize_hybrid(assets, constraints=cstr_long)
    assert res_short.sukuk_duration < res_long.sukuk_duration


# --- Determinism --------------------------------------------------------


def test_deterministic():
    assets = [_eq(symbol="A"), _sk(symbol="B")]
    r1 = optimize_hybrid(assets)
    r2 = optimize_hybrid(assets)
    for w1, w2 in zip(r1.weights, r2.weights, strict=True):
        assert abs(w1 - w2) < 1e-9


# --- Custom covariance --------------------------------------------------


def test_custom_covariance_size_mismatch_rejected():
    assets = [_eq(symbol="A"), _sk(symbol="B")]
    with pytest.raises(ValueError):
        optimize_hybrid(assets, covariance=[[0.04]])


def test_custom_covariance_asymmetric_rejected():
    assets = [_eq(symbol="A"), _sk(symbol="B")]
    with pytest.raises(ValueError):
        optimize_hybrid(assets, covariance=[[0.04, 0.01], [0.05, 0.04]])


# --- Render -------------------------------------------------------------


def test_render_contains_class_summary():
    assets = [_eq(symbol="A"), _sk(symbol="B")]
    res = optimize_hybrid(assets)
    out = render_hybrid(res)
    assert "Hybrid portfolio" in out
    assert "equity" in out.lower()
    assert "sukuk" in out.lower()


def test_render_no_secret_leak():
    assets = [_eq(symbol="A"), _sk(symbol="B")]
    res = optimize_hybrid(assets)
    out = render_hybrid(res)
    assert "covariance" not in out.lower()
    assert "gradient" not in out.lower()


def test_render_top_n_cap():
    assets = [_eq(symbol=f"E{i}") for i in range(5)] + [_sk(symbol=f"S{i}") for i in range(5)]
    res = optimize_hybrid(assets)
    out = render_hybrid(res, top_n=3)
    # head line + at most 3 bullets.
    bullets = [line for line in out.split("\n") if line.startswith("  •")]
    assert len(bullets) <= 3
