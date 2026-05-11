"""Tests for ml/halal_optimizer.py — Round-5 Wave 7.E."""

from __future__ import annotations

import pytest

from halal_trader.ml.halal_optimizer import (
    HalalAsset,
    HalalAssetClass,
    HalalOptResult,
    HalalPolicy,
    InfeasibleBasketError,
    optimize,
    render_result,
)


def _eq(symbol: str = "AAPL", sector: str = "technology", er: float = 0.08) -> HalalAsset:
    return HalalAsset(
        symbol=symbol,
        asset_class=HalalAssetClass.EQUITY,
        sector=sector,
        expected_return=er,
    )


def _sk(
    symbol: str = "MY-SK",
    sector: str = "sovereign",
    er: float = 0.04,
    dur: float = 5.0,
) -> HalalAsset:
    return HalalAsset(
        symbol=symbol,
        asset_class=HalalAssetClass.SUKUK,
        sector=sector,
        expected_return=er,
        duration_years=dur,
    )


def _pool(symbol: str = "MUDA-POOL", er: float = 0.06) -> HalalAsset:
    return HalalAsset(
        symbol=symbol,
        asset_class=HalalAssetClass.MUDARABAH_POOL,
        sector="pool",
        expected_return=er,
    )


# --- HalalAsset validation ----------------------------------------------


def test_equity_valid():
    a = _eq()
    assert a.asset_class is HalalAssetClass.EQUITY


def test_sukuk_valid():
    a = _sk()
    assert a.duration_years == 5.0


def test_pool_valid():
    a = _pool()
    assert a.asset_class is HalalAssetClass.MUDARABAH_POOL


def test_equity_with_duration_rejected():
    with pytest.raises(ValueError):
        HalalAsset(
            symbol="X",
            asset_class=HalalAssetClass.EQUITY,
            sector="tech",
            expected_return=0.05,
            duration_years=3.0,
        )


def test_sukuk_zero_duration_rejected():
    with pytest.raises(ValueError):
        HalalAsset(
            symbol="X",
            asset_class=HalalAssetClass.SUKUK,
            sector="sovereign",
            expected_return=0.04,
            duration_years=0.0,
        )


def test_asset_immutable():
    a = _eq()
    with pytest.raises(AttributeError):
        a.expected_return = 0.10  # type: ignore[misc]


# --- HalalPolicy validation ---------------------------------------------


def test_policy_default():
    p = HalalPolicy()
    assert p.mudarabah_pool_weight == 0.0


def test_policy_pool_at_one_rejected():
    with pytest.raises(ValueError):
        HalalPolicy(mudarabah_pool_weight=1.0)


def test_policy_negative_pool_rejected():
    with pytest.raises(ValueError):
        HalalPolicy(mudarabah_pool_weight=-0.1)


def test_policy_invalid_sukuk_band():
    with pytest.raises(ValueError):
        HalalPolicy(sukuk_min_weight=0.7, sukuk_max_weight=0.3)


def test_policy_invalid_sector_cap():
    with pytest.raises(ValueError):
        HalalPolicy(sector_caps={"tech": 1.5})


# --- optimize basic ------------------------------------------------------


def test_optimize_basic_three_class():
    assets = [
        _eq(symbol="A"),
        _sk(symbol="B"),
        _pool(symbol="C"),
    ]
    res = optimize(
        assets,
        policy=HalalPolicy(mudarabah_pool_weight=0.20, max_single_name=0.99),
    )
    assert isinstance(res, HalalOptResult)
    assert abs(sum(res.weights) - 1.0) < 1e-6


def test_optimize_empty_rejected():
    with pytest.raises(ValueError):
        optimize([])


def test_optimize_universe_too_large_rejected():
    with pytest.raises(ValueError):
        optimize([_eq(symbol=f"X{i}") for i in range(201)])


def test_optimize_pool_weight_pinned():
    """Pin: Mudarabah pool must hit the configured weight exactly."""
    assets = [_eq(symbol="A"), _sk(symbol="B"), _pool(symbol="C")]
    res = optimize(
        assets,
        policy=HalalPolicy(mudarabah_pool_weight=0.30, max_single_name=0.99),
    )
    assert res.mudarabah_weight == pytest.approx(0.30, abs=1e-4)


def test_optimize_pool_required_when_weight_set():
    """Pin: pool_weight > 0 with no pool asset → InfeasibleBasketError."""
    assets = [_eq(symbol="A"), _sk(symbol="B")]
    with pytest.raises(InfeasibleBasketError):
        optimize(
            assets,
            policy=HalalPolicy(mudarabah_pool_weight=0.20),
        )


def test_optimize_no_pool_when_weight_zero():
    """Default policy has pool_weight=0; no pool asset required."""
    assets = [_eq(symbol="A"), _sk(symbol="B")]
    res = optimize(assets)
    assert res.mudarabah_weight == 0.0


# --- AAOIFI sector cap defaults -----------------------------------------


def test_default_aaoifi_financials_cap():
    assets = [
        _eq(symbol="A", sector="financials", er=0.20),
        _eq(symbol="B", sector="financials", er=0.20),
        _sk(symbol="C"),
    ]
    res = optimize(assets, policy=HalalPolicy(max_single_name=0.99))
    fin_w = res.weights[0] + res.weights[1]
    assert fin_w <= 0.33 + 1e-6


def test_operator_can_override_sector_cap():
    assets = [
        _eq(symbol="A", sector="financials", er=0.20),
        _eq(symbol="B", sector="financials", er=0.20),
        _sk(symbol="C"),
    ]
    res = optimize(
        assets,
        policy=HalalPolicy(
            max_single_name=0.99,
            sector_caps={"financials": 0.10},
        ),
    )
    fin_w = res.weights[0] + res.weights[1]
    assert fin_w <= 0.10 + 1e-6


# --- Sukuk band ----------------------------------------------------------


def test_sukuk_min_weight_floor():
    assets = [_eq(symbol="A", er=0.20), _sk(symbol="B", er=0.04)]
    res = optimize(
        assets,
        policy=HalalPolicy(
            sukuk_min_weight=0.40,
            max_single_name=0.99,
        ),
    )
    assert res.sukuk_weight >= 0.40 - 1e-6


def test_sukuk_max_weight_ceiling():
    """Loosen the equity sector cap so the sukuk_max ceiling is the
    binding constraint."""
    assets = [_eq(symbol="A", sector="industrials", er=0.04), _sk(symbol="B", er=0.20)]
    res = optimize(
        assets,
        policy=HalalPolicy(
            sukuk_max_weight=0.30,
            max_single_name=0.99,
            sector_caps={"industrials": 0.99},
        ),
    )
    assert res.sukuk_weight <= 0.30 + 1e-6


# --- Determinism ---------------------------------------------------------


def test_deterministic():
    assets = [_eq(symbol="A"), _sk(symbol="B"), _pool(symbol="C")]
    pol = HalalPolicy(mudarabah_pool_weight=0.20)
    r1 = optimize(assets, policy=pol)
    r2 = optimize(assets, policy=pol)
    for w1, w2 in zip(r1.weights, r2.weights, strict=True):
        assert abs(w1 - w2) < 1e-9


# --- Custom covariance --------------------------------------------------


def test_custom_covariance_size_mismatch():
    assets = [_eq(symbol="A"), _sk(symbol="B")]
    with pytest.raises(ValueError):
        optimize(assets, covariance=[[0.04]])


def test_custom_covariance_asymmetric():
    assets = [_eq(symbol="A"), _sk(symbol="B")]
    with pytest.raises(ValueError):
        optimize(assets, covariance=[[0.04, 0.01], [0.05, 0.04]])


# --- HalalOptResult helpers ---------------------------------------------


def test_volatility_helper():
    assets = [_eq(symbol="A"), _sk(symbol="B")]
    res = optimize(assets)
    assert res.expected_volatility() >= 0


# --- Render --------------------------------------------------------------


def test_render_contains_class_summary():
    assets = [_eq(symbol="A"), _sk(symbol="B"), _pool(symbol="C")]
    res = optimize(
        assets,
        policy=HalalPolicy(mudarabah_pool_weight=0.20, max_single_name=0.99),
    )
    out = render_result(res)
    assert "Halal portfolio" in out
    assert "mudarabah" in out.lower()


def test_render_no_secret_leak():
    assets = [_eq(symbol="A"), _sk(symbol="B")]
    res = optimize(assets)
    out = render_result(res)
    assert "covariance" not in out.lower()
    assert "gradient" not in out.lower()
