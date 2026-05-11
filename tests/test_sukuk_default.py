"""Tests for markets/sukuk_default.py — Round-5 Wave 3.F."""

from __future__ import annotations

import math

import pytest

from halal_trader.halal.aaoifi_standard_17 import SukukType
from halal_trader.markets.sukuk_default import (
    PortfolioLossReport,
    RecoveryParams,
    Region,
    SukukExposure,
    cumulative_pd,
    expected_loss,
    hazard_multiplier,
    lookup_recovery,
    loss_at_quantile,
    portfolio_loss_report,
    quantile,
    render_report,
    simulate_losses,
)

# --- Region + RecoveryParams ----------------------------------------------


def test_region_string_values():
    assert Region.GCC.value == "gcc"
    assert Region.ASIA_PACIFIC.value == "asia_pacific"
    assert Region.EUROPE.value == "europe"
    assert Region.AFRICA.value == "africa"


def test_recovery_params_valid():
    rp = RecoveryParams(mean=0.6, std=0.15)
    assert rp.mean == 0.6
    assert rp.std == 0.15


@pytest.mark.parametrize("mean", [-0.01, 1.01, 1.5])
def test_recovery_params_invalid_mean(mean):
    with pytest.raises(ValueError):
        RecoveryParams(mean=mean, std=0.15)


@pytest.mark.parametrize("std", [-0.01, 0.51, 1.0])
def test_recovery_params_invalid_std(std):
    with pytest.raises(ValueError):
        RecoveryParams(mean=0.5, std=std)


def test_recovery_params_immutable():
    rp = RecoveryParams(mean=0.6, std=0.15)
    with pytest.raises(AttributeError):
        rp.mean = 0.7  # type: ignore[misc]


# --- lookup_recovery ------------------------------------------------------


def test_lookup_recovery_gcc_ijara_strongest():
    rp = lookup_recovery(SukukType.IJARA, Region.GCC)
    assert rp.mean == 0.80


def test_lookup_recovery_asia_partnership_low():
    """Partnership sukuk in Asia recover materially lower than GCC."""
    asia = lookup_recovery(SukukType.MUDARABAH, Region.ASIA_PACIFIC)
    gcc = lookup_recovery(SukukType.MUDARABAH, Region.GCC)
    assert asia.mean < gcc.mean


def test_lookup_recovery_asset_backed_dominates_partnership():
    """Pinned: Ijara recovery > Mudarabah recovery, in any region."""
    for region in Region:
        ijara = lookup_recovery(SukukType.IJARA, region)
        mudarabah = lookup_recovery(SukukType.MUDARABAH, region)
        assert ijara.mean > mudarabah.mean


def test_lookup_recovery_override_applied():
    overrides = {(SukukType.IJARA, Region.GCC): (0.95, 0.05)}
    rp = lookup_recovery(SukukType.IJARA, Region.GCC, overrides=overrides)
    assert rp.mean == 0.95
    assert rp.std == 0.05


def test_lookup_recovery_override_only_for_specific_pair():
    """An override for one pair must not bleed into another."""
    overrides = {(SukukType.IJARA, Region.GCC): (0.95, 0.05)}
    asia = lookup_recovery(SukukType.IJARA, Region.ASIA_PACIFIC, overrides=overrides)
    assert asia.mean != 0.95


def test_lookup_recovery_table_completeness():
    """Every (type, region) pair must have a calibration."""
    for st in SukukType:
        for region in Region:
            rp = lookup_recovery(st, region)
            assert 0.0 <= rp.mean <= 1.0


# --- hazard_multiplier ----------------------------------------------------


def test_hazard_multiplier_asset_backed_discount():
    assert hazard_multiplier(SukukType.IJARA) < 1.0


def test_hazard_multiplier_partnership_premium():
    assert hazard_multiplier(SukukType.MUDARABAH) > 1.0
    assert hazard_multiplier(SukukType.MUSHARAKAH) > 1.0


def test_hazard_multiplier_pure_debt_neutral():
    assert hazard_multiplier(SukukType.MURABAHA) == 1.0
    assert hazard_multiplier(SukukType.SALAM) == 1.0


def test_hazard_multiplier_ordering_pinned():
    """Pinned: Ijara < Istisna < Wakalah < Mudarabah/Musharakah."""
    assert (
        hazard_multiplier(SukukType.IJARA)
        < hazard_multiplier(SukukType.ISTISNA)
        < hazard_multiplier(SukukType.WAKALAH)
        < hazard_multiplier(SukukType.MUDARABAH)
    )


# --- cumulative_pd --------------------------------------------------------


def test_cumulative_pd_zero_horizon_rejected():
    with pytest.raises(ValueError):
        cumulative_pd(base_hazard=0.02, sukuk_type=SukukType.IJARA, horizon_years=0)


def test_cumulative_pd_negative_hazard_rejected():
    with pytest.raises(ValueError):
        cumulative_pd(base_hazard=-0.01, sukuk_type=SukukType.IJARA, horizon_years=1.0)


def test_cumulative_pd_increases_with_horizon():
    pd1 = cumulative_pd(base_hazard=0.02, sukuk_type=SukukType.IJARA, horizon_years=1)
    pd5 = cumulative_pd(base_hazard=0.02, sukuk_type=SukukType.IJARA, horizon_years=5)
    assert pd5 > pd1


def test_cumulative_pd_partnership_higher_than_asset_backed():
    pd_ijara = cumulative_pd(base_hazard=0.02, sukuk_type=SukukType.IJARA, horizon_years=5)
    pd_mudarabah = cumulative_pd(base_hazard=0.02, sukuk_type=SukukType.MUDARABAH, horizon_years=5)
    assert pd_mudarabah > pd_ijara


def test_cumulative_pd_bounded_above_by_1():
    """Even at extreme hazards, cumulative PD must be in [0, 1]."""
    pd = cumulative_pd(base_hazard=10.0, sukuk_type=SukukType.MUDARABAH, horizon_years=100)
    assert 0.0 <= pd <= 1.0


# --- SukukExposure --------------------------------------------------------


def test_exposure_valid():
    e = SukukExposure(
        issuer="GovOfMalaysia",
        sukuk_type=SukukType.IJARA,
        region=Region.ASIA_PACIFIC,
        face_value=1000.0,
    )
    assert e.face_value == 1000.0


def test_exposure_empty_issuer_rejected():
    with pytest.raises(ValueError):
        SukukExposure(
            issuer="",
            sukuk_type=SukukType.IJARA,
            region=Region.ASIA_PACIFIC,
            face_value=1000.0,
        )


def test_exposure_negative_face_rejected():
    with pytest.raises(ValueError):
        SukukExposure(
            issuer="GovOfMalaysia",
            sukuk_type=SukukType.IJARA,
            region=Region.ASIA_PACIFIC,
            face_value=-1.0,
        )


# --- expected_loss --------------------------------------------------------


def test_expected_loss_pinned_arithmetic():
    """Pin EL = PD × LGD × face on a fully-deterministic case."""
    e = SukukExposure(
        issuer="X",
        sukuk_type=SukukType.IJARA,
        region=Region.GCC,
        face_value=1000.0,
    )
    est = expected_loss(e, base_hazard=0.02, horizon_years=5)
    expected_pd = 1 - math.exp(-0.02 * 0.7 * 5)  # h=0.014, t=5
    expected_lgd = 1 - 0.80
    assert abs(est.pd - expected_pd) < 1e-9
    assert abs(est.mean_recovery - 0.80) < 1e-9
    assert abs(est.expected_loss - expected_pd * expected_lgd * 1000.0) < 1e-6


def test_expected_loss_lgd_helper():
    e = SukukExposure(
        issuer="X",
        sukuk_type=SukukType.IJARA,
        region=Region.GCC,
        face_value=1000.0,
    )
    est = expected_loss(e, base_hazard=0.02, horizon_years=5)
    assert abs(est.lgd() - 0.20) < 1e-9


def test_expected_loss_partnership_higher_than_asset_backed():
    """Same face, same hazard — partnership EL > asset-backed EL."""
    base = dict(
        issuer="X",
        face_value=1000.0,
        region=Region.ASIA_PACIFIC,
    )
    e_ijara = SukukExposure(sukuk_type=SukukType.IJARA, **base)
    e_mudarabah = SukukExposure(sukuk_type=SukukType.MUDARABAH, **base)
    el_i = expected_loss(e_ijara, base_hazard=0.02, horizon_years=5)
    el_m = expected_loss(e_mudarabah, base_hazard=0.02, horizon_years=5)
    assert el_m.expected_loss > el_i.expected_loss


def test_expected_loss_loss_std_positive():
    e = SukukExposure(
        issuer="X",
        sukuk_type=SukukType.MUDARABAH,
        region=Region.AFRICA,
        face_value=1000.0,
    )
    est = expected_loss(e, base_hazard=0.05, horizon_years=10)
    assert est.loss_std > 0


# --- loss_at_quantile -----------------------------------------------------


def test_loss_at_quantile_above_expected_loss():
    e = SukukExposure(
        issuer="X",
        sukuk_type=SukukType.MUDARABAH,
        region=Region.GCC,
        face_value=1000.0,
    )
    el = expected_loss(e, base_hazard=0.05, horizon_years=10).expected_loss
    var_95 = loss_at_quantile(e, base_hazard=0.05, horizon_years=10, quantile=0.95)
    assert var_95 > el


def test_loss_at_quantile_invalid_q_rejected():
    e = SukukExposure(
        issuer="X",
        sukuk_type=SukukType.IJARA,
        region=Region.GCC,
        face_value=1000.0,
    )
    with pytest.raises(ValueError):
        loss_at_quantile(e, base_hazard=0.05, horizon_years=10, quantile=0.0)
    with pytest.raises(ValueError):
        loss_at_quantile(e, base_hazard=0.05, horizon_years=10, quantile=1.0)


def test_loss_at_quantile_clamped_at_zero():
    """For very-low-hazard exposures, gaussian-approximation might
    return a negative — must clamp to 0."""
    e = SukukExposure(
        issuer="X",
        sukuk_type=SukukType.IJARA,
        region=Region.GCC,
        face_value=1000.0,
    )
    var_05 = loss_at_quantile(e, base_hazard=0.001, horizon_years=1, quantile=0.05)
    assert var_05 >= 0.0


# --- simulate_losses ------------------------------------------------------


def test_simulate_losses_seed_determinism():
    e = SukukExposure(
        issuer="X",
        sukuk_type=SukukType.MUDARABAH,
        region=Region.ASIA_PACIFIC,
        face_value=1000.0,
    )
    p1 = simulate_losses([e], base_hazard=0.05, horizon_years=10, n_paths=500, seed=42)
    p2 = simulate_losses([e], base_hazard=0.05, horizon_years=10, n_paths=500, seed=42)
    assert p1 == p2


def test_simulate_losses_seed_change_changes_paths():
    e = SukukExposure(
        issuer="X",
        sukuk_type=SukukType.MUDARABAH,
        region=Region.ASIA_PACIFIC,
        face_value=1000.0,
    )
    p1 = simulate_losses([e], base_hazard=0.05, horizon_years=10, n_paths=500, seed=42)
    p2 = simulate_losses([e], base_hazard=0.05, horizon_years=10, n_paths=500, seed=43)
    assert p1 != p2


def test_simulate_losses_empty_exposures():
    assert simulate_losses([], base_hazard=0.02, horizon_years=5) == ()


def test_simulate_losses_invalid_n_paths():
    e = SukukExposure(
        issuer="X",
        sukuk_type=SukukType.IJARA,
        region=Region.GCC,
        face_value=1000.0,
    )
    with pytest.raises(ValueError):
        simulate_losses([e], base_hazard=0.02, horizon_years=5, n_paths=0)


def test_simulate_losses_sorted_ascending():
    e = SukukExposure(
        issuer="X",
        sukuk_type=SukukType.MUDARABAH,
        region=Region.ASIA_PACIFIC,
        face_value=1000.0,
    )
    paths = simulate_losses([e], base_hazard=0.05, horizon_years=10, n_paths=500, seed=42)
    assert list(paths) == sorted(paths)


def test_simulate_losses_bounded_by_face():
    e = SukukExposure(
        issuer="X",
        sukuk_type=SukukType.MUDARABAH,
        region=Region.AFRICA,
        face_value=1000.0,
    )
    paths = simulate_losses([e], base_hazard=0.5, horizon_years=10, n_paths=500, seed=0)
    for p in paths:
        assert 0.0 <= p <= 1000.0


# --- quantile -------------------------------------------------------------


def test_quantile_min():
    paths = (1.0, 2.0, 3.0, 4.0, 5.0)
    assert quantile(paths, 0.0) == 1.0


def test_quantile_max():
    paths = (1.0, 2.0, 3.0, 4.0, 5.0)
    assert quantile(paths, 1.0) == 5.0


def test_quantile_median():
    paths = (1.0, 2.0, 3.0, 4.0, 5.0)
    assert quantile(paths, 0.5) == 3.0


def test_quantile_empty_rejected():
    with pytest.raises(ValueError):
        quantile((), 0.5)


def test_quantile_out_of_range_rejected():
    with pytest.raises(ValueError):
        quantile((1.0, 2.0), -0.1)
    with pytest.raises(ValueError):
        quantile((1.0, 2.0), 1.1)


# --- portfolio_loss_report ------------------------------------------------


def test_portfolio_loss_report_basic():
    exposures = [
        SukukExposure(
            issuer="A",
            sukuk_type=SukukType.IJARA,
            region=Region.GCC,
            face_value=1000.0,
        ),
        SukukExposure(
            issuer="B",
            sukuk_type=SukukType.MUDARABAH,
            region=Region.ASIA_PACIFIC,
            face_value=2000.0,
        ),
    ]
    rep = portfolio_loss_report(exposures, base_hazard=0.05, horizon_years=10, n_paths=500, seed=42)
    assert rep.n_exposures == 2
    assert rep.total_face == 3000.0
    assert rep.var_95 >= rep.expected_loss
    assert rep.var_99 >= rep.var_95
    assert rep.cvar_95 >= rep.var_95


def test_portfolio_loss_report_empty_rejected():
    with pytest.raises(ValueError):
        portfolio_loss_report([], base_hazard=0.02, horizon_years=5)


def test_portfolio_loss_report_seed_determinism():
    exposures = [
        SukukExposure(
            issuer="A",
            sukuk_type=SukukType.IJARA,
            region=Region.GCC,
            face_value=1000.0,
        )
    ]
    r1 = portfolio_loss_report(exposures, base_hazard=0.05, horizon_years=10, n_paths=500, seed=42)
    r2 = portfolio_loss_report(exposures, base_hazard=0.05, horizon_years=10, n_paths=500, seed=42)
    assert r1 == r2


# --- render_report --------------------------------------------------------


def test_render_report_no_secret_leak():
    """Pinned no-secret-leak: render output mentions face/loss only."""
    rep = PortfolioLossReport(
        n_exposures=2,
        total_face=3000.0,
        expected_loss=120.0,
        var_95=400.0,
        var_99=600.0,
        cvar_95=550.0,
    )
    out = render_report(rep, currency="USD")
    assert "📉" in out
    assert "USD" in out
    assert "Expected loss" in out
    assert "VaR 95%" in out
    assert "CVaR 95%" in out
    # No issuer or hazard rate leaks
    assert "issuer" not in out.lower()
    assert "hazard" not in out.lower()
