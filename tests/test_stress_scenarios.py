"""Tests for ml/stress_scenarios.py — Round-5 Wave 14.C."""

from __future__ import annotations

import pytest

from halal_trader.ml.stress_scenarios import (
    SCENARIOS,
    AssetClass,
    Position,
    Scenario,
    ScenarioKind,
    apply_scenario,
    historical_scenarios,
    render_result,
    scenario_by_kind,
    synthetic_scenarios,
)

# --- Validation -----------------------------------------------------------


def test_asset_class_values():
    assert AssetClass.EQUITIES.value == "equities"
    assert AssetClass.CRYPTO.value == "crypto"
    assert AssetClass.SUKUK.value == "sukuk"
    assert AssetClass.COMMODITIES.value == "commodities"
    assert AssetClass.CASH.value == "cash"


def test_scenario_kind_values_pinned():
    expected = {
        "black_monday_1987",
        "dotcom_2000",
        "gfc_2008",
        "covid_2020",
        "rate_hikes_2022",
        "vol_spike",
        "liquidity_freeze",
        "rate_shock",
        "currency_deval",
        "black_swan",
    }
    assert {k.value for k in ScenarioKind} == expected


def test_scenario_empty_description_rejected():
    with pytest.raises(ValueError):
        Scenario(
            kind=ScenarioKind.VOL_SPIKE,
            description="",
            shocks={AssetClass.EQUITIES: -0.1},
            is_historical=False,
        )


def test_scenario_unreasonable_shock_rejected():
    with pytest.raises(ValueError):
        Scenario(
            kind=ScenarioKind.VOL_SPIKE,
            description="x",
            shocks={AssetClass.EQUITIES: -2.0},  # below -100% impossible
            is_historical=False,
        )


def test_position_negative_value_rejected():
    with pytest.raises(ValueError):
        Position(symbol="A", asset_class=AssetClass.EQUITIES, market_value=-1.0)


def test_position_empty_symbol_rejected():
    with pytest.raises(ValueError):
        Position(symbol="", asset_class=AssetClass.EQUITIES, market_value=100.0)


# --- Catalogue ----------------------------------------------------------


def test_catalogue_covers_all_kinds():
    catalogued_kinds = {s.kind for s in SCENARIOS}
    assert catalogued_kinds == set(ScenarioKind)


def test_scenario_by_kind_returns_match():
    s = scenario_by_kind(ScenarioKind.GFC_2008)
    assert s.kind is ScenarioKind.GFC_2008


def test_scenario_by_kind_unknown_raises():
    """All known kinds should resolve, but a synthetic 'unknown' would raise."""

    class _FakeKind:
        pass

    with pytest.raises(KeyError):
        scenario_by_kind(_FakeKind())  # type: ignore[arg-type]


def test_historical_scenarios_filter():
    h = historical_scenarios()
    assert all(s.is_historical for s in h)
    assert any(s.kind is ScenarioKind.GFC_2008 for s in h)


def test_synthetic_scenarios_filter():
    s = synthetic_scenarios()
    assert all(not x.is_historical for x in s)
    assert any(x.kind is ScenarioKind.VOL_SPIKE for x in s)


def test_historical_and_synthetic_partition():
    assert len(historical_scenarios()) + len(synthetic_scenarios()) == len(SCENARIOS)


# --- Apply scenario -----------------------------------------------------


def test_apply_gfc_to_equity_portfolio():
    positions = [
        Position(symbol="AAPL", asset_class=AssetClass.EQUITIES, market_value=10000),
    ]
    result = apply_scenario(positions, scenario_by_kind(ScenarioKind.GFC_2008))
    assert result.starting_value == 10000
    assert result.projected_value == 5000  # -50%
    assert result.pct_change == pytest.approx(-0.50)


def test_apply_to_diversified_portfolio_blends_shocks():
    positions = [
        Position("AAPL", AssetClass.EQUITIES, 10000),
        Position("BTC", AssetClass.CRYPTO, 5000),
        Position("SUKUK", AssetClass.SUKUK, 5000),
    ]
    result = apply_scenario(positions, scenario_by_kind(ScenarioKind.COVID_2020))
    # equities -34%, crypto -50%, sukuk -5%
    expected = 10000 * 0.66 + 5000 * 0.50 + 5000 * 0.95
    assert result.projected_value == pytest.approx(expected)


def test_apply_per_position_records_each():
    positions = [
        Position("AAPL", AssetClass.EQUITIES, 10000),
        Position("MSFT", AssetClass.EQUITIES, 20000),
    ]
    result = apply_scenario(positions, scenario_by_kind(ScenarioKind.BLACK_MONDAY_1987))
    assert len(result.per_position) == 2
    assert result.per_position[0][0] == "AAPL"
    assert result.per_position[0][1] == 10000  # old
    assert result.per_position[0][2] == pytest.approx(7800)  # -22%


def test_apply_empty_portfolio():
    result = apply_scenario([], scenario_by_kind(ScenarioKind.GFC_2008))
    assert result.starting_value == 0
    assert result.projected_value == 0
    assert result.pct_change == 0


def test_apply_currency_deval_cash_drops():
    positions = [Position("USD", AssetClass.CASH, 10000)]
    result = apply_scenario(
        positions, scenario_by_kind(ScenarioKind.CURRENCY_DEVAL)
    )
    assert result.projected_value == pytest.approx(7000)


def test_apply_unknown_asset_class_uses_zero_shock():
    """Asset class missing from a scenario's shocks dict gets 0 shock."""
    custom = Scenario(
        kind=ScenarioKind.VOL_SPIKE,
        description="custom",
        shocks={AssetClass.EQUITIES: -0.10},  # missing CRYPTO
        is_historical=False,
    )
    positions = [Position("BTC", AssetClass.CRYPTO, 1000)]
    result = apply_scenario(positions, custom)
    assert result.projected_value == 1000  # unchanged


# --- Render -------------------------------------------------------------


def test_render_result_loss_uses_down_arrow():
    positions = [Position("AAPL", AssetClass.EQUITIES, 10000)]
    result = apply_scenario(positions, scenario_by_kind(ScenarioKind.GFC_2008))
    out = render_result(result)
    assert "▼" in out


def test_render_result_gain_uses_up_arrow():
    """Currency-deval scenario can produce gains for crypto-heavy portfolio."""
    positions = [Position("BTC", AssetClass.CRYPTO, 10000)]
    result = apply_scenario(
        positions, scenario_by_kind(ScenarioKind.CURRENCY_DEVAL)
    )
    out = render_result(result)
    assert "▲" in out


def test_render_no_secret_leak():
    positions = [Position("AAPL", AssetClass.EQUITIES, 10000)]
    result = apply_scenario(positions, scenario_by_kind(ScenarioKind.GFC_2008))
    out = render_result(result)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E ---------------------------------------------------------------


def test_e2e_run_all_scenarios_against_portfolio():
    positions = [
        Position("AAPL", AssetClass.EQUITIES, 50000),
        Position("BTC", AssetClass.CRYPTO, 20000),
        Position("SUKUK", AssetClass.SUKUK, 30000),
    ]
    results = [apply_scenario(positions, s) for s in SCENARIOS]
    assert len(results) == len(SCENARIOS)
    # Worst case = black swan
    worst = min(results, key=lambda r: r.pct_change)
    assert worst.scenario_kind is ScenarioKind.BLACK_SWAN
