"""Tests for halal/mudarabah_term_sheet.py — Round-5 Wave 6.B."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.halal.mudarabah_term_sheet import (
    MudarabahTermSheet,
    ProhibitedClause,
    render_scenario,
    render_term_sheet,
    render_validation,
    scenario_payout,
    validate_term_sheet,
)


def _sheet(
    deal_id: str = "D1",
    investor_name: str = "alice",
    founder_name: str = "bob",
    capital: float = 1_000_000.0,
    profit_share: float = 0.70,
    valuation: float = 5_000_000.0,
    closing_date: date = date(2026, 6, 1),
    horizon: int = 5,
    guaranteed: float = 0.0,
    has_liquidation_preference: bool = False,
    has_cumulative_dividend: bool = False,
    has_interest_bearing_debt: bool = False,
    has_fixed_payout: bool = False,
    has_ratchet_anti_dilution: bool = False,
    has_senior_preferred_shares: bool = False,
) -> MudarabahTermSheet:
    return MudarabahTermSheet(
        deal_id=deal_id,
        investor_name=investor_name,
        founder_name=founder_name,
        capital_amount_usd=capital,
        profit_share_investor=profit_share,
        valuation_usd=valuation,
        closing_date=closing_date,
        expected_horizon_years=horizon,
        guaranteed_return_pct=guaranteed,
        has_liquidation_preference=has_liquidation_preference,
        has_cumulative_dividend=has_cumulative_dividend,
        has_interest_bearing_debt=has_interest_bearing_debt,
        has_fixed_payout=has_fixed_payout,
        has_ratchet_anti_dilution=has_ratchet_anti_dilution,
        has_senior_preferred_shares=has_senior_preferred_shares,
    )


# --- Sheet validation ---------------------------------------------------


def test_sheet_valid():
    s = _sheet()
    assert s.profit_share_investor == 0.70
    assert s.founder_share() == pytest.approx(0.30)


def test_sheet_self_dealing_rejected():
    with pytest.raises(ValueError):
        _sheet(investor_name="x", founder_name="x")


def test_sheet_negative_capital_rejected():
    with pytest.raises(ValueError):
        _sheet(capital=-1.0)


def test_sheet_capital_above_valuation_rejected():
    with pytest.raises(ValueError):
        _sheet(capital=10_000_000.0, valuation=5_000_000.0)


def test_sheet_profit_share_at_one_rejected():
    """Pin: investor cannot take 100% — founder needs a share."""
    with pytest.raises(ValueError):
        _sheet(profit_share=1.0)


def test_sheet_profit_share_at_zero_rejected():
    """Pin: investor cannot take 0% — that's a gift, not Mudarabah."""
    with pytest.raises(ValueError):
        _sheet(profit_share=0.0)


def test_sheet_negative_horizon_rejected():
    with pytest.raises(ValueError):
        _sheet(horizon=0)


def test_sheet_negative_guaranteed_rejected():
    with pytest.raises(ValueError):
        _sheet(guaranteed=-0.01)


def test_sheet_immutable():
    s = _sheet()
    with pytest.raises(AttributeError):
        s.profit_share_investor = 0.5  # type: ignore[misc]


# --- validate_term_sheet — clean path -----------------------------------


def test_clean_sheet_passes():
    s = _sheet()
    res = validate_term_sheet(s)
    assert res.is_halal
    assert not res.prohibited_clauses


# --- Each prohibited clause -------------------------------------------


def test_guaranteed_return_rejected():
    s = _sheet(guaranteed=0.05)
    res = validate_term_sheet(s)
    assert not res.is_halal
    assert ProhibitedClause.GUARANTEED_RETURN in res.prohibited_clauses


def test_liquidation_preference_rejected():
    s = _sheet(has_liquidation_preference=True)
    res = validate_term_sheet(s)
    assert not res.is_halal
    assert ProhibitedClause.LIQUIDATION_PREFERENCE in res.prohibited_clauses


def test_cumulative_dividend_rejected():
    s = _sheet(has_cumulative_dividend=True)
    res = validate_term_sheet(s)
    assert not res.is_halal
    assert ProhibitedClause.CUMULATIVE_DIVIDEND in res.prohibited_clauses


def test_interest_debt_rejected():
    s = _sheet(has_interest_bearing_debt=True)
    res = validate_term_sheet(s)
    assert not res.is_halal
    assert ProhibitedClause.INTEREST_BEARING_DEBT in res.prohibited_clauses


def test_fixed_payout_rejected():
    s = _sheet(has_fixed_payout=True)
    res = validate_term_sheet(s)
    assert not res.is_halal
    assert ProhibitedClause.FIXED_PAYOUT in res.prohibited_clauses


def test_ratchet_anti_dilution_rejected():
    s = _sheet(has_ratchet_anti_dilution=True)
    res = validate_term_sheet(s)
    assert not res.is_halal
    assert ProhibitedClause.RATCHET_ANTI_DILUTION in res.prohibited_clauses


def test_senior_preferred_rejected():
    s = _sheet(has_senior_preferred_shares=True)
    res = validate_term_sheet(s)
    assert not res.is_halal
    assert ProhibitedClause.SENIOR_PREFERRED_SHARES in res.prohibited_clauses


def test_multiple_breaches_all_returned():
    """Pin: validation surfaces every breach in one pass."""
    s = _sheet(
        guaranteed=0.05,
        has_liquidation_preference=True,
        has_cumulative_dividend=True,
    )
    res = validate_term_sheet(s)
    assert not res.is_halal
    assert len(res.prohibited_clauses) == 3


def test_weighted_anti_dilution_is_halal():
    """Pin: weighted-average anti-dilution is permissible (default)."""
    s = _sheet()  # weighted_anti_dilution_allowed=True default
    res = validate_term_sheet(s)
    assert res.is_halal


# --- scenario_payout — profit branch -----------------------------------


def test_scenario_profit_split():
    """Pin: profit split per ratio."""
    s = _sheet(profit_share=0.70)
    out = scenario_payout(s, venture_pnl=1_000_000.0)
    assert out.investor_payout == pytest.approx(700_000.0)
    assert out.founder_payout == pytest.approx(300_000.0)


def test_scenario_zero_pnl():
    s = _sheet()
    out = scenario_payout(s, venture_pnl=0.0)
    assert out.investor_payout == 0.0
    assert out.founder_payout == 0.0


# --- scenario_payout — loss branch -------------------------------------


def test_scenario_loss_borne_by_investor_only():
    """Pin: loss borne 100% by investor (Standard 13)."""
    s = _sheet()
    out = scenario_payout(s, venture_pnl=-100_000.0)
    assert out.investor_payout == -100_000.0
    assert out.founder_payout == 0.0


def test_scenario_loss_capped_at_capital():
    """Pin: loss capped at capital (no clawback)."""
    s = _sheet(capital=500_000.0)
    out = scenario_payout(s, venture_pnl=-1_000_000.0)
    assert out.investor_payout == -500_000.0
    assert out.founder_payout == 0.0
    assert "capped" in out.note


def test_scenario_note_present():
    s = _sheet(profit_share=0.60)
    out = scenario_payout(s, venture_pnl=1000.0)
    assert "60%" in out.note or "0.60" in out.note or "split" in out.note


# --- Render --------------------------------------------------------------


def test_render_term_sheet_no_secret_leak():
    """Pin: render masks investor + founder names."""
    s = _sheet(investor_name="alice@example.com", founder_name="bob@example.com")
    out = render_term_sheet(s)
    assert "alice@example.com" not in out
    assert "bob@example.com" not in out


def test_render_validation_clean():
    s = _sheet()
    res = validate_term_sheet(s)
    out = render_validation(res)
    assert "✅" in out
    assert "halal-compliant" in out


def test_render_validation_dirty_lists_clauses():
    s = _sheet(
        guaranteed=0.05,
        has_liquidation_preference=True,
    )
    res = validate_term_sheet(s)
    out = render_validation(res)
    assert "❌" in out
    assert "guaranteed_return" in out
    assert "liquidation_preference" in out


def test_render_scenario_format():
    s = _sheet()
    out = scenario_payout(s, venture_pnl=1_000_000.0)
    text = render_scenario(out)
    assert "Scenario" in text
    assert "investor" in text
    assert "founder" in text
