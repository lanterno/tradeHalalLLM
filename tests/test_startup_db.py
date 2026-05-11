"""Tests for halal/startup_db.py — Round-5 Wave 6.A."""

from __future__ import annotations

import pytest

from halal_trader.halal.startup_db import (
    Sector,
    Stage,
    StartupDeal,
    Verdict,
    filter_approved,
    is_ambiguous_sector,
    is_haram_sector,
    render_result,
    screen_batch,
    screen_deal,
)


def _deal(
    deal_id: str = "D1",
    company_name: str = "ExampleCo",
    primary_sector: Sector = Sector.HALAL_TECHNOLOGY,
    stage: Stage = Stage.SERIES_B,
    valuation: float = 50_000_000.0,
    raise_amount: float = 10_000_000.0,
    haram_revenue: float = 0.0,
    debt_eq: float = 0.10,
    cash_mc: float = 0.10,
    rec_mc: float = 0.10,
) -> StartupDeal:
    return StartupDeal(
        deal_id=deal_id,
        company_name=company_name,
        primary_sector=primary_sector,
        stage=stage,
        valuation_usd=valuation,
        raise_amount_usd=raise_amount,
        haram_revenue_pct=haram_revenue,
        interest_debt_to_equity=debt_eq,
        cash_to_market_cap=cash_mc,
        receivables_to_market_cap=rec_mc,
    )


# --- Sector helpers -----------------------------------------------------


def test_is_haram_sector_true_for_alcohol():
    assert is_haram_sector(Sector.HARAM_ALCOHOL)


def test_is_haram_sector_false_for_tech():
    assert not is_haram_sector(Sector.HALAL_TECHNOLOGY)


def test_is_ambiguous_sector():
    assert is_ambiguous_sector(Sector.AMBIGUOUS_BIOTECH)
    assert not is_ambiguous_sector(Sector.HALAL_TECHNOLOGY)


# --- StartupDeal validation ---------------------------------------------


def test_deal_valid():
    d = _deal()
    assert d.company_name == "ExampleCo"


def test_deal_empty_id_rejected():
    with pytest.raises(ValueError):
        _deal(deal_id="")


def test_deal_empty_company_rejected():
    with pytest.raises(ValueError):
        _deal(company_name="")


def test_deal_negative_valuation_rejected():
    with pytest.raises(ValueError):
        _deal(valuation=-1.0)


def test_deal_invalid_haram_pct_rejected():
    with pytest.raises(ValueError):
        _deal(haram_revenue=-0.1)


def test_deal_immutable():
    d = _deal()
    with pytest.raises(AttributeError):
        d.valuation_usd = 0  # type: ignore[misc]


# --- Layer 1: sector REJECTED -------------------------------------------


def test_haram_alcohol_rejected():
    """Pin: any haram sector → REJECTED, sticky."""
    d = _deal(primary_sector=Sector.HARAM_ALCOHOL)
    res = screen_deal(d)
    assert res.verdict is Verdict.REJECTED
    assert res.sector_haram


def test_haram_gambling_rejected():
    d = _deal(primary_sector=Sector.HARAM_GAMBLING)
    res = screen_deal(d)
    assert res.verdict is Verdict.REJECTED


def test_haram_conventional_banking_rejected():
    d = _deal(primary_sector=Sector.HARAM_CONVENTIONAL_BANKING)
    res = screen_deal(d)
    assert res.verdict is Verdict.REJECTED


def test_haram_sticky_even_with_clean_finances():
    """Pin: a haram sector with clean revenue + capital structure is
    still REJECTED."""
    d = _deal(
        primary_sector=Sector.HARAM_ALCOHOL,
        haram_revenue=0.0,
        debt_eq=0.0,
        cash_mc=0.0,
        rec_mc=0.0,
    )
    res = screen_deal(d)
    assert res.verdict is Verdict.REJECTED


# --- Layer 2: revenue mix breach ---------------------------------------


def test_revenue_above_5pct_rejected():
    """Pin: AAOIFI 5% hard limit on haram revenue."""
    d = _deal(haram_revenue=0.06)
    res = screen_deal(d)
    assert res.verdict is Verdict.REJECTED
    assert res.revenue_breach


def test_revenue_at_limit_passes():
    d = _deal(haram_revenue=0.05)
    res = screen_deal(d)
    assert res.verdict is Verdict.APPROVED


# --- Layer 3: capital structure ----------------------------------------


def test_debt_above_33_rejected():
    """Pin: AAOIFI 33% debt-to-equity hard limit."""
    d = _deal(debt_eq=0.34)
    res = screen_deal(d)
    assert res.verdict is Verdict.REJECTED
    assert res.capital_breach


def test_debt_in_flag_band_flagged():
    """Pin: 30%-33% is the operator-review band → FLAGGED."""
    d = _deal(debt_eq=0.32)
    res = screen_deal(d)
    assert res.verdict is Verdict.FLAGGED


def test_debt_below_30_passes():
    d = _deal(debt_eq=0.20)
    res = screen_deal(d)
    assert res.verdict is Verdict.APPROVED


def test_cash_above_33_rejected_for_post_revenue():
    """Pin: cash + interest-bearing investments / market cap > 33%
    triggers for post-revenue stages."""
    d = _deal(stage=Stage.GROWTH, cash_mc=0.40)
    res = screen_deal(d)
    assert res.verdict is Verdict.REJECTED


def test_cash_above_33_lenient_for_pre_revenue():
    """Pin: pre-revenue startups (PRE_SEED / SEED / SERIES_A) get
    leniency on cash ratio (noisy metric pre-revenue)."""
    d = _deal(stage=Stage.SEED, cash_mc=0.40)
    res = screen_deal(d)
    assert res.verdict is Verdict.APPROVED


def test_receivables_above_33_rejected_for_post_revenue():
    d = _deal(stage=Stage.GROWTH, rec_mc=0.40)
    res = screen_deal(d)
    assert res.verdict is Verdict.REJECTED


def test_receivables_lenient_for_pre_revenue():
    d = _deal(stage=Stage.SEED, rec_mc=0.40)
    res = screen_deal(d)
    assert res.verdict is Verdict.APPROVED


# --- Ambiguous sector ---------------------------------------------------


def test_ambiguous_biotech_flagged():
    d = _deal(primary_sector=Sector.AMBIGUOUS_BIOTECH)
    res = screen_deal(d)
    assert res.verdict is Verdict.FLAGGED
    assert res.sector_ambiguous


def test_ambiguous_defense_flagged():
    d = _deal(primary_sector=Sector.AMBIGUOUS_DEFENSE)
    res = screen_deal(d)
    assert res.verdict is Verdict.FLAGGED


def test_ambiguous_with_breach_still_rejected():
    """Pin: capital breach takes precedence over ambiguous-flag."""
    d = _deal(
        primary_sector=Sector.AMBIGUOUS_BIOTECH,
        debt_eq=0.50,
    )
    res = screen_deal(d)
    assert res.verdict is Verdict.REJECTED


# --- Clean path ---------------------------------------------------------


def test_clean_deal_approved():
    d = _deal()
    res = screen_deal(d)
    assert res.verdict is Verdict.APPROVED
    assert not res.sector_haram
    assert not res.sector_ambiguous
    assert not res.revenue_breach
    assert not res.capital_breach


def test_approved_includes_clean_reason():
    d = _deal()
    res = screen_deal(d)
    assert any("clean" in r for r in res.reasons)


# --- screen_batch + filter_approved -------------------------------------


def test_screen_batch():
    deals = [
        _deal(deal_id="D1"),
        _deal(deal_id="D2", primary_sector=Sector.HARAM_ALCOHOL),
        _deal(deal_id="D3", debt_eq=0.32),
    ]
    results = screen_batch(deals)
    assert len(results) == 3
    verdicts = [r.verdict for r in results]
    assert verdicts == [Verdict.APPROVED, Verdict.REJECTED, Verdict.FLAGGED]


def test_filter_approved_passes_only_approved():
    deals = [
        _deal(deal_id="D1"),
        _deal(deal_id="D2", primary_sector=Sector.HARAM_GAMBLING),
        _deal(deal_id="D3", debt_eq=0.50),
    ]
    approved = filter_approved(deals)
    assert len(approved) == 1
    assert approved[0].deal_id == "D1"


# --- Render --------------------------------------------------------------


def test_render_approved_emoji():
    d = _deal()
    res = screen_deal(d)
    out = render_result(res)
    assert "✅" in out
    assert "approved" in out


def test_render_rejected_emoji_includes_reason():
    d = _deal(primary_sector=Sector.HARAM_GAMBLING)
    res = screen_deal(d)
    out = render_result(res)
    assert "❌" in out
    assert "haram" in out


def test_render_flagged_emoji():
    d = _deal(primary_sector=Sector.AMBIGUOUS_DEFENSE)
    res = screen_deal(d)
    out = render_result(res)
    assert "🟡" in out
