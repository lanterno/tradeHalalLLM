"""Tests for halal/lp_gp_fund.py — Round-5 Wave 6.G."""

from __future__ import annotations

import pytest

from halal_trader.halal.lp_gp_fund import (
    Fund,
    FundKind,
    FundStatus,
    FundTerms,
    LPCommitment,
    ProhibitedClause,
    annual_management_fee,
    distribute,
    is_halal,
    render_distribution,
    render_fund,
    transition_status,
    validate_clauses,
)


def _terms(
    fund_id: str = "F1",
    kind: FundKind = FundKind.MUDARABAH,
    gp_id: str = "gp-alice",
    gp_capital: float = 0.0,
    gp_profit_share: float = 0.20,
    mgmt_fee: float = 0.02,
    has_hurdle: bool = False,
    has_catch_up: bool = False,
    has_preferred_return: bool = False,
    has_guaranteed_return: bool = False,
) -> FundTerms:
    return FundTerms(
        fund_id=fund_id,
        kind=kind,
        gp_id=gp_id,
        gp_capital_usd=gp_capital,
        gp_profit_share_pct=gp_profit_share,
        base_management_fee_annual_pct=mgmt_fee,
        has_hurdle=has_hurdle,
        has_catch_up=has_catch_up,
        has_preferred_return=has_preferred_return,
        has_guaranteed_return=has_guaranteed_return,
    )


def _lp(
    lp_id: str = "lp-bob",
    committed: float = 1_000_000.0,
    funded: float = 1_000_000.0,
) -> LPCommitment:
    return LPCommitment(
        lp_id=lp_id,
        committed_capital_usd=committed,
        funded_capital_usd=funded,
    )


def _fund(
    terms: FundTerms | None = None,
    lps: tuple[LPCommitment, ...] | None = None,
    status: FundStatus = FundStatus.FORMING,
) -> Fund:
    if terms is None:
        terms = _terms()
    if lps is None:
        lps = (_lp(),)
    return Fund(terms=terms, lps=lps, status=status)


# --- LPCommitment validation --------------------------


def test_lp_valid():
    c = _lp()
    assert c.funded_capital_usd == 1_000_000.0


def test_lp_empty_id_rejected():
    with pytest.raises(ValueError):
        _lp(lp_id="")


def test_lp_negative_committed_rejected():
    with pytest.raises(ValueError):
        _lp(committed=-1.0)


def test_lp_funded_above_committed_rejected():
    with pytest.raises(ValueError):
        _lp(committed=1000.0, funded=2000.0)


# --- FundTerms validation -----------------------------


def test_terms_valid():
    t = _terms()
    assert t.gp_profit_share_pct == 0.20


def test_terms_invalid_profit_share():
    with pytest.raises(ValueError):
        _terms(gp_profit_share=1.0)
    with pytest.raises(ValueError):
        _terms(gp_profit_share=0.0)


def test_terms_mgmt_fee_above_3pct_rejected():
    with pytest.raises(ValueError):
        _terms(mgmt_fee=0.05)


def test_terms_musharakah_zero_gp_capital_rejected():
    with pytest.raises(ValueError):
        _terms(kind=FundKind.MUSHARAKAH, gp_capital=0.0)


def test_terms_immutable():
    t = _terms()
    with pytest.raises(AttributeError):
        t.gp_profit_share_pct = 0.5  # type: ignore[misc]


# --- validate_clauses ------------------------------


def test_clean_terms_no_violations():
    t = _terms()
    assert validate_clauses(t) == ()
    assert is_halal(t)


def test_hurdle_flagged():
    t = _terms(has_hurdle=True)
    v = validate_clauses(t)
    assert ProhibitedClause.HURDLE in v
    assert not is_halal(t)


def test_catch_up_flagged():
    t = _terms(has_catch_up=True)
    assert ProhibitedClause.CATCH_UP in validate_clauses(t)


def test_preferred_return_flagged():
    t = _terms(has_preferred_return=True)
    assert ProhibitedClause.PREFERRED_RETURN in validate_clauses(t)


def test_guaranteed_return_flagged():
    t = _terms(has_guaranteed_return=True)
    assert ProhibitedClause.GUARANTEED_RETURN in validate_clauses(t)


def test_all_violations_returned_in_one_pass():
    t = _terms(
        has_hurdle=True,
        has_catch_up=True,
        has_preferred_return=True,
        has_guaranteed_return=True,
    )
    v = validate_clauses(t)
    assert len(v) == 4


# --- Fund validation -------------------------------


def test_fund_valid():
    f = _fund()
    assert f.status is FundStatus.FORMING


def test_fund_empty_lps_rejected():
    with pytest.raises(ValueError):
        Fund(terms=_terms(), lps=())


def test_fund_duplicate_lp_id_rejected():
    bad = (_lp(lp_id="bob"), _lp(lp_id="bob"))
    with pytest.raises(ValueError):
        Fund(terms=_terms(), lps=bad)


def test_fund_gp_as_lp_rejected():
    bad = (_lp(lp_id="gp-alice"),)
    with pytest.raises(ValueError):
        Fund(terms=_terms(gp_id="gp-alice"), lps=bad)


def test_fund_haram_terms_rejected():
    bad_terms = _terms(has_hurdle=True)
    with pytest.raises(ValueError):
        Fund(terms=bad_terms, lps=(_lp(),))


def test_fund_total_capital():
    f = _fund(
        terms=_terms(kind=FundKind.MUSHARAKAH, gp_capital=500_000.0),
        lps=(_lp(funded=1_000_000.0),),
    )
    assert f.total_capital_usd() == 1_500_000.0


# --- FSM transitions ---------------------------


def test_transition_forming_to_active():
    f = _fund()
    f2 = transition_status(f, new_status=FundStatus.ACTIVE)
    assert f2.status is FundStatus.ACTIVE


def test_transition_active_to_dissolving():
    f = transition_status(_fund(), new_status=FundStatus.ACTIVE)
    f2 = transition_status(f, new_status=FundStatus.DISSOLVING)
    assert f2.status is FundStatus.DISSOLVING


def test_transition_dissolving_to_wound_down():
    f = transition_status(_fund(), new_status=FundStatus.ACTIVE)
    f = transition_status(f, new_status=FundStatus.DISSOLVING)
    f = transition_status(f, new_status=FundStatus.WOUND_DOWN)
    assert f.status is FundStatus.WOUND_DOWN


def test_transition_skip_intermediate_rejected():
    f = _fund()
    with pytest.raises(ValueError):
        transition_status(f, new_status=FundStatus.DISSOLVING)


def test_transition_wound_down_terminal():
    f = transition_status(_fund(), new_status=FundStatus.ACTIVE)
    f = transition_status(f, new_status=FundStatus.DISSOLVING)
    f = transition_status(f, new_status=FundStatus.WOUND_DOWN)
    with pytest.raises(ValueError):
        transition_status(f, new_status=FundStatus.ACTIVE)


# --- distribute — profit branch ----------------


def test_distribute_profit_mudarabah():
    """Pin: GP gets gp_profit_share; LPs split rest pro-rata."""
    f = _fund(
        terms=_terms(gp_profit_share=0.20),
        lps=(_lp(lp_id="lp-bob", funded=600_000.0), _lp(lp_id="lp-cat", funded=400_000.0)),
    )
    records = distribute(f, period_pnl=100_000.0)
    by_party = {r.party_id: r for r in records}
    # GP: 20% × 100k = 20k.
    assert by_party["gp-alice"].proceeds == pytest.approx(20_000.0)
    # LPs: split 80k pro-rata 60/40.
    assert by_party["lp-bob"].proceeds == pytest.approx(48_000.0)
    assert by_party["lp-cat"].proceeds == pytest.approx(32_000.0)


def test_distribute_zero_pnl():
    f = _fund()
    records = distribute(f, period_pnl=0.0)
    for r in records:
        assert r.proceeds == 0.0


# --- distribute — loss branch ----------------


def test_distribute_mudarabah_loss_lps_only():
    """Pin: Mudarabah loss → LPs absorb, GP gets 0."""
    f = _fund(
        lps=(_lp(lp_id="lp-bob", funded=600_000.0), _lp(lp_id="lp-cat", funded=400_000.0)),
    )
    records = distribute(f, period_pnl=-100_000.0)
    by_party = {r.party_id: r for r in records}
    assert by_party["gp-alice"].proceeds == 0.0
    assert by_party["lp-bob"].proceeds == pytest.approx(-60_000.0)
    assert by_party["lp-cat"].proceeds == pytest.approx(-40_000.0)


def test_distribute_musharakah_loss_pro_rata_all_capital():
    """Pin: Musharakah loss → all capital absorbs pro-rata."""
    f = _fund(
        terms=_terms(kind=FundKind.MUSHARAKAH, gp_capital=500_000.0),
        lps=(_lp(lp_id="lp-bob", funded=500_000.0),),
    )
    # 50/50 capital split → 50/50 loss.
    records = distribute(f, period_pnl=-100_000.0)
    by_party = {r.party_id: r for r in records}
    assert by_party["gp-alice"].proceeds == pytest.approx(-50_000.0)
    assert by_party["lp-bob"].proceeds == pytest.approx(-50_000.0)


def test_distribute_no_funded_rejected():
    f = _fund(lps=(_lp(funded=0.0),))
    with pytest.raises(ValueError):
        distribute(f, period_pnl=100_000.0)


# --- annual_management_fee ----------------


def test_mgmt_fee_full_year():
    f = _fund(terms=_terms(mgmt_fee=0.02))
    fee = annual_management_fee(f, aum_usd=10_000_000.0, days=365)
    # 2% × 10M = 200k.
    assert fee == pytest.approx(200_000.0)


def test_mgmt_fee_prorated():
    f = _fund(terms=_terms(mgmt_fee=0.02))
    fee = annual_management_fee(f, aum_usd=10_000_000.0, days=180)
    # 2% × 10M × 180/365 ≈ 98,630.
    assert fee == pytest.approx(98_630.137, rel=0.001)


def test_mgmt_fee_negative_aum_rejected():
    f = _fund()
    with pytest.raises(ValueError):
        annual_management_fee(f, aum_usd=-1.0)


def test_mgmt_fee_zero_days_rejected():
    f = _fund()
    with pytest.raises(ValueError):
        annual_management_fee(f, aum_usd=10_000_000.0, days=0)


# --- Render ---------------------------------


def test_render_fund_no_secret_leak():
    f = _fund(terms=_terms(gp_id="gp-alice@example.com"))
    out = render_fund(f)
    assert "gp-alice@example.com" not in out


def test_render_fund_status_emoji():
    f = _fund()
    out = render_fund(f)
    assert "🌱" in out


def test_render_distribution_format():
    f = _fund()
    records = distribute(f, period_pnl=100_000.0)
    out = render_distribution(records)
    assert "💸" in out
    assert "🎯" in out  # GP
    assert "👥" in out  # LP


def test_render_distribution_empty():
    out = render_distribution([])
    assert "No distribution" in out
