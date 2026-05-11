"""Tests for halal/spac_screen.py — Round-5 Wave 6.E."""

from __future__ import annotations

import pytest

from halal_trader.halal.spac_screen import (
    SectorRestriction,
    SPACProfile,
    SponsorPromoteKind,
    TrustHolding,
    Verdict,
    filter_approved,
    render_result,
    screen_batch,
    screen_spac,
)


def _profile(
    spac_id: str = "SP1",
    ticker: str = "HALAL",
    trust_holding: TrustHolding = TrustHolding.HALAL_MMF,
    sponsor_promote_kind: SponsorPromoteKind = SponsorPromoteKind.MUDARABAH_SHARE,
    sponsor_promote_pct: float = 0.18,
    target_sector: SectorRestriction = SectorRestriction.HALAL,
    redemption_returns_interest: bool = False,
    redemption_at_issue_or_better: bool = True,
    target_announced: bool = False,
) -> SPACProfile:
    return SPACProfile(
        spac_id=spac_id,
        ticker=ticker,
        trust_holding=trust_holding,
        sponsor_promote_kind=sponsor_promote_kind,
        sponsor_promote_pct=sponsor_promote_pct,
        target_sector=target_sector,
        redemption_returns_interest=redemption_returns_interest,
        redemption_at_issue_or_better=redemption_at_issue_or_better,
        target_announced=target_announced,
    )


# --- SPACProfile validation --------------------------------


def test_profile_valid():
    p = _profile()
    assert p.spac_id == "SP1"


def test_profile_empty_id_rejected():
    with pytest.raises(ValueError):
        _profile(spac_id="")


def test_profile_empty_ticker_rejected():
    with pytest.raises(ValueError):
        _profile(ticker="")


def test_profile_excessive_promote_rejected():
    with pytest.raises(ValueError):
        _profile(sponsor_promote_pct=0.75)


def test_profile_immutable():
    p = _profile()
    with pytest.raises(AttributeError):
        p.sponsor_promote_pct = 0.5  # type: ignore[misc]


# --- screen — clean path -------------------------------


def test_clean_spac_approved():
    p = _profile()
    r = screen_spac(p)
    assert r.verdict is Verdict.APPROVED
    assert not r.failures
    assert not r.flags


# --- screen — trust holding ----------------------------


def test_riba_tbills_rejected():
    p = _profile(trust_holding=TrustHolding.RIBA_BEARING_TBILLS)
    r = screen_spac(p)
    assert r.verdict is Verdict.REJECTED
    assert any("riba" in f.lower() for f in r.failures)


def test_halal_mmf_accepted():
    p = _profile(trust_holding=TrustHolding.HALAL_MMF)
    r = screen_spac(p)
    assert r.verdict is Verdict.APPROVED


def test_sukuk_basket_accepted():
    p = _profile(trust_holding=TrustHolding.SUKUK_BASKET)
    r = screen_spac(p)
    assert r.verdict is Verdict.APPROVED


def test_cash_no_interest_accepted():
    p = _profile(trust_holding=TrustHolding.CASH_NO_INTEREST)
    r = screen_spac(p)
    assert r.verdict is Verdict.APPROVED


# --- screen — sector -----------------------------------


def test_haram_sector_rejected():
    p = _profile(target_sector=SectorRestriction.HARAM)
    r = screen_spac(p)
    assert r.verdict is Verdict.REJECTED


def test_ambiguous_sector_flagged():
    p = _profile(target_sector=SectorRestriction.AMBIGUOUS)
    r = screen_spac(p)
    assert r.verdict is Verdict.FLAGGED


def test_unknown_sector_flagged():
    p = _profile(target_sector=SectorRestriction.UNKNOWN)
    r = screen_spac(p)
    assert r.verdict is Verdict.FLAGGED


# --- screen — sponsor promote --------------------------


def test_free_warrant_rejected():
    p = _profile(sponsor_promote_kind=SponsorPromoteKind.FREE_WARRANT)
    r = screen_spac(p)
    assert r.verdict is Verdict.REJECTED
    assert any("FREE_WARRANT" in f for f in r.failures)


def test_capital_proportional_accepted():
    p = _profile(sponsor_promote_kind=SponsorPromoteKind.CAPITAL_PROPORTIONAL)
    r = screen_spac(p)
    assert r.verdict is Verdict.APPROVED


def test_promote_pct_above_cap_rejected():
    p = _profile(sponsor_promote_pct=0.30)
    r = screen_spac(p)
    assert r.verdict is Verdict.REJECTED


def test_promote_pct_in_flag_band_flagged():
    """Pin: 0.20-0.25 → FLAGGED."""
    p = _profile(sponsor_promote_pct=0.22)
    r = screen_spac(p)
    assert r.verdict is Verdict.FLAGGED


def test_promote_pct_below_flag_band_approved():
    p = _profile(sponsor_promote_pct=0.15)
    r = screen_spac(p)
    assert r.verdict is Verdict.APPROVED


def test_promote_pct_at_cap_approved():
    """Pin: at the flag threshold exactly, no flag fires."""
    p = _profile(sponsor_promote_pct=0.20)
    r = screen_spac(p)
    assert r.verdict is Verdict.APPROVED


# --- screen — redemption --------------------------------


def test_redemption_returns_interest_rejected():
    p = _profile(redemption_returns_interest=True)
    r = screen_spac(p)
    assert r.verdict is Verdict.REJECTED


def test_redemption_below_issue_rejected():
    p = _profile(redemption_at_issue_or_better=False)
    r = screen_spac(p)
    assert r.verdict is Verdict.REJECTED


# --- screen — combined ---------------------------------


def test_combined_failures_all_captured():
    p = _profile(
        trust_holding=TrustHolding.RIBA_BEARING_TBILLS,
        sponsor_promote_kind=SponsorPromoteKind.FREE_WARRANT,
        target_sector=SectorRestriction.HARAM,
        redemption_returns_interest=True,
    )
    r = screen_spac(p)
    assert r.verdict is Verdict.REJECTED
    assert len(r.failures) >= 4


def test_failure_dominates_flag():
    """Pin: any failure flips to REJECTED even with FLAG signals."""
    p = _profile(
        trust_holding=TrustHolding.RIBA_BEARING_TBILLS,
        target_sector=SectorRestriction.AMBIGUOUS,  # flag
    )
    r = screen_spac(p)
    assert r.verdict is Verdict.REJECTED


# --- screen — policy overrides --------------------------


def test_screen_invalid_cap_rejected():
    p = _profile()
    with pytest.raises(ValueError):
        screen_spac(p, max_sponsor_promote_pct=0.0)


def test_screen_flag_band_above_cap_rejected():
    p = _profile()
    with pytest.raises(ValueError):
        screen_spac(p, max_sponsor_promote_pct=0.10, flag_sponsor_promote_pct=0.20)


def test_screen_custom_cap_changes_verdict():
    p = _profile(sponsor_promote_pct=0.15)
    r = screen_spac(p, max_sponsor_promote_pct=0.10, flag_sponsor_promote_pct=0.05)
    assert r.verdict is Verdict.REJECTED


# --- screen_batch / filter_approved -------------------


def test_screen_batch_per_profile():
    ps = [
        _profile(spac_id="SP1"),
        _profile(spac_id="SP2", target_sector=SectorRestriction.HARAM),
        _profile(spac_id="SP3", sponsor_promote_pct=0.22),
    ]
    out = screen_batch(ps)
    by_id = {r.spac_id: r for r in out}
    assert by_id["SP1"].verdict is Verdict.APPROVED
    assert by_id["SP2"].verdict is Verdict.REJECTED
    assert by_id["SP3"].verdict is Verdict.FLAGGED


def test_filter_approved():
    ps = [
        _profile(spac_id="SP1"),
        _profile(spac_id="SP2", target_sector=SectorRestriction.HARAM),
    ]
    approved = filter_approved(ps)
    assert len(approved) == 1
    assert approved[0].spac_id == "SP1"


# --- Render -----------------------------------------


def test_render_approved_emoji():
    r = screen_spac(_profile())
    out = render_result(r)
    assert "✅" in out


def test_render_rejected_lists_failures():
    r = screen_spac(_profile(trust_holding=TrustHolding.RIBA_BEARING_TBILLS))
    out = render_result(r)
    assert "❌" in out
    assert "riba" in out.lower()


def test_render_flagged_emoji():
    r = screen_spac(_profile(target_sector=SectorRestriction.AMBIGUOUS))
    out = render_result(r)
    assert "🟡" in out
