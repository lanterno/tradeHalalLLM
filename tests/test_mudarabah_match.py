"""Tests for halal/mudarabah_match.py — Round-5 Wave 17.H."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.halal.mudarabah_match import (
    CapitalListing,
    MatchRound,
    RiskTolerance,
    SkillListing,
    can_match,
    is_risk_compatible,
    profit_share_overlap,
    render_round,
    run_match_round,
)


def _cap(
    listing_id: str = "C1",
    user: str = "alice",
    amount: float = 10_000.0,
    risk: RiskTolerance = RiskTolerance.BALANCED,
    min_share: float = 0.50,
    listed_at: date = date(2026, 6, 1),
    horizon: int = 12,
) -> CapitalListing:
    return CapitalListing(
        listing_id=listing_id,
        user_id=user,
        amount_usd=amount,
        risk_tolerance=risk,
        min_profit_share_for_capital=min_share,
        listed_at=listed_at,
        horizon_months=horizon,
    )


def _skill(
    listing_id: str = "S1",
    user: str = "bob",
    capacity: float = 50_000.0,
    risk: RiskTolerance = RiskTolerance.BALANCED,
    max_share: float = 0.70,
    listed_at: date = date(2026, 6, 1),
    track: float = 0.5,
    horizon: int = 12,
) -> SkillListing:
    return SkillListing(
        listing_id=listing_id,
        user_id=user,
        capital_capacity_usd=capacity,
        risk_tolerance=risk,
        max_profit_share_for_capital=max_share,
        listed_at=listed_at,
        track_record_score=track,
        horizon_months=horizon,
    )


# --- Listing validation -------------------------------------------------


def test_capital_valid():
    c = _cap()
    assert c.amount_usd == 10_000.0


def test_capital_empty_id_rejected():
    with pytest.raises(ValueError):
        _cap(listing_id="")


def test_capital_negative_amount_rejected():
    with pytest.raises(ValueError):
        _cap(amount=-1.0)


def test_capital_share_at_one_rejected():
    with pytest.raises(ValueError):
        _cap(min_share=1.0)


def test_skill_valid():
    s = _skill()
    assert s.capital_capacity_usd == 50_000.0


def test_skill_invalid_track_rejected():
    with pytest.raises(ValueError):
        _skill(track=1.5)


def test_listings_immutable():
    c = _cap()
    with pytest.raises(AttributeError):
        c.amount_usd = 0  # type: ignore[misc]


# --- is_risk_compatible -------------------------------------------------


def test_risk_compatible_same_tier():
    assert is_risk_compatible(RiskTolerance.CONSERVATIVE, RiskTolerance.CONSERVATIVE)
    assert is_risk_compatible(RiskTolerance.BALANCED, RiskTolerance.BALANCED)
    assert is_risk_compatible(RiskTolerance.AGGRESSIVE, RiskTolerance.AGGRESSIVE)


def test_risk_aggressive_capital_can_take_lower_skill():
    """Pin: higher-tolerance capital tolerates lower-risk managers."""
    assert is_risk_compatible(RiskTolerance.AGGRESSIVE, RiskTolerance.BALANCED)
    assert is_risk_compatible(RiskTolerance.AGGRESSIVE, RiskTolerance.CONSERVATIVE)


def test_risk_conservative_capital_rejects_higher():
    assert not is_risk_compatible(RiskTolerance.CONSERVATIVE, RiskTolerance.AGGRESSIVE)
    assert not is_risk_compatible(RiskTolerance.CONSERVATIVE, RiskTolerance.BALANCED)


def test_risk_balanced_capital_rejects_aggressive():
    assert not is_risk_compatible(RiskTolerance.BALANCED, RiskTolerance.AGGRESSIVE)


# --- profit_share_overlap -----------------------------------------------


def test_profit_share_overlap_midpoint():
    """Pin: overlap → midpoint of [min, max]."""
    cap = _cap(min_share=0.50)
    skill = _skill(max_share=0.70)
    share = profit_share_overlap(cap, skill)
    assert share == pytest.approx(0.60)


def test_profit_share_no_overlap():
    """Cap wants ≥ 0.80, skill caps at 0.50 → no overlap."""
    cap = _cap(min_share=0.80)
    skill = _skill(max_share=0.50)
    assert profit_share_overlap(cap, skill) is None


def test_profit_share_too_narrow_overlap_no_match():
    """Pin: < 5pp overlap → no match."""
    cap = _cap(min_share=0.55)
    skill = _skill(max_share=0.58)  # 3pp overlap
    assert profit_share_overlap(cap, skill) is None


def test_profit_share_at_5pp_overlap_matches():
    cap = _cap(min_share=0.50)
    skill = _skill(max_share=0.55)  # exactly 5pp
    share = profit_share_overlap(cap, skill)
    assert share is not None


# --- can_match ---------------------------------------------------------


def test_can_match_happy_path():
    assert can_match(_cap(), _skill())


def test_can_match_risk_incompatible():
    assert not can_match(
        _cap(risk=RiskTolerance.CONSERVATIVE),
        _skill(risk=RiskTolerance.AGGRESSIVE),
    )


def test_can_match_share_overlap_fails():
    assert not can_match(
        _cap(min_share=0.80),
        _skill(max_share=0.50),
    )


def test_can_match_capital_below_min():
    assert not can_match(_cap(amount=100), _skill(), min_match_amount=500)


def test_can_match_skill_capacity_below_min():
    assert not can_match(_cap(amount=10_000), _skill(capacity=100), min_match_amount=500)


def test_can_match_cap_horizon_above_skill():
    """Pin: capital wants 24mo but skill only commits 12mo → no match."""
    assert not can_match(_cap(horizon=24), _skill(horizon=12))


# --- run_match_round — happy path ---------------------------------------


def test_run_match_simple_pair():
    caps = [_cap(amount=10_000)]
    skills = [_skill(capacity=50_000)]
    out = run_match_round(caps, skills)
    assert len(out.matches) == 1
    assert out.matches[0].matched_amount_usd == 10_000.0


def test_match_share_is_negotiated_midpoint():
    caps = [_cap(min_share=0.50)]
    skills = [_skill(max_share=0.70)]
    out = run_match_round(caps, skills)
    assert out.matches[0].agreed_profit_share_capital == pytest.approx(0.60)


def test_match_horizon_takes_min():
    caps = [_cap(horizon=12)]
    skills = [_skill(horizon=24)]
    out = run_match_round(caps, skills)
    assert out.matches[0].horizon_months == 12


# --- run_match_round — capacity-aware ----------------------------------


def test_match_skill_capacity_consumed():
    """Skill capacity 50k matches three 20k cap listings, last one
    only gets 10k available — too low for min_match → unmatched."""
    caps = [
        _cap(listing_id="C1", user="alice", amount=20_000, listed_at=date(2026, 6, 1)),
        _cap(listing_id="C2", user="bob", amount=20_000, listed_at=date(2026, 6, 2)),
        _cap(listing_id="C3", user="charlie", amount=20_000, listed_at=date(2026, 6, 3)),
    ]
    skills = [_skill(capacity=50_000)]
    out = run_match_round(caps, skills)
    # C1 → 20k; C2 → 20k; C3 → 10k remaining capacity (but capital
    # is 20k → matched_amount = min(20k, 10k) = 10k, which exceeds
    # min_match=500). All three should match.
    assert len(out.matches) == 3
    assert sum(m.matched_amount_usd for m in out.matches) == 50_000.0


def test_match_capacity_below_min_excluded():
    caps = [_cap(amount=10_000)]
    skills = [_skill(capacity=400)]  # below min_match=500
    out = run_match_round(caps, skills)
    assert not out.matches


# --- run_match_round — FIFO --------------------------------------------


def test_match_fifo_capital_order():
    """Pin: oldest capital listing matched first."""
    caps = [
        _cap(listing_id="C-old", amount=10_000, listed_at=date(2026, 5, 1)),
        _cap(listing_id="C-new", amount=10_000, listed_at=date(2026, 6, 1)),
    ]
    skills = [_skill(capacity=10_000)]  # only enough for one
    out = run_match_round(caps, skills)
    assert len(out.matches) == 1
    assert out.matches[0].capital_listing_id == "C-old"


def test_match_fifo_skill_order():
    """Pin: oldest skill listing matched first when both fit."""
    caps = [_cap(amount=10_000)]
    skills = [
        _skill(listing_id="S-new", capacity=50_000, listed_at=date(2026, 6, 1)),
        _skill(listing_id="S-old", capacity=50_000, listed_at=date(2026, 5, 1)),
    ]
    out = run_match_round(caps, skills)
    assert out.matches[0].skill_listing_id == "S-old"


# --- run_match_round — risk filter -------------------------------------


def test_run_match_risk_incompatible_excluded():
    caps = [_cap(risk=RiskTolerance.CONSERVATIVE)]
    skills = [_skill(risk=RiskTolerance.AGGRESSIVE)]
    out = run_match_round(caps, skills)
    assert not out.matches
    assert len(out.unmatched_capital) == 1
    assert len(out.unmatched_skill) == 1


# --- run_match_round — empty + no match --------------------------------


def test_match_empty_inputs():
    out = run_match_round([], [])
    assert not out.matches
    assert not out.unmatched_capital
    assert not out.unmatched_skill


def test_match_no_overlap_unmatched():
    caps = [_cap(min_share=0.80)]
    skills = [_skill(max_share=0.50)]
    out = run_match_round(caps, skills)
    assert not out.matches
    assert len(out.unmatched_capital) == 1


# --- Render --------------------------------------------------------------


def test_render_no_secret_leak():
    caps = [_cap(user="alice@example.com")]
    skills = [_skill(user="bob@example.com")]
    out = run_match_round(caps, skills)
    text = render_round(out)
    assert "alice@example.com" not in text
    assert "bob@example.com" not in text


def test_render_empty_round():
    out = render_round(
        MatchRound(
            matches=tuple(),
            unmatched_capital=tuple(),
            unmatched_skill=tuple(),
        )
    )
    assert "0 matched" in out


def test_render_includes_share_pct():
    caps = [_cap(min_share=0.50)]
    skills = [_skill(max_share=0.70)]
    out = run_match_round(caps, skills)
    text = render_round(out)
    # Share should be midpoint = 60%; rendering shows both legs.
    assert "60.00%" in text
    assert "40.00%" in text
