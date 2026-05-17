"""Tests for the halal-VC allocation gate."""

from __future__ import annotations

import dataclasses

import pytest

from halal_trader.web.halal_vc import (
    DEFAULT_POLICY,
    DealStage,
    FounderShariahCompliance,
    HalalSector,
    UseOfProceeds,
    VCAllocationPolicy,
    VCAllocationRequest,
    VCDeal,
    VCDealVerdict,
    evaluate_allocation,
    render_allocation_decision,
    render_screen_result,
    screen_deal,
)


def _deal(
    *,
    deal_id: str = "DEAL-001",
    company_name: str = "Halal SaaS Co.",
    sector: HalalSector = HalalSector.SAAS_B2B,
    stage: DealStage = DealStage.SERIES_A,
    use_of_proceeds: UseOfProceeds = UseOfProceeds.PRODUCT_DEVELOPMENT,
    founder_shariah_compliance: FounderShariahCompliance = (
        FounderShariahCompliance.SCHOLAR_BOARD_BACKED
    ),
    lockup_years: int = 7,
    minimum_check_usd: float = 25_000.0,
    has_scholar_board_review: bool = True,
) -> VCDeal:
    return VCDeal(
        deal_id=deal_id,
        company_name=company_name,
        sector=sector,
        stage=stage,
        use_of_proceeds=use_of_proceeds,
        founder_shariah_compliance=founder_shariah_compliance,
        lockup_years=lockup_years,
        minimum_check_usd=minimum_check_usd,
        has_scholar_board_review=has_scholar_board_review,
    )


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------


def test_default_policy_values() -> None:
    p = DEFAULT_POLICY
    assert p.max_per_deal_pct == 10.0
    assert p.min_lockup_disclosure_years == 0
    assert p.require_scholar_board_for_pre_product is True


def test_policy_rejects_zero_per_deal_cap() -> None:
    with pytest.raises(ValueError, match="max_per_deal_pct"):
        VCAllocationPolicy(max_per_deal_pct=0)


def test_policy_rejects_above_50_per_deal_cap() -> None:
    """Pin: > 50% is a category error (half a portfolio in one deal)."""

    with pytest.raises(ValueError, match="max_per_deal_pct"):
        VCAllocationPolicy(max_per_deal_pct=51.0)


def test_policy_accepts_25_per_deal_cap() -> None:
    """High-conviction operators bump cap to 25%."""

    p = VCAllocationPolicy(max_per_deal_pct=25.0)
    assert p.max_per_deal_pct == 25.0


def test_policy_rejects_negative_lockup_disclosure() -> None:
    with pytest.raises(ValueError, match="min_lockup_disclosure_years"):
        VCAllocationPolicy(min_lockup_disclosure_years=-1)


# ---------------------------------------------------------------------------
# VCDeal validation
# ---------------------------------------------------------------------------


def test_deal_rejects_empty_deal_id() -> None:
    with pytest.raises(ValueError, match="deal_id"):
        _deal(deal_id="")


def test_deal_rejects_empty_company_name() -> None:
    with pytest.raises(ValueError, match="company_name"):
        _deal(company_name="")


def test_deal_rejects_negative_lockup() -> None:
    with pytest.raises(ValueError, match="lockup_years"):
        _deal(lockup_years=-1)


def test_deal_rejects_negative_minimum_check() -> None:
    with pytest.raises(ValueError, match="minimum_check_usd"):
        _deal(minimum_check_usd=-1.0)


def test_deal_accepts_zero_lockup() -> None:
    """0-year lockup is rare but valid (liquid secondaries)."""

    d = _deal(lockup_years=0)
    assert d.lockup_years == 0


# ---------------------------------------------------------------------------
# Hard rejections — unconditional NOT_HALAL gates
# ---------------------------------------------------------------------------


def test_retire_debt_use_of_proceeds_is_not_halal() -> None:
    """Pin: raising equity to retire interest-bearing debt funds riba."""

    deal = _deal(use_of_proceeds=UseOfProceeds.RETIRE_DEBT)
    result = screen_deal(deal)
    assert result.verdict is VCDealVerdict.NOT_HALAL
    assert any("riba" in f for f in result.failures)


def test_retire_debt_overrides_other_clean_flags() -> None:
    """Pin: even with everything else clean, RETIRE_DEBT → NOT_HALAL."""

    deal = _deal(
        sector=HalalSector.HEALTHCARE,
        stage=DealStage.SERIES_C,
        use_of_proceeds=UseOfProceeds.RETIRE_DEBT,
        founder_shariah_compliance=FounderShariahCompliance.SCHOLAR_BOARD_BACKED,
        has_scholar_board_review=True,
    )
    result = screen_deal(deal)
    assert result.verdict is VCDealVerdict.NOT_HALAL


# ---------------------------------------------------------------------------
# INSUFFICIENT_DATA — undisclosed proceeds
# ---------------------------------------------------------------------------


def test_undisclosed_proceeds_returns_insufficient_data() -> None:
    deal = _deal(use_of_proceeds=UseOfProceeds.UNDISCLOSED)
    result = screen_deal(deal)
    assert result.verdict is VCDealVerdict.INSUFFICIENT_DATA
    assert any("UNDISCLOSED" in w for w in result.warnings)


def test_undisclosed_proceeds_returns_insufficient_data_even_with_clean_flags() -> None:
    deal = _deal(
        sector=HalalSector.HEALTHCARE,
        stage=DealStage.SERIES_A,
        use_of_proceeds=UseOfProceeds.UNDISCLOSED,
        founder_shariah_compliance=FounderShariahCompliance.SCHOLAR_BOARD_BACKED,
        has_scholar_board_review=True,
    )
    result = screen_deal(deal)
    assert result.verdict is VCDealVerdict.INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# HALAL — every check passes
# ---------------------------------------------------------------------------


def test_clean_series_a_saas_is_halal() -> None:
    """Default fixture: series A SaaS, scholar-board-backed founder, scholar-reviewed."""

    result = screen_deal(_deal())
    assert result.verdict is VCDealVerdict.HALAL
    assert result.failures == ()
    assert result.warnings == ()


def test_clean_growth_stage_healthcare_is_halal() -> None:
    deal = _deal(
        sector=HalalSector.HEALTHCARE,
        stage=DealStage.GROWTH,
    )
    result = screen_deal(deal)
    assert result.verdict is VCDealVerdict.HALAL


def test_clean_education_series_b_is_halal() -> None:
    deal = _deal(
        sector=HalalSector.EDUCATION,
        stage=DealStage.SERIES_B,
    )
    result = screen_deal(deal)
    assert result.verdict is VCDealVerdict.HALAL


# ---------------------------------------------------------------------------
# DOUBTFUL_PIVOT — pre-product startups
# ---------------------------------------------------------------------------


def test_pre_seed_with_scholar_board_is_doubtful_pivot() -> None:
    """Pin: pre-product startups frequently pivot → DOUBTFUL_PIVOT
    even with scholar-board backing."""

    deal = _deal(
        stage=DealStage.PRE_SEED,
        has_scholar_board_review=True,
    )
    result = screen_deal(deal)
    assert result.verdict is VCDealVerdict.DOUBTFUL_PIVOT
    assert any("pivot" in w for w in result.warnings)


def test_seed_without_scholar_board_is_doubtful_pivot_with_strict_warning() -> None:
    deal = _deal(
        stage=DealStage.SEED,
        has_scholar_board_review=False,
    )
    result = screen_deal(deal)
    assert result.verdict is VCDealVerdict.DOUBTFUL_PIVOT
    assert any("scholar-board" in w for w in result.warnings)


def test_series_a_does_not_trigger_pivot_warning() -> None:
    """Pin: Series A onward is not in the pre-product set."""

    deal = _deal(stage=DealStage.SERIES_A)
    result = screen_deal(deal)
    # Should be HALAL not DOUBTFUL_PIVOT
    assert result.verdict is VCDealVerdict.HALAL


# ---------------------------------------------------------------------------
# DOUBTFUL — soft warnings drive doubtful (non-pre-product)
# ---------------------------------------------------------------------------


def test_scrutiny_sector_is_doubtful() -> None:
    """Pin: sectors adjacent to non-halal verticals (halal-fintech /
    halal-food / modest-fashion) trigger DOUBTFUL warning."""

    deal = _deal(sector=HalalSector.HALAL_FINTECH)
    result = screen_deal(deal)
    assert result.verdict is VCDealVerdict.DOUBTFUL
    assert any("scrutiny" in w for w in result.warnings)


def test_unknown_founder_compliance_is_doubtful() -> None:
    deal = _deal(founder_shariah_compliance=FounderShariahCompliance.UNKNOWN)
    result = screen_deal(deal)
    assert result.verdict is VCDealVerdict.DOUBTFUL
    assert any("UNKNOWN" in w for w in result.warnings)


def test_self_declared_founder_is_doubtful() -> None:
    """Self-declared halal without scholar backing → DOUBTFUL."""

    deal = _deal(founder_shariah_compliance=FounderShariahCompliance.SELF_DECLARED_HALAL)
    result = screen_deal(deal)
    assert result.verdict is VCDealVerdict.DOUBTFUL
    assert any("self-declared" in w for w in result.warnings)


def test_no_scholar_board_review_is_doubtful() -> None:
    deal = _deal(has_scholar_board_review=False)
    result = screen_deal(deal)
    assert result.verdict is VCDealVerdict.DOUBTFUL
    assert any("scholar-board" in w.lower() for w in result.warnings)


def test_multiple_doubtful_signals_aggregate() -> None:
    deal = _deal(
        sector=HalalSector.HALAL_FINTECH,
        founder_shariah_compliance=FounderShariahCompliance.SELF_DECLARED_HALAL,
        has_scholar_board_review=False,
    )
    result = screen_deal(deal)
    assert result.verdict is VCDealVerdict.DOUBTFUL
    assert len(result.warnings) >= 3


# ---------------------------------------------------------------------------
# Closed sector enum
# ---------------------------------------------------------------------------


def test_closed_sector_set_excludes_non_halal() -> None:
    """Pin: no `alcohol` / `gambling` / `tobacco` / `weapons` /
    `adult` / `pork` / `conventional_banking` in sector enum."""

    values = {s.value for s in HalalSector}
    forbidden = {
        "alcohol",
        "gambling",
        "tobacco",
        "weapons",
        "adult",
        "pork",
        "conventional_banking",
    }
    assert values & forbidden == set()


def test_sector_string_values_pinned() -> None:
    assert HalalSector.HEALTHCARE.value == "healthcare"
    assert HalalSector.HALAL_FINTECH.value == "halal_fintech"
    assert HalalSector.MODEST_FASHION.value == "modest_fashion"


# ---------------------------------------------------------------------------
# Allocation evaluation
# ---------------------------------------------------------------------------


def test_clean_within_cap_request_is_allowed() -> None:
    deal = _deal()
    request = VCAllocationRequest(user_id="user-1", deal_id="DEAL-001", requested_pct=5.0)
    decision = evaluate_allocation(request, deal=deal)
    assert decision.allowed is True
    assert decision.deal_verdict is VCDealVerdict.HALAL
    assert decision.requested_pct == 5.0
    assert decision.cap_pct == 10.0


def test_request_at_cap_is_allowed() -> None:
    """Pin: exactly at the cap is allowed (boundary inclusive)."""

    deal = _deal()
    request = VCAllocationRequest(user_id="user-1", deal_id="DEAL-001", requested_pct=10.0)
    decision = evaluate_allocation(request, deal=deal)
    assert decision.allowed is True


def test_request_above_cap_is_blocked() -> None:
    deal = _deal()
    request = VCAllocationRequest(user_id="user-1", deal_id="DEAL-001", requested_pct=15.0)
    decision = evaluate_allocation(request, deal=deal)
    assert decision.allowed is False
    assert "exceeds" in decision.reason


def test_not_halal_deal_blocks_regardless_of_pct() -> None:
    """Pin: hard NOT_HALAL verdict blocks even tiny allocations."""

    deal = _deal(use_of_proceeds=UseOfProceeds.RETIRE_DEBT)
    request = VCAllocationRequest(user_id="user-1", deal_id="DEAL-001", requested_pct=1.0)
    decision = evaluate_allocation(request, deal=deal)
    assert decision.allowed is False
    assert decision.deal_verdict is VCDealVerdict.NOT_HALAL


def test_insufficient_data_blocks_allocation() -> None:
    deal = _deal(use_of_proceeds=UseOfProceeds.UNDISCLOSED)
    request = VCAllocationRequest(user_id="user-1", deal_id="DEAL-001", requested_pct=5.0)
    decision = evaluate_allocation(request, deal=deal)
    assert decision.allowed is False
    assert decision.deal_verdict is VCDealVerdict.INSUFFICIENT_DATA


def test_doubtful_within_cap_is_allowed_with_warning() -> None:
    """Pin: DOUBTFUL deals proceed but verdict carried in decision."""

    deal = _deal(sector=HalalSector.HALAL_FINTECH)
    request = VCAllocationRequest(user_id="user-1", deal_id="DEAL-001", requested_pct=5.0)
    decision = evaluate_allocation(request, deal=deal)
    assert decision.allowed is True
    assert decision.deal_verdict is VCDealVerdict.DOUBTFUL


def test_doubtful_pivot_within_cap_is_allowed() -> None:
    deal = _deal(stage=DealStage.SEED)
    request = VCAllocationRequest(user_id="user-1", deal_id="DEAL-001", requested_pct=3.0)
    decision = evaluate_allocation(request, deal=deal)
    assert decision.allowed is True
    assert decision.deal_verdict is VCDealVerdict.DOUBTFUL_PIVOT


def test_evaluate_rejects_mismatched_deal_id() -> None:
    deal = _deal(deal_id="DEAL-001")
    request = VCAllocationRequest(user_id="user-1", deal_id="DEAL-002", requested_pct=5.0)
    with pytest.raises(ValueError, match="does not match"):
        evaluate_allocation(request, deal=deal)


def test_strict_5pct_cap_blocks_8pct_request() -> None:
    """Custom cap flow: a 5% per-deal cap blocks an 8% request."""

    strict = VCAllocationPolicy(max_per_deal_pct=5.0)
    deal = _deal()
    request = VCAllocationRequest(user_id="user-1", deal_id="DEAL-001", requested_pct=8.0)
    decision = evaluate_allocation(request, deal=deal, policy=strict)
    assert decision.allowed is False


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def test_request_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        VCAllocationRequest(user_id="", deal_id="DEAL-001", requested_pct=5.0)


def test_request_rejects_empty_deal_id() -> None:
    with pytest.raises(ValueError, match="deal_id"):
        VCAllocationRequest(user_id="user-1", deal_id="", requested_pct=5.0)


def test_request_rejects_zero_pct() -> None:
    with pytest.raises(ValueError, match="requested_pct"):
        VCAllocationRequest(user_id="user-1", deal_id="DEAL-001", requested_pct=0)


def test_request_rejects_above_100_pct() -> None:
    with pytest.raises(ValueError, match="requested_pct"):
        VCAllocationRequest(user_id="user-1", deal_id="DEAL-001", requested_pct=101.0)


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_deal_is_frozen() -> None:
    d = _deal()
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.lockup_years = 5  # type: ignore[misc]


def test_request_is_frozen() -> None:
    r = VCAllocationRequest(user_id="user-1", deal_id="DEAL-001", requested_pct=5.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.requested_pct = 10.0  # type: ignore[misc]


def test_screen_result_is_frozen() -> None:
    result = screen_deal(_deal())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.verdict = VCDealVerdict.NOT_HALAL  # type: ignore[misc]


def test_decision_is_frozen() -> None:
    deal = _deal()
    request = VCAllocationRequest(user_id="user-1", deal_id="DEAL-001", requested_pct=5.0)
    decision = evaluate_allocation(request, deal=deal)
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.allowed = False  # type: ignore[misc]


def test_policy_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_POLICY.max_per_deal_pct = 50.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned
# ---------------------------------------------------------------------------


def test_stage_string_values() -> None:
    assert DealStage.PRE_SEED.value == "pre_seed"
    assert DealStage.SEED.value == "seed"
    assert DealStage.SERIES_A.value == "series_a"
    assert DealStage.SERIES_B.value == "series_b"
    assert DealStage.GROWTH.value == "growth"


def test_use_of_proceeds_string_values() -> None:
    assert UseOfProceeds.PRODUCT_DEVELOPMENT.value == "product_development"
    assert UseOfProceeds.RETIRE_DEBT.value == "retire_debt"
    assert UseOfProceeds.UNDISCLOSED.value == "undisclosed"


def test_founder_compliance_string_values() -> None:
    assert FounderShariahCompliance.SCHOLAR_BOARD_BACKED.value == "scholar_board_backed"
    assert FounderShariahCompliance.SELF_DECLARED_HALAL.value == "self_declared_halal"
    assert FounderShariahCompliance.UNKNOWN.value == "unknown"


def test_verdict_string_values() -> None:
    assert VCDealVerdict.HALAL.value == "halal"
    assert VCDealVerdict.NOT_HALAL.value == "not_halal"
    assert VCDealVerdict.DOUBTFUL.value == "doubtful"
    assert VCDealVerdict.DOUBTFUL_PIVOT.value == "doubtful_pivot"
    assert VCDealVerdict.INSUFFICIENT_DATA.value == "insufficient_data"


# ---------------------------------------------------------------------------
# Render output
# ---------------------------------------------------------------------------


def test_render_halal_deal() -> None:
    result = screen_deal(_deal())
    text = render_screen_result(result)
    assert "✅" in text
    assert "DEAL-001" in text
    assert "Halal SaaS Co." in text
    assert "HALAL" in text
    assert "saas_b2b" in text


def test_render_not_halal_deal() -> None:
    result = screen_deal(_deal(use_of_proceeds=UseOfProceeds.RETIRE_DEBT))
    text = render_screen_result(result)
    assert "❌" in text
    assert "NOT_HALAL" in text
    assert "failures:" in text


def test_render_doubtful_deal() -> None:
    result = screen_deal(_deal(sector=HalalSector.HALAL_FINTECH))
    text = render_screen_result(result)
    assert "⚠️" in text
    assert "DOUBTFUL" in text


def test_render_doubtful_pivot_deal() -> None:
    result = screen_deal(_deal(stage=DealStage.SEED))
    text = render_screen_result(result)
    assert "🌱" in text
    assert "DOUBTFUL_PIVOT" in text


def test_render_insufficient_data_deal() -> None:
    result = screen_deal(_deal(use_of_proceeds=UseOfProceeds.UNDISCLOSED))
    text = render_screen_result(result)
    assert "❓" in text
    assert "INSUFFICIENT_DATA" in text


def test_render_does_not_include_minimum_check() -> None:
    """Pin no-USD: deal screen render never includes minimum check size."""

    result = screen_deal(_deal(minimum_check_usd=100_000.0))
    text = render_screen_result(result)
    assert "100" not in text  # no "100,000" or "$100,000"
    assert "$" not in text


def test_render_allocation_decision_allowed() -> None:
    deal = _deal()
    request = VCAllocationRequest(user_id="user-1", deal_id="DEAL-001", requested_pct=5.0)
    decision = evaluate_allocation(request, deal=deal)
    text = render_allocation_decision(decision)
    assert "✅" in text
    assert "ALLOWED" in text
    assert "5.00%" in text
    assert "10.00%" in text  # cap


def test_render_allocation_decision_blocked() -> None:
    deal = _deal()
    request = VCAllocationRequest(user_id="user-1", deal_id="DEAL-001", requested_pct=15.0)
    decision = evaluate_allocation(request, deal=deal)
    text = render_allocation_decision(decision)
    assert "🚫" in text
    assert "BLOCKED" in text
    assert "exceeds" in text.lower()


# ---------------------------------------------------------------------------
# End-to-end realistic flows
# ---------------------------------------------------------------------------


def test_typical_halal_fintech_seed_journey() -> None:
    """Pre-seed halal-fintech deal with scholar board → DOUBTFUL_PIVOT.

    Allocation under 10% cap is allowed but with the verdict
    surfaced so the operator's UI renders the warnings prominently.
    """

    deal = _deal(
        deal_id="DEAL-FINTECH-001",
        company_name="Halal Wealth Mgmt",
        sector=HalalSector.HALAL_FINTECH,
        stage=DealStage.PRE_SEED,
        has_scholar_board_review=True,
    )
    result = screen_deal(deal)
    assert result.verdict is VCDealVerdict.DOUBTFUL_PIVOT

    request = VCAllocationRequest(user_id="user-1", deal_id="DEAL-FINTECH-001", requested_pct=5.0)
    decision = evaluate_allocation(request, deal=deal)
    assert decision.allowed is True
    assert decision.deal_verdict is VCDealVerdict.DOUBTFUL_PIVOT


def test_riba_back_door_pattern_blocked() -> None:
    """A glossy 'halal SaaS' deal with retire-debt proceeds → blocked."""

    deal = _deal(
        company_name="Halal SaaS (Plot Twist)",
        use_of_proceeds=UseOfProceeds.RETIRE_DEBT,
    )
    request = VCAllocationRequest(user_id="user-1", deal_id="DEAL-001", requested_pct=2.0)
    decision = evaluate_allocation(request, deal=deal)
    assert decision.allowed is False
    assert decision.deal_verdict is VCDealVerdict.NOT_HALAL


def test_concentration_protection_full_lifecycle() -> None:
    """A user requesting 30% in one deal under default 10% cap is blocked,
    but bumping cap to 25% still blocks a 30% request."""

    deal = _deal()

    # Default cap: 30% blocked
    request_30 = VCAllocationRequest(user_id="user-1", deal_id="DEAL-001", requested_pct=30.0)
    assert evaluate_allocation(request_30, deal=deal).allowed is False

    # High-conviction cap (25%): still blocks 30%
    high = VCAllocationPolicy(max_per_deal_pct=25.0)
    assert evaluate_allocation(request_30, deal=deal, policy=high).allowed is False

    # Cap = 25%, request = 25% → allowed (boundary inclusive)
    request_25 = VCAllocationRequest(user_id="user-1", deal_id="DEAL-001", requested_pct=25.0)
    assert evaluate_allocation(request_25, deal=deal, policy=high).allowed is True
