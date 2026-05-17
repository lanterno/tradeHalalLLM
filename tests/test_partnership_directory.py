"""Tests for `halal_trader.web.partnership_directory` (Wave 10.G).

Covers: stage funnel ordering, complementarity scoring,
certification ladder, active filter, no-secret render contract.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.web.partnership_directory import (
    OUR_CAPABILITIES,
    Capability,
    HalalCertLevel,
    IntegrationStage,
    Partner,
    StageOutOfOrderError,
    StageTransition,
    advance_stage,
    build_funnel,
    cert_meets_minimum,
    complementarity_score,
    create_partner,
    deactivate,
    filter_active,
    filter_at_stage,
    render_funnel,
    render_partner,
)

UTC = timezone.utc
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


# --------------------------- Enum string pins --------------------------------


def test_capability_string_values_pinned() -> None:
    assert Capability.MANAGED_PORTFOLIOS.value == "managed_portfolios"
    assert Capability.ROBO_ADVISOR.value == "robo_advisor"
    assert Capability.MUTUAL_FUNDS.value == "mutual_funds"
    assert Capability.ACTIVE_MANAGEMENT.value == "active_management"
    assert Capability.HIGH_FREQUENCY_TRADING.value == "high_frequency_trading"
    assert Capability.HALAL_SCREENING.value == "halal_screening"
    assert Capability.PURIFICATION_LEDGER.value == "purification_ledger"
    assert Capability.LLM_REASONING.value == "llm_reasoning"
    assert Capability.BACKTESTING.value == "backtesting"
    assert Capability.BROKER_API.value == "broker_api"
    assert Capability.USER_BASE.value == "user_base"


def test_halal_cert_level_string_values_pinned() -> None:
    assert HalalCertLevel.NONE.value == "none"
    assert HalalCertLevel.SELF_DECLARED.value == "self_declared"
    assert HalalCertLevel.THIRD_PARTY_AUDITED.value == "third_party_audited"
    assert HalalCertLevel.SCHOLAR_REVIEWED.value == "scholar_reviewed"
    assert HalalCertLevel.SHARIAH_BOARD_CERTIFIED.value == "shariah_board_certified"


def test_integration_stage_string_values_pinned() -> None:
    assert IntegrationStage.INITIAL_OUTREACH.value == "initial_outreach"
    assert IntegrationStage.MUTUAL_INTEREST.value == "mutual_interest"
    assert IntegrationStage.SCOPE_ALIGNED.value == "scope_aligned"
    assert IntegrationStage.LEGAL_REVIEW.value == "legal_review"
    assert IntegrationStage.INTEGRATION_BUILD.value == "integration_build"
    assert IntegrationStage.LIVE.value == "live"
    assert IntegrationStage.PAUSED.value == "paused"


def test_our_capabilities_pinned() -> None:
    """Pin: our-side capability inventory."""

    assert Capability.ACTIVE_MANAGEMENT in OUR_CAPABILITIES
    assert Capability.HIGH_FREQUENCY_TRADING in OUR_CAPABILITIES
    assert Capability.HALAL_SCREENING in OUR_CAPABILITIES
    assert Capability.PURIFICATION_LEDGER in OUR_CAPABILITIES
    # Things we don't have in-house — they're partner-side
    assert Capability.MANAGED_PORTFOLIOS not in OUR_CAPABILITIES
    assert Capability.MUTUAL_FUNDS not in OUR_CAPABILITIES
    assert Capability.USER_BASE not in OUR_CAPABILITIES


# --------------------------- StageTransition ---------------------------------


def test_transition_rejects_naive_decided_at() -> None:
    with pytest.raises(ValueError, match="decided_at"):
        StageTransition(
            from_stage=None,
            to_stage=IntegrationStage.INITIAL_OUTREACH,
            decided_at=datetime(2026, 5, 1),
        )


def test_transition_is_frozen() -> None:
    t = StageTransition(
        from_stage=None,
        to_stage=IntegrationStage.INITIAL_OUTREACH,
        decided_at=T0,
    )
    with pytest.raises(FrozenInstanceError):
        t.notes = "other"  # type: ignore[misc]


# --------------------------- Partner construction ----------------------------


def _wahed() -> Partner:
    return create_partner(
        partner_id="wahed",
        display_name="Wahed Invest",
        public_url="https://wahedinvest.com",
        capabilities=[
            Capability.MANAGED_PORTFOLIOS,
            Capability.ROBO_ADVISOR,
            Capability.HALAL_SCREENING,
            Capability.USER_BASE,
        ],
        halal_cert_level=HalalCertLevel.SHARIAH_BOARD_CERTIFIED,
        now=T0,
    )


def test_partner_rejects_empty_partner_id() -> None:
    with pytest.raises(ValueError, match="partner_id"):
        create_partner(
            partner_id="",
            display_name="X",
            public_url="https://x.com",
            capabilities=[],
            halal_cert_level=HalalCertLevel.NONE,
            now=T0,
        )


def test_partner_rejects_empty_display_name() -> None:
    with pytest.raises(ValueError, match="display_name"):
        create_partner(
            partner_id="x",
            display_name="",
            public_url="https://x.com",
            capabilities=[],
            halal_cert_level=HalalCertLevel.NONE,
            now=T0,
        )


def test_partner_rejects_url_without_scheme() -> None:
    with pytest.raises(ValueError, match="http"):
        create_partner(
            partner_id="x",
            display_name="X",
            public_url="x.com",
            capabilities=[],
            halal_cert_level=HalalCertLevel.NONE,
            now=T0,
        )


def test_partner_accepts_https_and_http() -> None:
    partner_https = create_partner(
        partner_id="a",
        display_name="A",
        public_url="https://a.com",
        capabilities=[],
        halal_cert_level=HalalCertLevel.NONE,
        now=T0,
    )
    partner_http = create_partner(
        partner_id="b",
        display_name="B",
        public_url="http://b.com",
        capabilities=[],
        halal_cert_level=HalalCertLevel.NONE,
        now=T0,
    )
    assert partner_https.public_url.startswith("https://")
    assert partner_http.public_url.startswith("http://")


def test_create_partner_starts_at_initial_outreach() -> None:
    p = _wahed()
    assert p.current_stage is IntegrationStage.INITIAL_OUTREACH
    assert len(p.transitions) == 1
    assert p.transitions[0].from_stage is None
    assert p.transitions[0].to_stage is IntegrationStage.INITIAL_OUTREACH
    assert p.active is True


def test_create_partner_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now"):
        create_partner(
            partner_id="x",
            display_name="X",
            public_url="https://x.com",
            capabilities=[],
            halal_cert_level=HalalCertLevel.NONE,
            now=datetime(2026, 5, 1),
        )


def test_partner_is_frozen() -> None:
    p = _wahed()
    with pytest.raises(FrozenInstanceError):
        p.active = False  # type: ignore[misc]


# --------------------------- advance_stage -----------------------------------


def test_advance_stage_one_step_forward() -> None:
    p = _wahed()
    p = advance_stage(p, IntegrationStage.MUTUAL_INTEREST, now=T0)
    assert p.current_stage is IntegrationStage.MUTUAL_INTEREST
    assert len(p.transitions) == 2


def test_advance_stage_skip_rejected() -> None:
    """Pin: must advance one stage at a time."""

    p = _wahed()
    with pytest.raises(StageOutOfOrderError) as exc_info:
        advance_stage(p, IntegrationStage.LEGAL_REVIEW, now=T0)
    assert exc_info.value.from_stage is IntegrationStage.INITIAL_OUTREACH
    assert exc_info.value.to_stage is IntegrationStage.LEGAL_REVIEW


def test_advance_stage_full_funnel_to_live() -> None:
    p = _wahed()
    for stage in [
        IntegrationStage.MUTUAL_INTEREST,
        IntegrationStage.SCOPE_ALIGNED,
        IntegrationStage.LEGAL_REVIEW,
        IntegrationStage.INTEGRATION_BUILD,
        IntegrationStage.LIVE,
    ]:
        p = advance_stage(p, stage, now=T0)
    assert p.current_stage is IntegrationStage.LIVE


def test_advance_stage_to_paused_from_anywhere() -> None:
    """Pin: PAUSED can be entered from any stage."""

    p = _wahed()
    p = advance_stage(p, IntegrationStage.MUTUAL_INTEREST, now=T0)
    p = advance_stage(p, IntegrationStage.SCOPE_ALIGNED, now=T0)
    p = advance_stage(p, IntegrationStage.PAUSED, now=T0)
    assert p.current_stage is IntegrationStage.PAUSED


def test_advance_stage_resume_from_paused() -> None:
    p = _wahed()
    p = advance_stage(p, IntegrationStage.MUTUAL_INTEREST, now=T0)
    p = advance_stage(p, IntegrationStage.PAUSED, now=T0)
    # Operator picks up where they left off
    p = advance_stage(p, IntegrationStage.SCOPE_ALIGNED, now=T0)
    assert p.current_stage is IntegrationStage.SCOPE_ALIGNED


def test_advance_stage_same_stage_rejected() -> None:
    p = _wahed()
    with pytest.raises(ValueError, match="already at"):
        advance_stage(p, IntegrationStage.INITIAL_OUTREACH, now=T0)


def test_advance_stage_naive_now_rejected() -> None:
    p = _wahed()
    with pytest.raises(ValueError, match="now"):
        advance_stage(p, IntegrationStage.MUTUAL_INTEREST, now=datetime(2026, 5, 1))


def test_advance_stage_returns_new_state() -> None:
    """Pin: state is immutable; operations return new state."""

    original = _wahed()
    new_state = advance_stage(original, IntegrationStage.MUTUAL_INTEREST, now=T0)
    assert original.current_stage is IntegrationStage.INITIAL_OUTREACH
    assert new_state.current_stage is IntegrationStage.MUTUAL_INTEREST


def test_advance_stage_records_notes() -> None:
    p = _wahed()
    p = advance_stage(
        p,
        IntegrationStage.MUTUAL_INTEREST,
        now=T0,
        notes="had a great call with their CTO",
    )
    last = p.transitions[-1]
    assert last.notes == "had a great call with their CTO"


# --------------------------- deactivate --------------------------------------


def test_deactivate_marks_inactive_preserves_transitions() -> None:
    p = _wahed()
    p = advance_stage(p, IntegrationStage.MUTUAL_INTEREST, now=T0)
    p = deactivate(p, now=T0 + timedelta(days=30))
    assert p.active is False
    # Transitions preserved
    assert len(p.transitions) == 2


def test_deactivate_rejects_naive_now() -> None:
    p = _wahed()
    with pytest.raises(ValueError, match="now"):
        deactivate(p, now=datetime(2026, 5, 1))


# --------------------------- complementarity_score ---------------------------


def test_complementarity_perfect_disjoint_is_one() -> None:
    """Pin: a partner with all capabilities we DON'T have, scoring 1.0."""

    p = create_partner(
        partner_id="p",
        display_name="P",
        public_url="https://p.com",
        capabilities=[
            Capability.MANAGED_PORTFOLIOS,
            Capability.MUTUAL_FUNDS,
            Capability.USER_BASE,
        ],
        halal_cert_level=HalalCertLevel.NONE,
        now=T0,
    )
    score = complementarity_score(p)
    assert score == 1.0


def test_complementarity_perfect_overlap_is_zero() -> None:
    """Pin: a partner with exactly our capabilities scores 0.0."""

    p = create_partner(
        partner_id="p",
        display_name="P",
        public_url="https://p.com",
        capabilities=list(OUR_CAPABILITIES),
        halal_cert_level=HalalCertLevel.NONE,
        now=T0,
    )
    score = complementarity_score(p)
    assert score == 0.0


def test_complementarity_wahed_is_high() -> None:
    """Wahed has managed portfolios + robo + user base + halal screening
    (one overlap with us); high complementarity."""

    score = complementarity_score(_wahed())
    assert score > 0.6


def test_complementarity_empty_partner_zero() -> None:
    p = create_partner(
        partner_id="p",
        display_name="P",
        public_url="https://p.com",
        capabilities=[],
        halal_cert_level=HalalCertLevel.NONE,
        now=T0,
    )
    # Disjoint with our_capabilities → score = 1 - 0/|union| = 1.0
    score = complementarity_score(p)
    assert score == 1.0


def test_complementarity_custom_our_capabilities() -> None:
    p = create_partner(
        partner_id="p",
        display_name="P",
        public_url="https://p.com",
        capabilities=[Capability.HALAL_SCREENING],
        halal_cert_level=HalalCertLevel.NONE,
        now=T0,
    )
    # If we treat ourselves as having only HALAL_SCREENING too,
    # the score is 0.0 (perfect overlap)
    custom = frozenset({Capability.HALAL_SCREENING})
    score = complementarity_score(p, our_capabilities=custom)
    assert score == 0.0


# --------------------------- cert_meets_minimum ------------------------------


def test_cert_meets_minimum_self_meets_self() -> None:
    p = create_partner(
        partner_id="p",
        display_name="P",
        public_url="https://p.com",
        capabilities=[],
        halal_cert_level=HalalCertLevel.SELF_DECLARED,
        now=T0,
    )
    assert cert_meets_minimum(p, minimum=HalalCertLevel.SELF_DECLARED)


def test_cert_meets_minimum_higher_meets_lower() -> None:
    p = create_partner(
        partner_id="p",
        display_name="P",
        public_url="https://p.com",
        capabilities=[],
        halal_cert_level=HalalCertLevel.SHARIAH_BOARD_CERTIFIED,
        now=T0,
    )
    assert cert_meets_minimum(p, minimum=HalalCertLevel.SCHOLAR_REVIEWED)


def test_cert_meets_minimum_lower_fails_higher() -> None:
    p = create_partner(
        partner_id="p",
        display_name="P",
        public_url="https://p.com",
        capabilities=[],
        halal_cert_level=HalalCertLevel.SELF_DECLARED,
        now=T0,
    )
    assert not cert_meets_minimum(p, minimum=HalalCertLevel.SCHOLAR_REVIEWED)


def test_cert_none_below_everything() -> None:
    p = create_partner(
        partner_id="p",
        display_name="P",
        public_url="https://p.com",
        capabilities=[],
        halal_cert_level=HalalCertLevel.NONE,
        now=T0,
    )
    assert not cert_meets_minimum(p, minimum=HalalCertLevel.SELF_DECLARED)


# --------------------------- filter_active / filter_at_stage -----------------


def test_filter_active_excludes_inactive() -> None:
    a = _wahed()
    b = create_partner(
        partner_id="b",
        display_name="B",
        public_url="https://b.com",
        capabilities=[],
        halal_cert_level=HalalCertLevel.NONE,
        now=T0,
    )
    b = deactivate(b, now=T0)
    active = filter_active([a, b])
    ids = {p.partner_id for p in active}
    assert ids == {"wahed"}


def test_filter_at_stage_returns_only_matching() -> None:
    a = _wahed()
    b = create_partner(
        partner_id="b",
        display_name="B",
        public_url="https://b.com",
        capabilities=[],
        halal_cert_level=HalalCertLevel.NONE,
        now=T0,
    )
    b = advance_stage(b, IntegrationStage.MUTUAL_INTEREST, now=T0)
    initial = filter_at_stage([a, b], IntegrationStage.INITIAL_OUTREACH)
    mutual = filter_at_stage([a, b], IntegrationStage.MUTUAL_INTEREST)
    assert {p.partner_id for p in initial} == {"wahed"}
    assert {p.partner_id for p in mutual} == {"b"}


# --------------------------- build_funnel ------------------------------------


def test_build_funnel_counts_per_stage() -> None:
    a = _wahed()
    b = create_partner(
        partner_id="b",
        display_name="B",
        public_url="https://b.com",
        capabilities=[],
        halal_cert_level=HalalCertLevel.NONE,
        now=T0,
    )
    b = advance_stage(b, IntegrationStage.MUTUAL_INTEREST, now=T0)
    funnel = build_funnel([a, b])
    assert funnel.total_active == 2
    assert funnel.count_at(IntegrationStage.INITIAL_OUTREACH) == 1
    assert funnel.count_at(IntegrationStage.MUTUAL_INTEREST) == 1


def test_build_funnel_excludes_inactive() -> None:
    a = _wahed()
    b = create_partner(
        partner_id="b",
        display_name="B",
        public_url="https://b.com",
        capabilities=[],
        halal_cert_level=HalalCertLevel.NONE,
        now=T0,
    )
    b = deactivate(b, now=T0)
    funnel = build_funnel([a, b])
    assert funnel.total_active == 1


def test_build_funnel_total_live_count() -> None:
    p = _wahed()
    for stage in [
        IntegrationStage.MUTUAL_INTEREST,
        IntegrationStage.SCOPE_ALIGNED,
        IntegrationStage.LEGAL_REVIEW,
        IntegrationStage.INTEGRATION_BUILD,
        IntegrationStage.LIVE,
    ]:
        p = advance_stage(p, stage, now=T0)
    funnel = build_funnel([p])
    assert funnel.total_live == 1


# --------------------------- render ------------------------------------------


def test_render_partner_includes_display_name_and_url() -> None:
    p = _wahed()
    out = render_partner(p)
    assert "Wahed Invest" in out
    assert "wahedinvest.com" in out


def test_render_partner_shows_stage_emoji() -> None:
    p = _wahed()
    out = render_partner(p)
    # Stage initial_outreach → 📨
    assert "📨" in out


def test_render_partner_shows_cert_emoji() -> None:
    p = _wahed()
    out = render_partner(p)
    # Sharia certified → 🕌
    assert "🕌" in out


def test_render_partner_shows_complementarity() -> None:
    p = _wahed()
    out = render_partner(p)
    # Wahed scores high (>= 60%)
    assert "complementarity" in out


def test_render_partner_shows_inactive_marker() -> None:
    p = _wahed()
    p = deactivate(p, now=T0)
    out = render_partner(p)
    assert "inactive" in out


def test_render_partner_no_secret_leak() -> None:
    """Pin: render never includes internal fields / NDA docs / revenue."""

    p = _wahed()
    out = render_partner(p)
    # No internal contact patterns
    assert "@" not in out  # no email-shaped substrings
    assert "$" not in out  # no revenue
    assert "USD" not in out
    assert "ARR" not in out
    assert "MRR" not in out
    assert "internal" not in out.lower()
    assert "nda" not in out.lower()


def test_render_funnel_basic() -> None:
    a = _wahed()
    b = create_partner(
        partner_id="b",
        display_name="B",
        public_url="https://b.com",
        capabilities=[],
        halal_cert_level=HalalCertLevel.NONE,
        now=T0,
    )
    b = advance_stage(b, IntegrationStage.MUTUAL_INTEREST, now=T0)
    funnel = build_funnel([a, b])
    out = render_funnel(funnel)
    assert "2 active" in out
    assert "initial_outreach: 1" in out
    assert "mutual_interest: 1" in out


def test_render_funnel_omits_zero_count_stages() -> None:
    a = _wahed()
    funnel = build_funnel([a])
    out = render_funnel(funnel)
    # initial_outreach: 1 should appear; live: 0 should not
    assert "initial_outreach: 1" in out
    assert "live:" not in out


# --------------------------- e2e flows ---------------------------------------


def test_e2e_wahed_partnership_full_funnel() -> None:
    p = _wahed()
    t = T0
    for stage in [
        IntegrationStage.MUTUAL_INTEREST,
        IntegrationStage.SCOPE_ALIGNED,
        IntegrationStage.LEGAL_REVIEW,
        IntegrationStage.INTEGRATION_BUILD,
        IntegrationStage.LIVE,
    ]:
        t += timedelta(days=14)
        p = advance_stage(p, stage, now=t)
    assert p.current_stage is IntegrationStage.LIVE
    # Full audit trail
    assert len(p.transitions) == 6


def test_e2e_paused_then_revived() -> None:
    p = _wahed()
    p = advance_stage(p, IntegrationStage.MUTUAL_INTEREST, now=T0)
    p = advance_stage(p, IntegrationStage.PAUSED, now=T0 + timedelta(days=30))
    # 6 months later, BD revives
    p = advance_stage(p, IntegrationStage.SCOPE_ALIGNED, now=T0 + timedelta(days=180))
    assert p.current_stage is IntegrationStage.SCOPE_ALIGNED
    # Audit trail preserved
    transition_to_stages = [t.to_stage for t in p.transitions]
    assert IntegrationStage.PAUSED in transition_to_stages


def test_e2e_replay_consistency() -> None:
    """Pin: applying same operations produces equal partner states."""

    def build() -> Partner:
        p = _wahed()
        p = advance_stage(p, IntegrationStage.MUTUAL_INTEREST, now=T0)
        p = advance_stage(p, IntegrationStage.SCOPE_ALIGNED, now=T0)
        return p

    a = build()
    b = build()
    assert a == b
