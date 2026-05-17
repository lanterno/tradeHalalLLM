"""Tests for the Shariah Supervisory Board governance engine."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from halal_trader.halal.ssb_governance import (
    DEFAULT_POLICY,
    FiqhSchool,
    Ruling,
    RulingOutcome,
    RulingScope,
    ScholarMember,
    SSBPolicy,
    Vote,
    needs_quarterly_review,
    render_board_composition,
    render_ruling,
    validate_board,
)

_NOW = datetime(2026, 5, 1, tzinfo=UTC)


def _member(
    *,
    name: str = "Mufti A",
    school: FiqhSchool = FiqhSchool.HANAFI,
    appointed_at: datetime | None = None,
    expires_at: datetime | None = None,
    bio_url: str = "",
) -> ScholarMember:
    return ScholarMember(
        name=name,
        school=school,
        appointed_at=appointed_at or (_NOW - timedelta(days=30)),
        expires_at=expires_at or (_NOW + timedelta(days=365 * 2)),
        bio_url=bio_url,
    )


def _diverse_board() -> tuple[ScholarMember, ...]:
    """A valid 3-member, 3-school board for use as the default."""

    return (
        _member(name="Mufti Hanafi", school=FiqhSchool.HANAFI),
        _member(name="Mufti Shafii", school=FiqhSchool.SHAFII),
        _member(name="Mufti Maliki", school=FiqhSchool.MALIKI),
    )


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------


def test_default_policy_values() -> None:
    p = DEFAULT_POLICY
    assert p.minimum_members == 3
    assert p.minimum_schools == 3
    assert p.supermajority_pct == pytest.approx(2.0 / 3.0)
    assert p.review_cycle_days == 90
    assert p.term_length_days == 365 * 3


def test_policy_rejects_zero_minimum_members() -> None:
    with pytest.raises(ValueError, match="minimum_members"):
        SSBPolicy(minimum_members=0)


def test_policy_rejects_zero_minimum_schools() -> None:
    with pytest.raises(ValueError, match="minimum_schools"):
        SSBPolicy(minimum_schools=0)


def test_policy_rejects_schools_exceeding_members() -> None:
    """Pin: schools floor cannot exceed members floor."""

    with pytest.raises(ValueError, match="cannot exceed"):
        SSBPolicy(minimum_members=2, minimum_schools=3)


def test_policy_rejects_supermajority_at_or_below_simple_majority() -> None:
    """Pin: a simple majority isn't enough; supermajority must be > 0.5."""

    with pytest.raises(ValueError, match="supermajority_pct"):
        SSBPolicy(supermajority_pct=0.5)


def test_policy_rejects_supermajority_above_1() -> None:
    with pytest.raises(ValueError, match="supermajority_pct"):
        SSBPolicy(supermajority_pct=1.01)


def test_policy_rejects_zero_review_cycle() -> None:
    with pytest.raises(ValueError, match="review_cycle_days"):
        SSBPolicy(review_cycle_days=0)


def test_policy_rejects_zero_term_length() -> None:
    with pytest.raises(ValueError, match="term_length_days"):
        SSBPolicy(term_length_days=0)


# ---------------------------------------------------------------------------
# ScholarMember validation
# ---------------------------------------------------------------------------


def test_member_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        _member(name="")


def test_member_rejects_naive_appointed_at() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _member(appointed_at=datetime(2026, 5, 1))


def test_member_rejects_naive_expires_at() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _member(expires_at=datetime(2027, 5, 1))


def test_member_rejects_expires_before_appointed() -> None:
    with pytest.raises(ValueError, match="must be after"):
        _member(
            appointed_at=_NOW,
            expires_at=_NOW - timedelta(days=1),
        )


def test_member_is_active_within_term() -> None:
    m = _member()
    assert m.is_active(now=_NOW) is True


def test_member_is_inactive_after_term() -> None:
    m = _member(
        appointed_at=_NOW - timedelta(days=400),
        expires_at=_NOW - timedelta(days=10),
    )
    assert m.is_active(now=_NOW) is False


def test_member_is_inactive_before_appointment() -> None:
    m = _member(
        appointed_at=_NOW + timedelta(days=10),
        expires_at=_NOW + timedelta(days=400),
    )
    assert m.is_active(now=_NOW) is False


def test_member_is_active_rejects_naive_now() -> None:
    m = _member()
    with pytest.raises(ValueError, match="timezone-aware"):
        m.is_active(now=datetime(2026, 5, 1))


# ---------------------------------------------------------------------------
# Board composition validation
# ---------------------------------------------------------------------------


def test_diverse_three_school_board_is_valid() -> None:
    result = validate_board(_diverse_board(), now=_NOW)
    assert result.is_valid is True
    assert result.member_count == 3
    assert result.school_count == 3


def test_three_member_single_school_board_fails() -> None:
    """Pin: three Hanafis fails the 3-school diversity requirement."""

    members = (
        _member(name="A", school=FiqhSchool.HANAFI),
        _member(name="B", school=FiqhSchool.HANAFI),
        _member(name="C", school=FiqhSchool.HANAFI),
    )
    result = validate_board(members, now=_NOW)
    assert result.is_valid is False
    assert any("school count" in f for f in result.failures)


def test_two_school_three_member_board_fails() -> None:
    """Pin: 2 Hanafis + 1 Shafi'i fails diversity (only 2 schools)."""

    members = (
        _member(name="A", school=FiqhSchool.HANAFI),
        _member(name="B", school=FiqhSchool.HANAFI),
        _member(name="C", school=FiqhSchool.SHAFII),
    )
    result = validate_board(members, now=_NOW)
    assert result.is_valid is False
    assert any("school count" in f for f in result.failures)


def test_two_member_board_fails_minimum() -> None:
    members = (
        _member(name="A", school=FiqhSchool.HANAFI),
        _member(name="B", school=FiqhSchool.SHAFII),
    )
    result = validate_board(members, now=_NOW)
    assert result.is_valid is False
    assert any("member count" in f for f in result.failures)


def test_board_with_expired_member_warns_and_excludes() -> None:
    """Expired member is dropped from active count + surfaces a warning."""

    members = (
        _member(name="A", school=FiqhSchool.HANAFI),
        _member(name="B", school=FiqhSchool.SHAFII),
        _member(name="C", school=FiqhSchool.MALIKI),
        _member(
            name="Expired",
            school=FiqhSchool.HANBALI,
            appointed_at=_NOW - timedelta(days=400),
            expires_at=_NOW - timedelta(days=10),
        ),
    )
    result = validate_board(members, now=_NOW)
    assert result.is_valid is True
    assert result.member_count == 3  # expired excluded
    assert any("past term" in w for w in result.warnings)


def test_board_with_expired_majority_fails() -> None:
    """If too many members expire, board falls below minimum."""

    members = (
        _member(name="A", school=FiqhSchool.HANAFI),
        _member(
            name="B",
            school=FiqhSchool.SHAFII,
            appointed_at=_NOW - timedelta(days=400),
            expires_at=_NOW - timedelta(days=10),
        ),
        _member(
            name="C",
            school=FiqhSchool.MALIKI,
            appointed_at=_NOW - timedelta(days=400),
            expires_at=_NOW - timedelta(days=10),
        ),
    )
    result = validate_board(members, now=_NOW)
    assert result.is_valid is False
    assert result.member_count == 1


def test_board_rejects_duplicate_scholar_name() -> None:
    members = (
        _member(name="A", school=FiqhSchool.HANAFI),
        _member(name="A", school=FiqhSchool.SHAFII),
        _member(name="C", school=FiqhSchool.MALIKI),
    )
    result = validate_board(members, now=_NOW)
    assert any("duplicate" in f for f in result.failures)


def test_board_with_four_members_four_schools_is_valid() -> None:
    members = (
        _member(name="A", school=FiqhSchool.HANAFI),
        _member(name="B", school=FiqhSchool.SHAFII),
        _member(name="C", school=FiqhSchool.MALIKI),
        _member(name="D", school=FiqhSchool.HANBALI),
    )
    result = validate_board(members, now=_NOW)
    assert result.is_valid is True
    assert result.member_count == 4
    assert result.school_count == 4


def test_validate_board_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_board(_diverse_board(), now=datetime(2026, 5, 1))


def test_strict_policy_with_four_school_minimum_flips_diversity_verdict() -> None:
    """Custom policy bumps minimum_schools to 4."""

    strict = SSBPolicy(minimum_members=4, minimum_schools=4)
    members = (
        _member(name="A", school=FiqhSchool.HANAFI),
        _member(name="B", school=FiqhSchool.SHAFII),
        _member(name="C", school=FiqhSchool.MALIKI),
    )
    result = validate_board(members, now=_NOW, policy=strict)
    assert result.is_valid is False


# ---------------------------------------------------------------------------
# Vote validation
# ---------------------------------------------------------------------------


def test_vote_rejects_empty_member_name() -> None:
    with pytest.raises(ValueError, match="member_name"):
        Vote(
            member_name="",
            school=FiqhSchool.HANAFI,
            outcome=RulingOutcome.PERMISSIBLE,
        )


def test_vote_impermissible_requires_rationale() -> None:
    """Pin: rejections need justification."""

    with pytest.raises(ValueError, match="rationale required"):
        Vote(
            member_name="Mufti A",
            school=FiqhSchool.HANAFI,
            outcome=RulingOutcome.IMPERMISSIBLE,
            rationale="",
        )


def test_vote_conditional_requires_rationale() -> None:
    """Pin: conditional verdicts need explanation."""

    with pytest.raises(ValueError, match="rationale required"):
        Vote(
            member_name="Mufti A",
            school=FiqhSchool.HANAFI,
            outcome=RulingOutcome.PERMISSIBLE_WITH_CONDITIONS,
            rationale="",
        )


def test_vote_whitespace_rationale_rejected_for_impermissible() -> None:
    """Pin: whitespace doesn't satisfy the rationale requirement."""

    with pytest.raises(ValueError, match="rationale required"):
        Vote(
            member_name="Mufti A",
            school=FiqhSchool.HANAFI,
            outcome=RulingOutcome.IMPERMISSIBLE,
            rationale="   ",
        )


def test_vote_permissible_allows_empty_rationale() -> None:
    v = Vote(
        member_name="Mufti A",
        school=FiqhSchool.HANAFI,
        outcome=RulingOutcome.PERMISSIBLE,
    )
    assert v.rationale == ""


def test_vote_deferred_allows_empty_rationale() -> None:
    v = Vote(
        member_name="Mufti A",
        school=FiqhSchool.HANAFI,
        outcome=RulingOutcome.DEFERRED,
    )
    assert v.rationale == ""


# ---------------------------------------------------------------------------
# Consensus computation — conservative tiebreak
# ---------------------------------------------------------------------------


def _vote(
    name: str = "Mufti X",
    school: FiqhSchool = FiqhSchool.HANAFI,
    outcome: RulingOutcome = RulingOutcome.PERMISSIBLE,
    rationale: str = "valid per AAOIFI Standard",
) -> Vote:
    return Vote(member_name=name, school=school, outcome=outcome, rationale=rationale)


def _ruling(
    *,
    votes: tuple[Vote, ...],
    conditions: tuple[str, ...] = (),
    ruling_id: str = "SSB-2026-Q2-001",
) -> Ruling:
    return Ruling(
        ruling_id=ruling_id,
        scope=RulingScope.PRODUCT,
        subject="commodity-ETF screener",
        description="Rules on the Wave 1.G commodity screener for halal compliance",
        issued_at=_NOW,
        votes=votes,
        conditions=conditions,
    )


def test_unanimous_permissible_is_permissible() -> None:
    votes = (
        _vote("A", FiqhSchool.HANAFI, RulingOutcome.PERMISSIBLE),
        _vote("B", FiqhSchool.SHAFII, RulingOutcome.PERMISSIBLE),
        _vote("C", FiqhSchool.MALIKI, RulingOutcome.PERMISSIBLE),
    )
    r = _ruling(votes=votes)
    assert r.consensus() is RulingOutcome.PERMISSIBLE


def test_any_impermissible_overrides_permissible() -> None:
    """Pin: conservative tiebreak — single IMPERMISSIBLE wins."""

    votes = (
        _vote("A", FiqhSchool.HANAFI, RulingOutcome.PERMISSIBLE),
        _vote("B", FiqhSchool.SHAFII, RulingOutcome.PERMISSIBLE),
        _vote("C", FiqhSchool.MALIKI, RulingOutcome.IMPERMISSIBLE),
    )
    r = _ruling(votes=votes)
    assert r.consensus() is RulingOutcome.IMPERMISSIBLE


def test_2_of_3_majority_is_below_supermajority_returns_deferred() -> None:
    """Pin: 2/3 = 0.667 ≥ 0.667 supermajority threshold → PERMISSIBLE.

    But a 1-of-3 vote is below threshold → DEFERRED.
    """

    votes = (
        _vote("A", FiqhSchool.HANAFI, RulingOutcome.PERMISSIBLE),
        _vote("B", FiqhSchool.SHAFII, RulingOutcome.DEFERRED),
        _vote("C", FiqhSchool.MALIKI, RulingOutcome.DEFERRED),
    )
    r = _ruling(votes=votes)
    assert r.consensus() is RulingOutcome.DEFERRED


def test_2_of_3_pass_meets_supermajority() -> None:
    """Pin: 2/3 = 0.667 meets the 2/3 supermajority threshold."""

    votes = (
        _vote("A", FiqhSchool.HANAFI, RulingOutcome.PERMISSIBLE),
        _vote("B", FiqhSchool.SHAFII, RulingOutcome.PERMISSIBLE),
        _vote("C", FiqhSchool.MALIKI, RulingOutcome.DEFERRED),
    )
    r = _ruling(votes=votes)
    assert r.consensus() is RulingOutcome.PERMISSIBLE


def test_mixed_pass_with_conditional_returns_conditional() -> None:
    """Pin: any conditional pass in a passing supermajority → conditional."""

    votes = (
        _vote("A", FiqhSchool.HANAFI, RulingOutcome.PERMISSIBLE),
        _vote(
            "B",
            FiqhSchool.SHAFII,
            RulingOutcome.PERMISSIBLE_WITH_CONDITIONS,
            "must comply with X",
        ),
        _vote("C", FiqhSchool.MALIKI, RulingOutcome.PERMISSIBLE),
    )
    r = _ruling(
        votes=votes,
        conditions=("must comply with X",),
    )
    assert r.consensus() is RulingOutcome.PERMISSIBLE_WITH_CONDITIONS


def test_all_deferred_returns_deferred() -> None:
    votes = (
        _vote("A", FiqhSchool.HANAFI, RulingOutcome.DEFERRED),
        _vote("B", FiqhSchool.SHAFII, RulingOutcome.DEFERRED),
        _vote("C", FiqhSchool.MALIKI, RulingOutcome.DEFERRED),
    )
    r = _ruling(votes=votes)
    assert r.consensus() is RulingOutcome.DEFERRED


def test_strict_supermajority_policy_flips_marginal_verdict() -> None:
    """A 75% supermajority threshold rejects a 2-of-3 = 67%."""

    strict = SSBPolicy(supermajority_pct=0.75)
    votes = (
        _vote("A", FiqhSchool.HANAFI, RulingOutcome.PERMISSIBLE),
        _vote("B", FiqhSchool.SHAFII, RulingOutcome.PERMISSIBLE),
        _vote("C", FiqhSchool.MALIKI, RulingOutcome.DEFERRED),
    )
    r = _ruling(votes=votes)
    assert r.consensus(policy=strict) is RulingOutcome.DEFERRED


def test_empty_votes_return_deferred() -> None:
    """Pin: zero votes → DEFERRED (cannot rule without input)."""

    r = Ruling(
        ruling_id="SSB-2026-Q2-001",
        scope=RulingScope.PRODUCT,
        subject="test",
        description="test",
        issued_at=_NOW,
        votes=(),
    )
    assert r.consensus() is RulingOutcome.DEFERRED


# ---------------------------------------------------------------------------
# Ruling validation
# ---------------------------------------------------------------------------


def test_ruling_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="ruling_id"):
        Ruling(
            ruling_id="",
            scope=RulingScope.PRODUCT,
            subject="test",
            description="test",
            issued_at=_NOW,
            votes=(_vote(),),
        )


def test_ruling_rejects_empty_subject() -> None:
    with pytest.raises(ValueError, match="subject"):
        Ruling(
            ruling_id="SSB-2026-Q2-001",
            scope=RulingScope.PRODUCT,
            subject="",
            description="test",
            issued_at=_NOW,
            votes=(_vote(),),
        )


def test_ruling_rejects_empty_description() -> None:
    with pytest.raises(ValueError, match="description"):
        Ruling(
            ruling_id="SSB-2026-Q2-001",
            scope=RulingScope.PRODUCT,
            subject="test",
            description="",
            issued_at=_NOW,
            votes=(_vote(),),
        )


def test_ruling_rejects_naive_issued_at() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        Ruling(
            ruling_id="SSB-2026-Q2-001",
            scope=RulingScope.PRODUCT,
            subject="test",
            description="test",
            issued_at=datetime(2026, 5, 1),
            votes=(_vote(),),
        )


def test_ruling_rejects_conditional_vote_without_conditions() -> None:
    """Pin: a PERMISSIBLE_WITH_CONDITIONS vote requires explicit conditions."""

    votes = (
        _vote("A", FiqhSchool.HANAFI, RulingOutcome.PERMISSIBLE_WITH_CONDITIONS, "needs X"),
        _vote("B", FiqhSchool.SHAFII, RulingOutcome.PERMISSIBLE),
        _vote("C", FiqhSchool.MALIKI, RulingOutcome.PERMISSIBLE),
    )
    with pytest.raises(ValueError, match="no conditions listed"):
        _ruling(votes=votes, conditions=())  # missing conditions


def test_ruling_accepts_conditional_with_conditions() -> None:
    votes = (
        _vote("A", FiqhSchool.HANAFI, RulingOutcome.PERMISSIBLE_WITH_CONDITIONS, "needs X"),
        _vote("B", FiqhSchool.SHAFII, RulingOutcome.PERMISSIBLE),
        _vote("C", FiqhSchool.MALIKI, RulingOutcome.PERMISSIBLE),
    )
    r = _ruling(votes=votes, conditions=("operator must enable X",))
    assert r.conditions == ("operator must enable X",)


def test_ruling_rejects_duplicate_voter() -> None:
    """Pin: one scholar, one vote."""

    votes = (
        _vote("Mufti A", FiqhSchool.HANAFI),
        _vote("Mufti A", FiqhSchool.SHAFII),
        _vote("Mufti C", FiqhSchool.MALIKI),
    )
    with pytest.raises(ValueError, match="duplicate vote"):
        _ruling(votes=votes)


# ---------------------------------------------------------------------------
# Quarterly review cadence
# ---------------------------------------------------------------------------


def test_empty_rulings_needs_review() -> None:
    """Pin: a board that has issued no rulings is overdue from day one."""

    assert needs_quarterly_review((), now=_NOW) is True


def test_recent_ruling_does_not_need_review() -> None:
    r = _ruling(votes=(_vote(),))
    assert needs_quarterly_review((r,), now=_NOW + timedelta(days=10)) is False


def test_old_ruling_needs_review() -> None:
    r = _ruling(votes=(_vote(),))
    assert needs_quarterly_review((r,), now=_NOW + timedelta(days=100)) is True


def test_review_threshold_at_exactly_90_days() -> None:
    """Pin: at exactly 90d, no review needed (greater-than not >=)."""

    r = _ruling(votes=(_vote(),))
    # at exactly 90d, the timedelta comparison is strict greater-than,
    # so no review needed at the boundary
    assert needs_quarterly_review((r,), now=_NOW + timedelta(days=90)) is False


def test_review_threshold_just_past_90_days() -> None:
    r = _ruling(votes=(_vote(),))
    assert needs_quarterly_review((r,), now=_NOW + timedelta(days=91)) is True


def test_uses_most_recent_ruling() -> None:
    """Pin: only the latest issued_at matters."""

    old = Ruling(
        ruling_id="OLD",
        scope=RulingScope.PRODUCT,
        subject="x",
        description="x",
        issued_at=_NOW - timedelta(days=300),
        votes=(_vote(),),
    )
    recent = Ruling(
        ruling_id="NEW",
        scope=RulingScope.PRODUCT,
        subject="y",
        description="y",
        issued_at=_NOW - timedelta(days=10),
        votes=(_vote(),),
    )
    assert needs_quarterly_review((old, recent), now=_NOW) is False


def test_custom_review_cycle_flows_through() -> None:
    """Strict 30-day review cycle catches rulings the default 90 wouldn't."""

    strict = SSBPolicy(review_cycle_days=30)
    r = _ruling(votes=(_vote(),))
    assert needs_quarterly_review((r,), now=_NOW + timedelta(days=45), policy=strict) is True


def test_review_rejects_naive_now() -> None:
    r = _ruling(votes=(_vote(),))
    with pytest.raises(ValueError, match="timezone-aware"):
        needs_quarterly_review((r,), now=datetime(2026, 5, 1))


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_member_is_frozen() -> None:
    m = _member()
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.school = FiqhSchool.SHAFII  # type: ignore[misc]


def test_vote_is_frozen() -> None:
    v = _vote()
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.outcome = RulingOutcome.IMPERMISSIBLE  # type: ignore[misc]


def test_ruling_is_frozen() -> None:
    r = _ruling(votes=(_vote(),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.subject = "other"  # type: ignore[misc]


def test_policy_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_POLICY.minimum_members = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned for JSON / DB stability
# ---------------------------------------------------------------------------


def test_school_string_values() -> None:
    assert FiqhSchool.HANAFI.value == "hanafi"
    assert FiqhSchool.SHAFII.value == "shafii"
    assert FiqhSchool.MALIKI.value == "maliki"
    assert FiqhSchool.HANBALI.value == "hanbali"


def test_scope_string_values() -> None:
    assert RulingScope.PRODUCT.value == "product"
    assert RulingScope.STRATEGY.value == "strategy"
    assert RulingScope.POLICY.value == "policy"
    assert RulingScope.CERTIFICATION.value == "certification"


def test_outcome_string_values() -> None:
    assert RulingOutcome.PERMISSIBLE.value == "permissible"
    assert RulingOutcome.IMPERMISSIBLE.value == "impermissible"
    assert RulingOutcome.PERMISSIBLE_WITH_CONDITIONS.value == "permissible_with_conditions"
    assert RulingOutcome.DEFERRED.value == "deferred"


# ---------------------------------------------------------------------------
# Render output
# ---------------------------------------------------------------------------


def test_render_permissible_ruling() -> None:
    r = _ruling(
        votes=(
            _vote("A", FiqhSchool.HANAFI, RulingOutcome.PERMISSIBLE),
            _vote("B", FiqhSchool.SHAFII, RulingOutcome.PERMISSIBLE),
            _vote("C", FiqhSchool.MALIKI, RulingOutcome.PERMISSIBLE),
        )
    )
    text = render_ruling(r)
    assert "✅" in text
    assert "PERMISSIBLE" in text
    assert "SSB-2026-Q2-001" in text
    assert "commodity-ETF screener" in text


def test_render_impermissible_ruling() -> None:
    r = _ruling(
        votes=(
            _vote("A", FiqhSchool.HANAFI, RulingOutcome.PERMISSIBLE),
            _vote("B", FiqhSchool.SHAFII, RulingOutcome.IMPERMISSIBLE, "violates riba rule"),
            _vote("C", FiqhSchool.MALIKI, RulingOutcome.PERMISSIBLE),
        )
    )
    text = render_ruling(r)
    assert "❌" in text
    assert "IMPERMISSIBLE" in text
    assert "violates riba rule" in text


def test_render_conditional_ruling_includes_conditions() -> None:
    r = _ruling(
        votes=(
            _vote("A", FiqhSchool.HANAFI, RulingOutcome.PERMISSIBLE),
            _vote(
                "B",
                FiqhSchool.SHAFII,
                RulingOutcome.PERMISSIBLE_WITH_CONDITIONS,
                "operator must enable X",
            ),
            _vote("C", FiqhSchool.MALIKI, RulingOutcome.PERMISSIBLE),
        ),
        conditions=("operator must enable X", "Y must be reviewed annually"),
    )
    text = render_ruling(r)
    assert "⚠️" in text
    assert "PERMISSIBLE_WITH_CONDITIONS" in text
    assert "operator must enable X" in text
    assert "Y must be reviewed annually" in text


def test_render_deferred_ruling() -> None:
    r = _ruling(
        votes=(
            _vote("A", FiqhSchool.HANAFI, RulingOutcome.DEFERRED),
            _vote("B", FiqhSchool.SHAFII, RulingOutcome.DEFERRED),
            _vote("C", FiqhSchool.MALIKI, RulingOutcome.DEFERRED),
        )
    )
    text = render_ruling(r)
    assert "⏳" in text
    assert "DEFERRED" in text


def test_render_ruling_no_operator_pii() -> None:
    """Pin no-PII contract: ruling never references operator user / portfolio."""

    r = _ruling(
        votes=(
            _vote("A", FiqhSchool.HANAFI),
            _vote("B", FiqhSchool.SHAFII),
            _vote("C", FiqhSchool.MALIKI),
        )
    )
    text = render_ruling(r)
    assert "user_id" not in text
    assert "portfolio" not in text
    assert "balance" not in text


def test_render_board_composition_valid() -> None:
    result = validate_board(_diverse_board(), now=_NOW)
    text = render_board_composition(result)
    assert "✅" in text
    assert "VALID" in text
    assert "active members: 3" in text
    assert "schools represented: 3" in text


def test_render_board_composition_invalid() -> None:
    members = (
        _member(name="A", school=FiqhSchool.HANAFI),
        _member(name="B", school=FiqhSchool.HANAFI),
    )
    result = validate_board(members, now=_NOW)
    text = render_board_composition(result)
    assert "❌" in text
    assert "INVALID" in text
    assert "failures:" in text


# ---------------------------------------------------------------------------
# End-to-end realistic scenarios
# ---------------------------------------------------------------------------


def test_typical_quarterly_meeting_lifecycle() -> None:
    """A typical quarter: validate board → issue ruling on a new product →
    confirm next quarterly review trigger fires at 91d."""

    board = _diverse_board()
    composition = validate_board(board, now=_NOW)
    assert composition.is_valid

    # The board votes on a new product
    ruling = Ruling(
        ruling_id="SSB-2026-Q2-001",
        scope=RulingScope.PRODUCT,
        subject="Wave 1.G commodity-ETF screener",
        description="Reviewing the commodity ETF Shariah screen for halal compliance",
        issued_at=_NOW,
        votes=(
            _vote("Mufti Hanafi", FiqhSchool.HANAFI, RulingOutcome.PERMISSIBLE),
            _vote("Mufti Shafii", FiqhSchool.SHAFII, RulingOutcome.PERMISSIBLE),
            _vote(
                "Mufti Maliki",
                FiqhSchool.MALIKI,
                RulingOutcome.PERMISSIBLE_WITH_CONDITIONS,
                "no leveraged commodities",
            ),
        ),
        conditions=("operator must keep leveraged-commodity gate enabled",),
    )
    assert ruling.consensus() is RulingOutcome.PERMISSIBLE_WITH_CONDITIONS

    # No review needed for the next ~3 months
    assert needs_quarterly_review((ruling,), now=_NOW + timedelta(days=80)) is False
    # Review is due after 91 days
    assert needs_quarterly_review((ruling,), now=_NOW + timedelta(days=91)) is True


def test_dissenting_minority_blocks_pass() -> None:
    """A 2/3 supermajority pass is overridden by a single dissent."""

    votes = (
        _vote("Mufti A", FiqhSchool.HANAFI, RulingOutcome.PERMISSIBLE),
        _vote("Mufti B", FiqhSchool.SHAFII, RulingOutcome.PERMISSIBLE),
        _vote(
            "Mufti C",
            FiqhSchool.MALIKI,
            RulingOutcome.IMPERMISSIBLE,
            "underlying receivables breach AAOIFI Standard 17",
        ),
    )
    r = _ruling(votes=votes)
    # Conservative tiebreak — IMPERMISSIBLE wins over 2-of-3 PERMISSIBLE
    assert r.consensus() is RulingOutcome.IMPERMISSIBLE
