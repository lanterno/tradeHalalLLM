"""Tests for the KYC/AML state engine."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from halal_trader.web.kyc import (
    Activity,
    JurisdictionRequirement,
    KYCLevel,
    KYCPolicy,
    KYCStatus,
    RiskLevel,
    SanctionsOutcome,
    UserKYCState,
    default_policy,
    is_expired,
    permits,
    render_decision,
    render_user_state,
)

_NOW = datetime(2026, 5, 1, tzinfo=UTC)


def _state(
    *,
    user_id: str = "user-1",
    jurisdiction: str = "US",
    level: KYCLevel = KYCLevel.IDENTITY_VERIFIED,
    status: KYCStatus = KYCStatus.VERIFIED,
    risk_level: RiskLevel = RiskLevel.LOW,
    sanctions_outcome: SanctionsOutcome = SanctionsOutcome.CLEAR,
    verified_at: datetime | None = None,
) -> UserKYCState:
    return UserKYCState(
        user_id=user_id,
        jurisdiction=jurisdiction,
        level=level,
        status=status,
        risk_level=risk_level,
        sanctions_outcome=sanctions_outcome,
        verified_at=verified_at if verified_at is not None else (_NOW - timedelta(days=30)),
    )


# ---------------------------------------------------------------------------
# Level ordering
# ---------------------------------------------------------------------------


def test_level_int_value_ordering() -> None:
    assert KYCLevel.NONE.int_value < KYCLevel.EMAIL_VERIFIED.int_value
    assert KYCLevel.EMAIL_VERIFIED.int_value < KYCLevel.IDENTITY_VERIFIED.int_value
    assert KYCLevel.IDENTITY_VERIFIED.int_value < KYCLevel.ADDRESS_VERIFIED.int_value
    assert KYCLevel.ADDRESS_VERIFIED.int_value < KYCLevel.ENHANCED_DUE_DILIGENCE.int_value


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------


def test_default_policy_includes_supported_jurisdictions() -> None:
    p = default_policy()
    assert "US" in p.jurisdictions
    assert "GB" in p.jurisdictions
    assert "EU" in p.jurisdictions
    assert "AE" in p.jurisdictions
    assert "SA" in p.jurisdictions
    assert "PK" in p.jurisdictions
    assert "MY" in p.jurisdictions


def test_default_policy_uae_requires_address_verified() -> None:
    """Pin: UAE / EU bump to ADDRESS_VERIFIED; US is IDENTITY_VERIFIED."""

    p = default_policy()
    assert p.jurisdictions["AE"].minimum_level_for_real_money is KYCLevel.ADDRESS_VERIFIED
    assert p.jurisdictions["US"].minimum_level_for_real_money is KYCLevel.IDENTITY_VERIFIED


def test_default_policy_expiry_is_one_year() -> None:
    p = default_policy()
    assert p.expiry_days == 365


def test_policy_rejects_zero_expiry() -> None:
    with pytest.raises(ValueError, match="expiry_days"):
        KYCPolicy(expiry_days=0)


def test_policy_rejects_negative_expiry() -> None:
    with pytest.raises(ValueError, match="expiry_days"):
        KYCPolicy(expiry_days=-1)


def test_jurisdiction_requirement_rejects_empty_jurisdiction() -> None:
    with pytest.raises(ValueError, match="jurisdiction"):
        JurisdictionRequirement(jurisdiction="")


# ---------------------------------------------------------------------------
# UserKYCState validation
# ---------------------------------------------------------------------------


def test_state_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        _state(user_id="")


def test_state_rejects_empty_jurisdiction() -> None:
    with pytest.raises(ValueError, match="jurisdiction"):
        _state(jurisdiction="")


def test_state_rejects_naive_verified_at() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _state(verified_at=datetime(2026, 5, 1))


def test_state_accepts_none_verified_at() -> None:
    s = UserKYCState(
        user_id="user-1",
        jurisdiction="US",
        level=KYCLevel.NONE,
        status=KYCStatus.NOT_STARTED,
        risk_level=RiskLevel.LOW,
        sanctions_outcome=SanctionsOutcome.CLEAR,
        verified_at=None,
    )
    assert s.verified_at is None


# ---------------------------------------------------------------------------
# is_expired
# ---------------------------------------------------------------------------


def test_is_expired_returns_false_for_never_verified() -> None:
    s = _state(verified_at=None, status=KYCStatus.NOT_STARTED)
    assert is_expired(s, now=_NOW, policy=default_policy()) is False


def test_is_expired_returns_false_within_horizon() -> None:
    s = _state(verified_at=_NOW - timedelta(days=100))
    assert is_expired(s, now=_NOW, policy=default_policy()) is False


def test_is_expired_returns_true_past_horizon() -> None:
    s = _state(verified_at=_NOW - timedelta(days=400))
    assert is_expired(s, now=_NOW, policy=default_policy()) is True


def test_is_expired_at_exact_horizon_inclusive() -> None:
    """Pin: at exactly 365 days, the verification has expired."""

    s = _state(verified_at=_NOW - timedelta(days=365))
    assert is_expired(s, now=_NOW, policy=default_policy()) is True


def test_is_expired_just_inside_horizon_is_false() -> None:
    s = _state(verified_at=_NOW - timedelta(days=364, hours=23))
    assert is_expired(s, now=_NOW, policy=default_policy()) is False


def test_is_expired_rejects_naive_now() -> None:
    s = _state()
    with pytest.raises(ValueError, match="timezone-aware"):
        is_expired(s, now=datetime(2026, 5, 1), policy=default_policy())


def test_custom_expiry_flows_through() -> None:
    strict = KYCPolicy(expiry_days=180)
    s = _state(verified_at=_NOW - timedelta(days=200))
    assert is_expired(s, now=_NOW, policy=strict) is True


# ---------------------------------------------------------------------------
# Sanctions MATCH gate
# ---------------------------------------------------------------------------


def test_sanctions_match_blocks_real_money_trading() -> None:
    """The most-restrictive gate."""

    s = _state(sanctions_outcome=SanctionsOutcome.MATCH)
    decision = permits(s, activity=Activity.REAL_MONEY_TRADING, now=_NOW, policy=default_policy())
    assert decision.allowed is False
    assert "sanctions" in decision.reason.lower()


def test_sanctions_match_blocks_paper_trading() -> None:
    """Pin: even paper trading blocked under MATCH (only signup allowed)."""

    s = _state(sanctions_outcome=SanctionsOutcome.MATCH)
    decision = permits(s, activity=Activity.PAPER_TRADING, now=_NOW, policy=default_policy())
    assert decision.allowed is False


def test_sanctions_match_blocks_withdraw() -> None:
    """Pin: even WITHDRAW blocked under sanctions match — funds stay
    pending compliance review (this is the law)."""

    s = _state(sanctions_outcome=SanctionsOutcome.MATCH)
    decision = permits(s, activity=Activity.WITHDRAW, now=_NOW, policy=default_policy())
    assert decision.allowed is False


def test_sanctions_match_permits_signup() -> None:
    """Pin: SIGNUP must remain available so user can be notified."""

    s = _state(sanctions_outcome=SanctionsOutcome.MATCH)
    decision = permits(s, activity=Activity.SIGNUP, now=_NOW, policy=default_policy())
    assert decision.allowed is True


def test_false_positive_treated_as_clear() -> None:
    """Pin: FALSE_POSITIVE (set by compliance ops after review) → CLEAR for gating."""

    s = _state(sanctions_outcome=SanctionsOutcome.FALSE_POSITIVE)
    decision = permits(s, activity=Activity.REAL_MONEY_TRADING, now=_NOW, policy=default_policy())
    assert decision.allowed is True


# ---------------------------------------------------------------------------
# KYC-free activities
# ---------------------------------------------------------------------------


def test_signup_always_permitted_when_clear() -> None:
    s = _state(level=KYCLevel.NONE, status=KYCStatus.NOT_STARTED, verified_at=None)
    decision = permits(s, activity=Activity.SIGNUP, now=_NOW, policy=default_policy())
    assert decision.allowed is True


def test_demo_trading_permitted_with_no_kyc() -> None:
    s = _state(level=KYCLevel.NONE, status=KYCStatus.NOT_STARTED, verified_at=None)
    decision = permits(s, activity=Activity.DEMO_TRADING, now=_NOW, policy=default_policy())
    assert decision.allowed is True


def test_paper_trading_permitted_with_no_kyc() -> None:
    s = _state(level=KYCLevel.NONE, status=KYCStatus.NOT_STARTED, verified_at=None)
    decision = permits(s, activity=Activity.PAPER_TRADING, now=_NOW, policy=default_policy())
    assert decision.allowed is True


# ---------------------------------------------------------------------------
# Real-money trading gates
# ---------------------------------------------------------------------------


def test_real_money_trading_blocked_when_not_started() -> None:
    s = _state(level=KYCLevel.NONE, status=KYCStatus.NOT_STARTED, verified_at=None)
    decision = permits(s, activity=Activity.REAL_MONEY_TRADING, now=_NOW, policy=default_policy())
    assert decision.allowed is False
    assert "NOT_STARTED" in decision.reason


def test_real_money_trading_blocked_when_in_progress() -> None:
    s = _state(level=KYCLevel.EMAIL_VERIFIED, status=KYCStatus.IN_PROGRESS, verified_at=None)
    decision = permits(s, activity=Activity.REAL_MONEY_TRADING, now=_NOW, policy=default_policy())
    assert decision.allowed is False
    assert "IN_PROGRESS" in decision.reason


def test_real_money_trading_blocked_when_rejected() -> None:
    s = _state(status=KYCStatus.REJECTED)
    decision = permits(s, activity=Activity.REAL_MONEY_TRADING, now=_NOW, policy=default_policy())
    assert decision.allowed is False
    assert "REJECTED" in decision.reason


def test_real_money_trading_blocked_when_under_review() -> None:
    s = _state(status=KYCStatus.UNDER_REVIEW)
    decision = permits(s, activity=Activity.REAL_MONEY_TRADING, now=_NOW, policy=default_policy())
    assert decision.allowed is False
    assert "UNDER_REVIEW" in decision.reason


def test_real_money_trading_permitted_when_us_identity_verified() -> None:
    s = _state(jurisdiction="US", level=KYCLevel.IDENTITY_VERIFIED)
    decision = permits(s, activity=Activity.REAL_MONEY_TRADING, now=_NOW, policy=default_policy())
    assert decision.allowed is True
    assert decision.required_level is KYCLevel.IDENTITY_VERIFIED


def test_real_money_trading_blocked_when_below_jurisdiction_minimum() -> None:
    """Pin: EU requires ADDRESS_VERIFIED; IDENTITY_VERIFIED isn't enough."""

    s = _state(jurisdiction="EU", level=KYCLevel.IDENTITY_VERIFIED)
    decision = permits(s, activity=Activity.REAL_MONEY_TRADING, now=_NOW, policy=default_policy())
    assert decision.allowed is False
    assert "below jurisdiction" in decision.reason
    assert decision.required_level is KYCLevel.ADDRESS_VERIFIED


def test_real_money_trading_permitted_when_eu_address_verified() -> None:
    s = _state(jurisdiction="EU", level=KYCLevel.ADDRESS_VERIFIED)
    decision = permits(s, activity=Activity.REAL_MONEY_TRADING, now=_NOW, policy=default_policy())
    assert decision.allowed is True


def test_unregistered_jurisdiction_blocks_real_money() -> None:
    """Pin: an unregistered jurisdiction blocks real money — operator
    must explicitly add a JurisdictionRequirement."""

    s = _state(jurisdiction="ZZ", level=KYCLevel.ADDRESS_VERIFIED)
    decision = permits(s, activity=Activity.REAL_MONEY_TRADING, now=_NOW, policy=default_policy())
    assert decision.allowed is False
    assert "not registered" in decision.reason


def test_real_money_deposit_follows_same_rules_as_trading() -> None:
    """Pin: deposit / trading both gated under real-money inflow."""

    s = _state(jurisdiction="EU", level=KYCLevel.IDENTITY_VERIFIED)
    decision = permits(s, activity=Activity.REAL_MONEY_DEPOSIT, now=_NOW, policy=default_policy())
    assert decision.allowed is False
    assert decision.required_level is KYCLevel.ADDRESS_VERIFIED


# ---------------------------------------------------------------------------
# Expiry gates
# ---------------------------------------------------------------------------


def test_expired_kyc_blocks_real_money_trading() -> None:
    s = _state(verified_at=_NOW - timedelta(days=400))
    decision = permits(s, activity=Activity.REAL_MONEY_TRADING, now=_NOW, policy=default_policy())
    assert decision.allowed is False
    assert "expired" in decision.reason


def test_expired_kyc_blocks_real_money_deposit() -> None:
    s = _state(verified_at=_NOW - timedelta(days=400))
    decision = permits(s, activity=Activity.REAL_MONEY_DEPOSIT, now=_NOW, policy=default_policy())
    assert decision.allowed is False
    assert "expired" in decision.reason


def test_expired_kyc_permits_withdraw() -> None:
    """Pin: expired KYC must permit WITHDRAW so users can retrieve funds.

    Trapping a user's balance during their KYC re-verification is
    operationally awful and legally questionable.
    """

    s = _state(verified_at=_NOW - timedelta(days=400))
    decision = permits(s, activity=Activity.WITHDRAW, now=_NOW, policy=default_policy())
    assert decision.allowed is True
    assert "trap" in decision.reason.lower() or "withdraw" in decision.reason.lower()


def test_expired_status_field_also_triggers_expiry_path() -> None:
    """Pin: status=EXPIRED is treated equivalent to elapsed-time-expired."""

    s = _state(status=KYCStatus.EXPIRED, verified_at=_NOW - timedelta(days=10))
    decision = permits(s, activity=Activity.REAL_MONEY_TRADING, now=_NOW, policy=default_policy())
    assert decision.allowed is False
    assert "expired" in decision.reason


# ---------------------------------------------------------------------------
# Risk-level gates
# ---------------------------------------------------------------------------


def test_high_risk_blocks_real_money_trading_without_edd() -> None:
    """Pin: HIGH risk requires ENHANCED_DUE_DILIGENCE for real-money inflow."""

    s = _state(
        jurisdiction="US",
        level=KYCLevel.IDENTITY_VERIFIED,
        risk_level=RiskLevel.HIGH,
    )
    decision = permits(s, activity=Activity.REAL_MONEY_TRADING, now=_NOW, policy=default_policy())
    assert decision.allowed is False
    assert "ENHANCED_DUE_DILIGENCE" in decision.reason
    assert decision.required_level is KYCLevel.ENHANCED_DUE_DILIGENCE


def test_high_risk_with_edd_permits_real_money_trading() -> None:
    s = _state(
        jurisdiction="US",
        level=KYCLevel.ENHANCED_DUE_DILIGENCE,
        risk_level=RiskLevel.HIGH,
    )
    decision = permits(s, activity=Activity.REAL_MONEY_TRADING, now=_NOW, policy=default_policy())
    assert decision.allowed is True


def test_medium_risk_with_identity_verified_permits_real_money_trading() -> None:
    """Pin: MEDIUM risk follows standard ladder; doesn't trigger EDD."""

    s = _state(
        jurisdiction="US",
        level=KYCLevel.IDENTITY_VERIFIED,
        risk_level=RiskLevel.MEDIUM,
    )
    decision = permits(s, activity=Activity.REAL_MONEY_TRADING, now=_NOW, policy=default_policy())
    assert decision.allowed is True


def test_high_risk_does_not_block_withdraw() -> None:
    """Pin: HIGH risk doesn't trap funds — WITHDRAW path stays open."""

    s = _state(
        jurisdiction="US",
        level=KYCLevel.IDENTITY_VERIFIED,
        risk_level=RiskLevel.HIGH,
    )
    decision = permits(s, activity=Activity.WITHDRAW, now=_NOW, policy=default_policy())
    assert decision.allowed is True


# ---------------------------------------------------------------------------
# Withdrawal flow
# ---------------------------------------------------------------------------


def test_withdraw_permitted_under_verified_kyc() -> None:
    s = _state(level=KYCLevel.IDENTITY_VERIFIED)
    decision = permits(s, activity=Activity.WITHDRAW, now=_NOW, policy=default_policy())
    assert decision.allowed is True


# ---------------------------------------------------------------------------
# permits() input validation
# ---------------------------------------------------------------------------


def test_permits_rejects_naive_now() -> None:
    s = _state()
    with pytest.raises(ValueError, match="timezone-aware"):
        permits(
            s,
            activity=Activity.REAL_MONEY_TRADING,
            now=datetime(2026, 5, 1),
            policy=default_policy(),
        )


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_state_is_frozen() -> None:
    s = _state()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.level = KYCLevel.NONE  # type: ignore[misc]


def test_decision_is_frozen() -> None:
    decision = permits(_state(), activity=Activity.SIGNUP, now=_NOW, policy=default_policy())
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.allowed = False  # type: ignore[misc]


def test_policy_is_frozen() -> None:
    p = default_policy()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.expiry_days = 30  # type: ignore[misc]


def test_jurisdiction_requirement_is_frozen() -> None:
    j = JurisdictionRequirement(jurisdiction="US")
    with pytest.raises(dataclasses.FrozenInstanceError):
        j.minimum_level_for_real_money = KYCLevel.NONE  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned for JSON / DB stability
# ---------------------------------------------------------------------------


def test_level_string_values() -> None:
    assert KYCLevel.NONE.value == "none"
    assert KYCLevel.EMAIL_VERIFIED.value == "email_verified"
    assert KYCLevel.IDENTITY_VERIFIED.value == "identity_verified"
    assert KYCLevel.ADDRESS_VERIFIED.value == "address_verified"
    assert KYCLevel.ENHANCED_DUE_DILIGENCE.value == "enhanced_due_diligence"


def test_status_string_values() -> None:
    assert KYCStatus.NOT_STARTED.value == "not_started"
    assert KYCStatus.IN_PROGRESS.value == "in_progress"
    assert KYCStatus.VERIFIED.value == "verified"
    assert KYCStatus.EXPIRED.value == "expired"
    assert KYCStatus.REJECTED.value == "rejected"
    assert KYCStatus.UNDER_REVIEW.value == "under_review"


def test_risk_level_string_values() -> None:
    assert RiskLevel.LOW.value == "low"
    assert RiskLevel.MEDIUM.value == "medium"
    assert RiskLevel.HIGH.value == "high"


def test_sanctions_outcome_string_values() -> None:
    assert SanctionsOutcome.CLEAR.value == "clear"
    assert SanctionsOutcome.MATCH.value == "match"
    assert SanctionsOutcome.FALSE_POSITIVE.value == "false_positive"


def test_activity_string_values() -> None:
    assert Activity.SIGNUP.value == "signup"
    assert Activity.DEMO_TRADING.value == "demo_trading"
    assert Activity.PAPER_TRADING.value == "paper_trading"
    assert Activity.REAL_MONEY_DEPOSIT.value == "real_money_deposit"
    assert Activity.REAL_MONEY_TRADING.value == "real_money_trading"
    assert Activity.WITHDRAW.value == "withdraw"


# ---------------------------------------------------------------------------
# Render output — pinned no-PII contract
# ---------------------------------------------------------------------------


def test_render_user_state_includes_level_status() -> None:
    s = _state()
    text = render_user_state(s)
    assert "user-1" in text
    assert "identity_verified" in text
    assert "verified" in text
    assert "US" in text
    assert "low" in text
    assert "clear" in text


def test_render_user_state_no_id_document_data() -> None:
    """Pin no-PII contract: render never includes ID document number /
    photo / address fields. The state itself doesn't carry them, but
    pin via test that the render doesn't reference them either."""

    s = _state()
    text = render_user_state(s)
    assert "passport" not in text.lower()
    assert "license" not in text.lower()
    assert "ssn" not in text.lower()


def test_render_user_state_emoji_per_status() -> None:
    s_verified = _state(status=KYCStatus.VERIFIED)
    s_rejected = _state(status=KYCStatus.REJECTED)
    s_review = _state(status=KYCStatus.UNDER_REVIEW)
    assert "✅" in render_user_state(s_verified)
    assert "❌" in render_user_state(s_rejected)
    assert "🔍" in render_user_state(s_review)


def test_render_user_state_handles_never_verified() -> None:
    s = UserKYCState(
        user_id="user-1",
        jurisdiction="US",
        level=KYCLevel.NONE,
        status=KYCStatus.NOT_STARTED,
        risk_level=RiskLevel.LOW,
        sanctions_outcome=SanctionsOutcome.CLEAR,
        verified_at=None,
    )
    text = render_user_state(s)
    assert "verified_at: never" in text


def test_render_decision_allowed() -> None:
    decision = permits(
        _state(), activity=Activity.REAL_MONEY_TRADING, now=_NOW, policy=default_policy()
    )
    text = render_decision(decision)
    assert "✅" in text
    assert "ALLOWED" in text


def test_render_decision_blocked() -> None:
    s = _state(level=KYCLevel.NONE, status=KYCStatus.NOT_STARTED, verified_at=None)
    decision = permits(s, activity=Activity.REAL_MONEY_TRADING, now=_NOW, policy=default_policy())
    text = render_decision(decision)
    assert "🚫" in text
    assert "BLOCKED" in text
    assert "reason" in text


# ---------------------------------------------------------------------------
# End-to-end realistic flows
# ---------------------------------------------------------------------------


def test_typical_us_user_journey_paper_to_real_money() -> None:
    """A US user signs up, paper-trades, completes KYC, real-trades."""

    # Pre-KYC state
    s_initial = UserKYCState(
        user_id="user-1",
        jurisdiction="US",
        level=KYCLevel.NONE,
        status=KYCStatus.NOT_STARTED,
        risk_level=RiskLevel.LOW,
        sanctions_outcome=SanctionsOutcome.CLEAR,
        verified_at=None,
    )
    p = default_policy()
    # Can sign up + paper trade
    assert permits(s_initial, activity=Activity.SIGNUP, now=_NOW, policy=p).allowed is True
    assert permits(s_initial, activity=Activity.PAPER_TRADING, now=_NOW, policy=p).allowed is True
    # Cannot real-trade
    assert (
        permits(s_initial, activity=Activity.REAL_MONEY_TRADING, now=_NOW, policy=p).allowed
        is False
    )

    # Post-KYC state
    s_verified = dataclasses.replace(
        s_initial,
        level=KYCLevel.IDENTITY_VERIFIED,
        status=KYCStatus.VERIFIED,
        verified_at=_NOW,
    )
    assert (
        permits(s_verified, activity=Activity.REAL_MONEY_TRADING, now=_NOW, policy=p).allowed
        is True
    )


def test_typical_eu_user_journey_needs_address_verification() -> None:
    """An EU user needs ADDRESS_VERIFIED (not just IDENTITY)."""

    s = _state(jurisdiction="EU", level=KYCLevel.IDENTITY_VERIFIED)
    p = default_policy()
    assert permits(s, activity=Activity.REAL_MONEY_TRADING, now=_NOW, policy=p).allowed is False
    s_address = dataclasses.replace(s, level=KYCLevel.ADDRESS_VERIFIED)
    assert (
        permits(s_address, activity=Activity.REAL_MONEY_TRADING, now=_NOW, policy=p).allowed is True
    )


def test_full_block_under_sanctions_match_review_then_clear() -> None:
    """Sanctions MATCH → all blocked; compliance ops marks FALSE_POSITIVE → re-opens."""

    s_match = _state(sanctions_outcome=SanctionsOutcome.MATCH)
    p = default_policy()
    for activity in (
        Activity.PAPER_TRADING,
        Activity.REAL_MONEY_DEPOSIT,
        Activity.REAL_MONEY_TRADING,
        Activity.WITHDRAW,
    ):
        assert permits(s_match, activity=activity, now=_NOW, policy=p).allowed is False

    s_cleared = dataclasses.replace(s_match, sanctions_outcome=SanctionsOutcome.FALSE_POSITIVE)
    for activity in (
        Activity.PAPER_TRADING,
        Activity.REAL_MONEY_TRADING,
        Activity.WITHDRAW,
    ):
        assert permits(s_cleared, activity=activity, now=_NOW, policy=p).allowed is True
