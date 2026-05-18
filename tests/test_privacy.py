"""Tests for the GDPR/CCPA privacy engine."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from halal_trader.web.privacy import (
    CategoryHolding,
    ConsentRecord,
    DataCategory,
    DeletionAction,
    LegalBasis,
    RetentionPolicy,
    build_deletion_plan,
    build_export_plan,
    default_retention_policy,
    is_overdue_default,
    render_deletion_plan,
    render_export_plan,
    revoke_consent,
)

_NOW = datetime(2026, 5, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# RetentionPolicy validation
# ---------------------------------------------------------------------------


def test_default_policy_covers_every_category() -> None:
    p = default_retention_policy()
    for cat in DataCategory:
        assert cat in p.retention_days
        assert cat in p.legal_basis


def test_default_policy_legal_obligation_categories() -> None:
    p = default_retention_policy()
    # Trade audit + halal audit + purification ledger are legal-obligation
    assert p.legal_basis[DataCategory.TRADING_HISTORY] is LegalBasis.LEGAL_OBLIGATION
    assert p.legal_basis[DataCategory.HALAL_AUDIT] is LegalBasis.LEGAL_OBLIGATION
    assert p.legal_basis[DataCategory.PURIFICATION_LEDGER] is LegalBasis.LEGAL_OBLIGATION


def test_default_policy_consent_categories() -> None:
    p = default_retention_policy()
    assert p.legal_basis[DataCategory.LLM_PROMPTS] is LegalBasis.CONSENT
    assert p.legal_basis[DataCategory.MARKETING_ANALYTICS] is LegalBasis.CONSENT


def test_default_policy_retention_asymmetry() -> None:
    """Trade audit retained 7y; marketing analytics 90d."""

    p = default_retention_policy()
    assert p.retention_days[DataCategory.TRADING_HISTORY] == 365 * 7
    assert p.retention_days[DataCategory.MARKETING_ANALYTICS] == 90
    assert p.retention_days[DataCategory.USAGE_TELEMETRY] == 30


def test_policy_rejects_partial_retention_map() -> None:
    partial = {DataCategory.PII: 100}
    with pytest.raises(ValueError, match="retention_days missing"):
        RetentionPolicy(retention_days=partial)


def test_policy_rejects_partial_legal_basis_map() -> None:
    p = default_retention_policy()
    partial_basis = {DataCategory.PII: LegalBasis.CONTRACT}
    with pytest.raises(ValueError, match="legal_basis missing"):
        RetentionPolicy(retention_days=p.retention_days, legal_basis=partial_basis)


def test_policy_rejects_zero_retention() -> None:
    p = default_retention_policy()
    bad = dict(p.retention_days)
    bad[DataCategory.PII] = 0
    with pytest.raises(ValueError, match="must be positive"):
        RetentionPolicy(retention_days=bad)


def test_policy_is_overdue_returns_true_when_past_horizon() -> None:
    p = default_retention_policy()
    # USAGE_TELEMETRY default is 30d
    recorded = _NOW - timedelta(days=31)
    assert p.is_overdue(DataCategory.USAGE_TELEMETRY, recorded_at=recorded, now=_NOW) is True


def test_policy_is_overdue_at_exact_horizon_is_inclusive() -> None:
    """Pin: at exactly retention_days, the row is overdue."""

    p = default_retention_policy()
    recorded = _NOW - timedelta(days=30)
    assert p.is_overdue(DataCategory.USAGE_TELEMETRY, recorded_at=recorded, now=_NOW) is True


def test_policy_is_overdue_just_inside_horizon_is_false() -> None:
    p = default_retention_policy()
    recorded = _NOW - timedelta(days=29, hours=23)
    assert p.is_overdue(DataCategory.USAGE_TELEMETRY, recorded_at=recorded, now=_NOW) is False


def test_policy_is_overdue_rejects_naive_datetime() -> None:
    p = default_retention_policy()
    with pytest.raises(ValueError, match="timezone-aware"):
        p.is_overdue(DataCategory.PII, recorded_at=datetime(2026, 5, 1), now=_NOW)


def test_is_overdue_default_helper() -> None:
    recorded = _NOW - timedelta(days=400)
    assert (
        is_overdue_default(DataCategory.MARKETING_ANALYTICS, recorded_at=recorded, now=_NOW) is True
    )


# ---------------------------------------------------------------------------
# CategoryHolding validation
# ---------------------------------------------------------------------------


def test_holding_rejects_negative_count() -> None:
    with pytest.raises(ValueError, match="count"):
        CategoryHolding(category=DataCategory.PII, count=-1)


def test_holding_accepts_zero_count() -> None:
    """Zero count is valid — used to indicate "no data in this category"."""

    h = CategoryHolding(category=DataCategory.PII, count=0)
    assert h.count == 0


# ---------------------------------------------------------------------------
# ConsentRecord validation
# ---------------------------------------------------------------------------


def test_consent_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        ConsentRecord(
            user_id="",
            category=DataCategory.MARKETING_ANALYTICS,
            granted_at=_NOW,
        )


def test_consent_rejects_naive_granted_at() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        ConsentRecord(
            user_id="user-1",
            category=DataCategory.MARKETING_ANALYTICS,
            granted_at=datetime(2026, 5, 1),
        )


def test_consent_rejects_revoked_before_granted() -> None:
    with pytest.raises(ValueError, match="revoked_at cannot be before"):
        ConsentRecord(
            user_id="user-1",
            category=DataCategory.MARKETING_ANALYTICS,
            granted_at=_NOW,
            revoked_at=_NOW - timedelta(days=1),
        )


def test_consent_is_active_when_revoked_none() -> None:
    c = ConsentRecord(
        user_id="user-1",
        category=DataCategory.MARKETING_ANALYTICS,
        granted_at=_NOW,
    )
    assert c.is_active is True


def test_consent_is_inactive_after_revoke() -> None:
    c = ConsentRecord(
        user_id="user-1",
        category=DataCategory.MARKETING_ANALYTICS,
        granted_at=_NOW - timedelta(days=10),
        revoked_at=_NOW,
    )
    assert c.is_active is False


def test_revoke_consent_returns_new_record_with_revoked_at_set() -> None:
    c = ConsentRecord(
        user_id="user-1",
        category=DataCategory.MARKETING_ANALYTICS,
        granted_at=_NOW - timedelta(days=10),
    )
    revoked = revoke_consent(c, now=_NOW)
    assert revoked.revoked_at == _NOW
    assert revoked.granted_at == c.granted_at
    assert c.revoked_at is None  # input unchanged


def test_revoke_already_revoked_consent_is_noop() -> None:
    """Pin: re-revoking doesn't advance the revoked_at timestamp."""

    earlier = _NOW - timedelta(days=5)
    c = ConsentRecord(
        user_id="user-1",
        category=DataCategory.MARKETING_ANALYTICS,
        granted_at=_NOW - timedelta(days=10),
        revoked_at=earlier,
    )
    re_revoked = revoke_consent(c, now=_NOW)
    assert re_revoked is c
    assert re_revoked.revoked_at == earlier


def test_revoke_rejects_naive_now() -> None:
    c = ConsentRecord(
        user_id="user-1",
        category=DataCategory.MARKETING_ANALYTICS,
        granted_at=_NOW,
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        revoke_consent(c, now=datetime(2026, 5, 1))


# ---------------------------------------------------------------------------
# Export plan
# ---------------------------------------------------------------------------


def test_export_plan_includes_every_populated_category() -> None:
    """Pin: export covers every category with count > 0."""

    holdings = (
        CategoryHolding(category=DataCategory.PII, count=1),
        CategoryHolding(category=DataCategory.TRADING_HISTORY, count=42),
        CategoryHolding(category=DataCategory.LLM_PROMPTS, count=10),
    )
    plan = build_export_plan(
        user_id="user-1",
        holdings=holdings,
        policy=default_retention_policy(),
        requested_at=_NOW,
    )
    cats = {e.category for e in plan.entries}
    assert DataCategory.PII in cats
    assert DataCategory.TRADING_HISTORY in cats
    assert DataCategory.LLM_PROMPTS in cats
    assert plan.total_count == 53


def test_export_plan_excludes_zero_count_categories() -> None:
    """Pin: zero-count categories are omitted (nothing to export)."""

    holdings = (
        CategoryHolding(category=DataCategory.PII, count=1),
        CategoryHolding(category=DataCategory.LLM_PROMPTS, count=0),
    )
    plan = build_export_plan(
        user_id="user-1",
        holdings=holdings,
        policy=default_retention_policy(),
        requested_at=_NOW,
    )
    cats = {e.category for e in plan.entries}
    assert DataCategory.LLM_PROMPTS not in cats
    assert DataCategory.PII in cats


def test_export_plan_with_no_holdings_is_empty() -> None:
    plan = build_export_plan(
        user_id="user-1",
        holdings=(),
        policy=default_retention_policy(),
        requested_at=_NOW,
    )
    assert plan.entries == ()
    assert plan.total_count == 0


def test_export_plan_carries_legal_basis_per_category() -> None:
    holdings = (
        CategoryHolding(category=DataCategory.TRADING_HISTORY, count=1),
        CategoryHolding(category=DataCategory.MARKETING_ANALYTICS, count=1),
    )
    plan = build_export_plan(
        user_id="user-1",
        holdings=holdings,
        policy=default_retention_policy(),
        requested_at=_NOW,
    )
    by_cat = {e.category: e for e in plan.entries}
    assert by_cat[DataCategory.TRADING_HISTORY].legal_basis is LegalBasis.LEGAL_OBLIGATION
    assert by_cat[DataCategory.MARKETING_ANALYTICS].legal_basis is LegalBasis.CONSENT


def test_export_plan_carries_recorded_at_timestamps() -> None:
    holdings = (
        CategoryHolding(
            category=DataCategory.PII,
            count=1,
            oldest_recorded_at=_NOW - timedelta(days=100),
            newest_recorded_at=_NOW - timedelta(days=1),
        ),
    )
    plan = build_export_plan(
        user_id="user-1",
        holdings=holdings,
        policy=default_retention_policy(),
        requested_at=_NOW,
    )
    e = plan.entries[0]
    assert e.oldest_recorded_at == _NOW - timedelta(days=100)
    assert e.newest_recorded_at == _NOW - timedelta(days=1)


def test_export_plan_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        build_export_plan(
            user_id="",
            holdings=(),
            policy=default_retention_policy(),
            requested_at=_NOW,
        )


def test_export_plan_rejects_naive_requested_at() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        build_export_plan(
            user_id="user-1",
            holdings=(),
            policy=default_retention_policy(),
            requested_at=datetime(2026, 5, 1),
        )


# ---------------------------------------------------------------------------
# Deletion plan — legal-obligation pin
# ---------------------------------------------------------------------------


def test_legal_obligation_categories_anonymise_not_delete() -> None:
    """The load-bearing pin: trading_history can't be hard-deleted but
    can be anonymised."""

    holdings = (CategoryHolding(category=DataCategory.TRADING_HISTORY, count=42),)
    plan = build_deletion_plan(
        user_id="user-1",
        holdings=holdings,
        policy=default_retention_policy(),
        requested_at=_NOW,
    )
    e = plan.entries[0]
    assert e.action is DeletionAction.ANONYMISE
    assert "legal obligation" in e.reason


def test_legal_obligation_denied_when_anonymise_disabled() -> None:
    """If the operator's policy doesn't permit anonymisation as
    fallback, legal-obligation categories return DENIED."""

    holdings = (CategoryHolding(category=DataCategory.TRADING_HISTORY, count=42),)
    plan = build_deletion_plan(
        user_id="user-1",
        holdings=holdings,
        policy=default_retention_policy(),
        requested_at=_NOW,
        operator_allows_anonymise=False,
    )
    e = plan.entries[0]
    assert e.action is DeletionAction.DENIED
    assert "does not permit anonymisation" in e.reason


def test_halal_audit_is_anonymised_not_deleted() -> None:
    """Pin: shariah-audit ledger has the same legal-obligation flag."""

    holdings = (CategoryHolding(category=DataCategory.HALAL_AUDIT, count=10),)
    plan = build_deletion_plan(
        user_id="user-1",
        holdings=holdings,
        policy=default_retention_policy(),
        requested_at=_NOW,
    )
    assert plan.entries[0].action is DeletionAction.ANONYMISE


def test_purification_ledger_is_anonymised_not_deleted() -> None:
    holdings = (CategoryHolding(category=DataCategory.PURIFICATION_LEDGER, count=5),)
    plan = build_deletion_plan(
        user_id="user-1",
        holdings=holdings,
        policy=default_retention_policy(),
        requested_at=_NOW,
    )
    assert plan.entries[0].action is DeletionAction.ANONYMISE


# ---------------------------------------------------------------------------
# Deletion plan — consent pin
# ---------------------------------------------------------------------------


def test_consent_categories_hard_delete() -> None:
    """Consent-basis data must hard-delete on Article 17 request."""

    holdings = (
        CategoryHolding(category=DataCategory.LLM_PROMPTS, count=20),
        CategoryHolding(category=DataCategory.MARKETING_ANALYTICS, count=5),
    )
    plan = build_deletion_plan(
        user_id="user-1",
        holdings=holdings,
        policy=default_retention_policy(),
        requested_at=_NOW,
    )
    for e in plan.entries:
        assert e.action is DeletionAction.HARD_DELETE


def test_contract_categories_hard_delete() -> None:
    """Contract-basis data hard-deletes on user account closure."""

    holdings = (
        CategoryHolding(category=DataCategory.PII, count=1),
        CategoryHolding(category=DataCategory.AUTH_CREDENTIALS, count=1),
        CategoryHolding(category=DataCategory.BROKER_KEYS, count=3),
    )
    plan = build_deletion_plan(
        user_id="user-1",
        holdings=holdings,
        policy=default_retention_policy(),
        requested_at=_NOW,
    )
    for e in plan.entries:
        assert e.action is DeletionAction.HARD_DELETE


def test_legitimate_interest_overridden_by_article_17() -> None:
    """Pin: legitimate-interest is overridden by an explicit
    Article 17 request — user's right to erasure outweighs the
    legitimate-interest balancing test."""

    holdings = (
        CategoryHolding(category=DataCategory.USAGE_TELEMETRY, count=100),
        CategoryHolding(category=DataCategory.DEVICE_FINGERPRINT, count=5),
    )
    plan = build_deletion_plan(
        user_id="user-1",
        holdings=holdings,
        policy=default_retention_policy(),
        requested_at=_NOW,
    )
    for e in plan.entries:
        assert e.action is DeletionAction.HARD_DELETE
        assert "Article 17" in e.reason


# ---------------------------------------------------------------------------
# Deletion plan — count-based filtering + multi-category
# ---------------------------------------------------------------------------


def test_zero_count_categories_omitted_from_deletion_plan() -> None:
    holdings = (
        CategoryHolding(category=DataCategory.PII, count=1),
        CategoryHolding(category=DataCategory.LLM_PROMPTS, count=0),
    )
    plan = build_deletion_plan(
        user_id="user-1",
        holdings=holdings,
        policy=default_retention_policy(),
        requested_at=_NOW,
    )
    cats = {e.category for e in plan.entries}
    assert DataCategory.LLM_PROMPTS not in cats
    assert DataCategory.PII in cats


def test_mixed_category_deletion_plan_has_correct_aggregate_counts() -> None:
    holdings = (
        CategoryHolding(category=DataCategory.PII, count=1),  # contract → delete
        CategoryHolding(category=DataCategory.TRADING_HISTORY, count=42),  # legal → anonymise
        CategoryHolding(category=DataCategory.LLM_PROMPTS, count=10),  # consent → delete
    )
    plan = build_deletion_plan(
        user_id="user-1",
        holdings=holdings,
        policy=default_retention_policy(),
        requested_at=_NOW,
    )
    assert plan.deleted_count == 11  # 1 + 10
    assert plan.anonymised_count == 42
    assert plan.denied_count == 0


def test_deletion_plan_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        build_deletion_plan(
            user_id="",
            holdings=(),
            policy=default_retention_policy(),
            requested_at=_NOW,
        )


def test_deletion_plan_rejects_naive_requested_at() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        build_deletion_plan(
            user_id="user-1",
            holdings=(),
            policy=default_retention_policy(),
            requested_at=datetime(2026, 5, 1),
        )


# ---------------------------------------------------------------------------
# Deletion plan — consent records considered
# ---------------------------------------------------------------------------


def test_deletion_plan_with_revoked_consent_still_hard_deletes() -> None:
    """Even if consent was previously revoked, the consent-basis
    category still hard-deletes on Article 17 request."""

    holdings = (CategoryHolding(category=DataCategory.LLM_PROMPTS, count=10),)
    consents = (
        ConsentRecord(
            user_id="user-1",
            category=DataCategory.LLM_PROMPTS,
            granted_at=_NOW - timedelta(days=10),
            revoked_at=_NOW - timedelta(days=1),
        ),
    )
    plan = build_deletion_plan(
        user_id="user-1",
        holdings=holdings,
        policy=default_retention_policy(),
        requested_at=_NOW,
        consents=consents,
    )
    e = plan.entries[0]
    assert e.action is DeletionAction.HARD_DELETE


def test_deletion_plan_filters_consents_by_user_id() -> None:
    """Pin: another user's consent records are ignored."""

    holdings = (CategoryHolding(category=DataCategory.LLM_PROMPTS, count=5),)
    consents = (
        ConsentRecord(
            user_id="user-OTHER",  # different user
            category=DataCategory.LLM_PROMPTS,
            granted_at=_NOW - timedelta(days=10),
        ),
    )
    plan = build_deletion_plan(
        user_id="user-1",
        holdings=holdings,
        policy=default_retention_policy(),
        requested_at=_NOW,
        consents=consents,
    )
    e = plan.entries[0]
    # still hard-deletes, but the reason cites "no active consent"
    # because user-1's consent record is absent
    assert e.action is DeletionAction.HARD_DELETE
    assert "no active consent" in e.reason


# ---------------------------------------------------------------------------
# End-to-end realistic flow
# ---------------------------------------------------------------------------


def test_full_deletion_lifecycle_for_typical_user() -> None:
    """A user with a typical mix: PII + auth + broker keys + trades +
    halal audit + purification + LLM prompts + telemetry + marketing.
    Article 17 request → some delete, some anonymise."""

    holdings = (
        CategoryHolding(category=DataCategory.PII, count=1),
        CategoryHolding(category=DataCategory.AUTH_CREDENTIALS, count=1),
        CategoryHolding(category=DataCategory.BROKER_KEYS, count=2),
        CategoryHolding(category=DataCategory.TRADING_HISTORY, count=156),
        CategoryHolding(category=DataCategory.HALAL_AUDIT, count=156),
        CategoryHolding(category=DataCategory.PURIFICATION_LEDGER, count=4),
        CategoryHolding(category=DataCategory.LLM_PROMPTS, count=80),
        CategoryHolding(category=DataCategory.LLM_RESPONSES, count=80),
        CategoryHolding(category=DataCategory.USAGE_TELEMETRY, count=300),
        CategoryHolding(category=DataCategory.MARKETING_ANALYTICS, count=12),
    )
    plan = build_deletion_plan(
        user_id="user-1",
        holdings=holdings,
        policy=default_retention_policy(),
        requested_at=_NOW,
    )
    # PII + AUTH + BROKER_KEYS (contract) hard-delete = 1+1+2 = 4
    # LLM_PROMPTS + LLM_RESPONSES + MARKETING (consent) hard-delete = 80+80+12 = 172
    # USAGE_TELEMETRY (legitimate_interest, overridden) = 300
    # Total hard-deleted: 4 + 172 + 300 = 476
    assert plan.deleted_count == 476
    # TRADING + HALAL_AUDIT + PURIFICATION (legal_obligation) anonymise = 156+156+4 = 316
    assert plan.anonymised_count == 316
    assert plan.denied_count == 0


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_export_plan_is_frozen() -> None:
    plan = build_export_plan(
        user_id="user-1",
        holdings=(),
        policy=default_retention_policy(),
        requested_at=_NOW,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.user_id = "other"  # type: ignore[misc]


def test_deletion_plan_is_frozen() -> None:
    plan = build_deletion_plan(
        user_id="user-1",
        holdings=(),
        policy=default_retention_policy(),
        requested_at=_NOW,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.user_id = "other"  # type: ignore[misc]


def test_consent_record_is_frozen() -> None:
    c = ConsentRecord(
        user_id="user-1",
        category=DataCategory.MARKETING_ANALYTICS,
        granted_at=_NOW,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.granted_at = _NOW + timedelta(days=1)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned for JSON / DB stability
# ---------------------------------------------------------------------------


def test_data_category_string_values() -> None:
    assert DataCategory.PII.value == "pii"
    assert DataCategory.TRADING_HISTORY.value == "trading_history"
    assert DataCategory.HALAL_AUDIT.value == "halal_audit"
    assert DataCategory.PURIFICATION_LEDGER.value == "purification_ledger"
    assert DataCategory.MARKETING_ANALYTICS.value == "marketing_analytics"


def test_legal_basis_string_values() -> None:
    assert LegalBasis.CONSENT.value == "consent"
    assert LegalBasis.CONTRACT.value == "contract"
    assert LegalBasis.LEGITIMATE_INTEREST.value == "legitimate_interest"
    assert LegalBasis.LEGAL_OBLIGATION.value == "legal_obligation"


def test_deletion_action_string_values() -> None:
    assert DeletionAction.HARD_DELETE.value == "hard_delete"
    assert DeletionAction.ANONYMISE.value == "anonymise"
    assert DeletionAction.DENIED.value == "denied"


# ---------------------------------------------------------------------------
# Render output — pinned no-PII contract
# ---------------------------------------------------------------------------


def test_render_export_plan_includes_summary() -> None:
    plan = build_export_plan(
        user_id="user-1",
        holdings=(CategoryHolding(category=DataCategory.PII, count=1),),
        policy=default_retention_policy(),
        requested_at=_NOW,
    )
    text = render_export_plan(plan)
    assert "user-1" in text
    assert "pii" in text
    assert "📦" in text
    assert "total rows: 1" in text


def test_render_export_plan_handles_empty() -> None:
    plan = build_export_plan(
        user_id="user-1",
        holdings=(),
        policy=default_retention_policy(),
        requested_at=_NOW,
    )
    text = render_export_plan(plan)
    assert "no data on file" in text


def test_render_deletion_plan_uses_action_emojis() -> None:
    holdings = (
        CategoryHolding(category=DataCategory.PII, count=1),
        CategoryHolding(category=DataCategory.TRADING_HISTORY, count=42),
    )
    plan = build_deletion_plan(
        user_id="user-1",
        holdings=holdings,
        policy=default_retention_policy(),
        requested_at=_NOW,
    )
    text = render_deletion_plan(plan)
    assert "🗑️" in text  # hard-delete emoji
    assert "🫥" in text  # anonymise emoji
    assert "HARD_DELETE" in text
    assert "ANONYMISE" in text


def test_render_deletion_plan_with_denied_uses_lock_emoji() -> None:
    holdings = (CategoryHolding(category=DataCategory.TRADING_HISTORY, count=10),)
    plan = build_deletion_plan(
        user_id="user-1",
        holdings=holdings,
        policy=default_retention_policy(),
        requested_at=_NOW,
        operator_allows_anonymise=False,
    )
    text = render_deletion_plan(plan)
    assert "🔒" in text
    assert "DENIED" in text


def test_render_deletion_plan_handles_empty() -> None:
    plan = build_deletion_plan(
        user_id="user-1",
        holdings=(),
        policy=default_retention_policy(),
        requested_at=_NOW,
    )
    text = render_deletion_plan(plan)
    assert "no data on file" in text


def test_render_does_not_contain_field_values() -> None:
    """Pin no-PII contract: the receipt summarises action + count, never field values."""

    holdings = (CategoryHolding(category=DataCategory.PII, count=1),)
    plan = build_export_plan(
        user_id="user-1",
        holdings=holdings,
        policy=default_retention_policy(),
        requested_at=_NOW,
    )
    text = render_export_plan(plan)
    # the receipt mentions count + category but not field values
    assert "@" not in text  # no email-shaped strings
    assert "phone" not in text
    assert "address" not in text
