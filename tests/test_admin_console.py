"""Tests for `halal_trader.web.admin_console` (Wave 3.G).

Covers the admin view-model: admin-email gate, action audit trail,
aggregation totals, tier breakdown, no-secret render contract.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.web.admin_console import (
    AdminAction,
    AdminActionRecord,
    AdminActionRequest,
    AdminAuthorizationError,
    AdminEmailList,
    UserSummary,
    audit_admin_action,
    build_admin_view,
    render_action_record,
    render_admin_view,
    render_user_summary,
)
from halal_trader.web.billing_state import SubscriptionStatus
from halal_trader.web.quotas import Tier

UTC = timezone.utc
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


# ---------------------------- AdminEmailList ---------------------------------


def test_admin_email_list_normalises_case_and_whitespace() -> None:
    lst = AdminEmailList(emails=frozenset({"Admin@Foo.com", "  bob@bar.com  "}))
    assert lst.is_admin("admin@foo.com")
    assert lst.is_admin("ADMIN@FOO.COM")
    assert lst.is_admin("bob@bar.com")
    assert lst.is_admin("  Bob@Bar.com  ")


def test_admin_email_list_rejects_non_admin() -> None:
    lst = AdminEmailList(emails=frozenset({"admin@foo.com"}))
    assert not lst.is_admin("hacker@evil.com")
    assert not lst.is_admin("admin@foo.org")


def test_admin_email_list_rejects_empty_set() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        AdminEmailList(emails=frozenset())


def test_admin_email_list_rejects_empty_email() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        AdminEmailList(emails=frozenset({""}))


def test_admin_email_list_rejects_whitespace_email() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        AdminEmailList(emails=frozenset({"   "}))


def test_admin_email_list_rejects_email_without_at() -> None:
    with pytest.raises(ValueError, match="@"):
        AdminEmailList(emails=frozenset({"notanemail"}))


def test_admin_email_list_is_frozen() -> None:
    lst = AdminEmailList(emails=frozenset({"admin@foo.com"}))
    with pytest.raises(FrozenInstanceError):
        lst.emails = frozenset({"other@bar.com"})  # type: ignore[misc]


# --------------------------- UserSummary -------------------------------------


def _make_user(**overrides: object) -> UserSummary:
    base: dict[str, object] = {
        "user_id": "user_1",
        "email": "user1@example.com",
        "effective_tier": Tier.PRO,
        "subscription_status": SubscriptionStatus.ACTIVE,
        "llm_usd_used_today": 1.50,
        "llm_usd_limit_today": 10.00,
        "halt_active": False,
        "last_active_at": T0,
        "joined_at": T0 - timedelta(days=30),
    }
    base.update(overrides)
    return UserSummary(**base)  # type: ignore[arg-type]


def test_user_summary_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        _make_user(user_id="")


def test_user_summary_rejects_empty_email() -> None:
    with pytest.raises(ValueError, match="email"):
        _make_user(email="")


def test_user_summary_rejects_email_without_at() -> None:
    with pytest.raises(ValueError, match="@"):
        _make_user(email="notanemail")


def test_user_summary_rejects_negative_llm_used() -> None:
    with pytest.raises(ValueError, match="llm_usd_used_today"):
        _make_user(llm_usd_used_today=-0.01)


def test_user_summary_rejects_negative_llm_limit() -> None:
    with pytest.raises(ValueError, match="llm_usd_limit_today"):
        _make_user(llm_usd_limit_today=-0.01)


def test_user_summary_rejects_naive_joined_at() -> None:
    with pytest.raises(ValueError, match="joined_at"):
        _make_user(joined_at=datetime(2026, 5, 1))


def test_user_summary_rejects_naive_last_active_at() -> None:
    with pytest.raises(ValueError, match="last_active_at"):
        _make_user(last_active_at=datetime(2026, 5, 1))


def test_user_summary_accepts_none_last_active_at() -> None:
    user = _make_user(last_active_at=None)
    assert user.last_active_at is None


def test_user_summary_is_frozen() -> None:
    user = _make_user()
    with pytest.raises(FrozenInstanceError):
        user.halt_active = True  # type: ignore[misc]


def test_llm_usage_fraction_basic() -> None:
    user = _make_user(llm_usd_used_today=2.5, llm_usd_limit_today=10.0)
    assert user.llm_usage_fraction == 0.25


def test_llm_usage_fraction_zero_limit_returns_zero() -> None:
    user = _make_user(llm_usd_used_today=0.0, llm_usd_limit_today=0.0)
    assert user.llm_usage_fraction == 0.0


def test_llm_usage_fraction_at_limit_is_one() -> None:
    user = _make_user(llm_usd_used_today=10.0, llm_usd_limit_today=10.0)
    assert user.llm_usage_fraction == 1.0


# ------------------------- AdminActionRequest --------------------------------


def _make_request(**overrides: object) -> AdminActionRequest:
    base: dict[str, object] = {
        "action_id": "act_1",
        "action": AdminAction.RESUME_USER,
        "target_user_id": "user_1",
        "performed_by": "admin@foo.com",
        "reason": "",
        "performed_at": T0,
    }
    base.update(overrides)
    return AdminActionRequest(**base)  # type: ignore[arg-type]


def test_request_rejects_empty_action_id() -> None:
    with pytest.raises(ValueError, match="action_id"):
        _make_request(action_id="")


def test_request_rejects_empty_target_user_id() -> None:
    with pytest.raises(ValueError, match="target_user_id"):
        _make_request(target_user_id="")


def test_request_rejects_empty_performed_by() -> None:
    with pytest.raises(ValueError, match="performed_by"):
        _make_request(performed_by="")


def test_request_rejects_performed_by_without_at() -> None:
    with pytest.raises(ValueError, match="@"):
        _make_request(performed_by="notanemail")


def test_request_rejects_naive_performed_at() -> None:
    with pytest.raises(ValueError, match="performed_at"):
        _make_request(performed_at=datetime(2026, 5, 1))


def test_request_halt_user_requires_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        _make_request(action=AdminAction.HALT_USER, reason="")


def test_request_halt_user_rejects_whitespace_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        _make_request(action=AdminAction.HALT_USER, reason="   ")


def test_request_override_tier_requires_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        _make_request(
            action=AdminAction.OVERRIDE_TIER,
            reason="",
            target_tier=Tier.PRO,
        )


def test_request_suspend_user_requires_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        _make_request(action=AdminAction.SUSPEND_USER, reason="")


def test_request_revoke_session_requires_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        _make_request(action=AdminAction.REVOKE_SESSION, reason="")


def test_request_resume_user_does_not_require_reason() -> None:
    req = _make_request(action=AdminAction.RESUME_USER, reason="")
    assert req.reason == ""


def test_request_inspect_user_does_not_require_reason() -> None:
    req = _make_request(action=AdminAction.INSPECT_USER, reason="")
    assert req.reason == ""


def test_request_override_tier_requires_target_tier() -> None:
    with pytest.raises(ValueError, match="target_tier"):
        _make_request(
            action=AdminAction.OVERRIDE_TIER,
            reason="upgraded as part of beta program",
        )


def test_request_override_tier_with_target_tier_accepted() -> None:
    req = _make_request(
        action=AdminAction.OVERRIDE_TIER,
        reason="upgraded as part of beta program",
        target_tier=Tier.ENTERPRISE,
    )
    assert req.target_tier is Tier.ENTERPRISE


def test_request_is_frozen() -> None:
    req = _make_request()
    with pytest.raises(FrozenInstanceError):
        req.target_user_id = "other"  # type: ignore[misc]


# ------------------------- audit_admin_action --------------------------------


def test_audit_admin_action_rejects_non_admin() -> None:
    admins = AdminEmailList(emails=frozenset({"admin@foo.com"}))
    req = _make_request(performed_by="hacker@evil.com")
    with pytest.raises(AdminAuthorizationError):
        audit_admin_action(req, admin_emails=admins)


def test_audit_admin_action_accepts_admin() -> None:
    admins = AdminEmailList(emails=frozenset({"admin@foo.com"}))
    req = _make_request(
        action=AdminAction.HALT_USER,
        reason="suspicious trading pattern",
    )
    record = audit_admin_action(req, admin_emails=admins)
    assert record.action is AdminAction.HALT_USER
    assert record.target_user_id == "user_1"
    assert record.performed_by == "admin@foo.com"
    assert record.reason == "suspicious trading pattern"


def test_audit_admin_action_normalises_admin_email() -> None:
    """Pin: case-insensitive admin match, normalised on the record."""

    admins = AdminEmailList(emails=frozenset({"admin@foo.com"}))
    req = _make_request(performed_by="ADMIN@FOO.COM")
    record = audit_admin_action(req, admin_emails=admins)
    assert record.performed_by == "admin@foo.com"


def test_audit_admin_action_is_deterministic() -> None:
    admins = AdminEmailList(emails=frozenset({"admin@foo.com"}))
    req = _make_request(action=AdminAction.HALT_USER, reason="suspicious activity")
    a = audit_admin_action(req, admin_emails=admins)
    b = audit_admin_action(req, admin_emails=admins)
    assert a == b


def test_audit_admin_action_authorization_error_carries_email() -> None:
    admins = AdminEmailList(emails=frozenset({"admin@foo.com"}))
    req = _make_request(performed_by="hacker@evil.com")
    with pytest.raises(AdminAuthorizationError) as exc_info:
        audit_admin_action(req, admin_emails=admins)
    assert exc_info.value.email == "hacker@evil.com"


# ------------------------- AdminActionRecord ---------------------------------


def test_admin_action_record_is_frozen() -> None:
    record = AdminActionRecord(
        action_id="act",
        action=AdminAction.HALT_USER,
        target_user_id="user_1",
        performed_by="admin@foo.com",
        reason="reason",
        performed_at=T0,
    )
    with pytest.raises(FrozenInstanceError):
        record.target_user_id = "other"  # type: ignore[misc]


# ------------------------- build_admin_view ----------------------------------


def test_build_admin_view_empty() -> None:
    view = build_admin_view([], now=T0)
    assert view.total_users == 0
    assert view.active_users_24h == 0
    assert view.halt_active_count == 0
    assert view.total_llm_usd_today == 0.0
    assert view.users == ()


def test_build_admin_view_counts_users() -> None:
    users = [
        _make_user(user_id="u1", email="a@x.com"),
        _make_user(user_id="u2", email="b@x.com"),
        _make_user(user_id="u3", email="c@x.com"),
    ]
    view = build_admin_view(users, now=T0)
    assert view.total_users == 3


def test_build_admin_view_active_users_within_window() -> None:
    users = [
        _make_user(user_id="u1", email="a@x.com", last_active_at=T0 - timedelta(hours=1)),
        _make_user(user_id="u2", email="b@x.com", last_active_at=T0 - timedelta(hours=25)),
        _make_user(user_id="u3", email="c@x.com", last_active_at=None),
    ]
    view = build_admin_view(users, now=T0)
    assert view.active_users_24h == 1


def test_build_admin_view_active_window_boundary_inclusive() -> None:
    """Pin: 24h ago exactly is still active (>=, not >)."""

    users = [
        _make_user(user_id="u1", email="a@x.com", last_active_at=T0 - timedelta(hours=24)),
    ]
    view = build_admin_view(users, now=T0)
    assert view.active_users_24h == 1


def test_build_admin_view_halt_count() -> None:
    users = [
        _make_user(user_id="u1", email="a@x.com", halt_active=True),
        _make_user(user_id="u2", email="b@x.com", halt_active=False),
        _make_user(user_id="u3", email="c@x.com", halt_active=True),
    ]
    view = build_admin_view(users, now=T0)
    assert view.halt_active_count == 2


def test_build_admin_view_total_llm_spend() -> None:
    users = [
        _make_user(user_id="u1", email="a@x.com", llm_usd_used_today=1.50),
        _make_user(user_id="u2", email="b@x.com", llm_usd_used_today=2.25),
        _make_user(user_id="u3", email="c@x.com", llm_usd_used_today=0.00),
    ]
    view = build_admin_view(users, now=T0)
    assert view.total_llm_usd_today == pytest.approx(3.75)


def test_build_admin_view_tier_breakdown() -> None:
    users = [
        _make_user(user_id="u1", email="a@x.com", effective_tier=Tier.FREE),
        _make_user(user_id="u2", email="b@x.com", effective_tier=Tier.PRO),
        _make_user(user_id="u3", email="c@x.com", effective_tier=Tier.PRO),
        _make_user(user_id="u4", email="d@x.com", effective_tier=Tier.ENTERPRISE),
    ]
    view = build_admin_view(users, now=T0)
    counts = dict(view.tier_breakdown)
    assert counts[Tier.FREE] == 1
    assert counts[Tier.PRO] == 2
    assert counts[Tier.ENTERPRISE] == 1


def test_build_admin_view_tier_breakdown_includes_zero_tiers() -> None:
    """Pin: every Tier appears in the breakdown even with zero users."""

    users = [_make_user(user_id="u1", email="a@x.com", effective_tier=Tier.PRO)]
    view = build_admin_view(users, now=T0)
    counts = dict(view.tier_breakdown)
    assert Tier.FREE in counts
    assert Tier.PRO in counts
    assert Tier.ENTERPRISE in counts
    assert counts[Tier.FREE] == 0
    assert counts[Tier.ENTERPRISE] == 0


def test_build_admin_view_users_sorted_by_joined_at() -> None:
    """Pin: deterministic ordering."""

    users = [
        _make_user(user_id="u3", email="c@x.com", joined_at=T0 - timedelta(days=1)),
        _make_user(user_id="u1", email="a@x.com", joined_at=T0 - timedelta(days=30)),
        _make_user(user_id="u2", email="b@x.com", joined_at=T0 - timedelta(days=10)),
    ]
    view = build_admin_view(users, now=T0)
    assert [u.user_id for u in view.users] == ["u1", "u2", "u3"]


def test_build_admin_view_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now"):
        build_admin_view([], now=datetime(2026, 5, 1))


def test_build_admin_view_rejects_zero_active_window() -> None:
    with pytest.raises(ValueError, match="active_window"):
        build_admin_view([], now=T0, active_window=timedelta(0))


def test_build_admin_view_rejects_negative_active_window() -> None:
    with pytest.raises(ValueError, match="active_window"):
        build_admin_view([], now=T0, active_window=timedelta(hours=-1))


def test_build_admin_view_custom_active_window() -> None:
    """Pass a 7-day window for weekly-active aggregation."""

    users = [
        _make_user(
            user_id="u1",
            email="a@x.com",
            last_active_at=T0 - timedelta(days=3),
        ),
        _make_user(
            user_id="u2",
            email="b@x.com",
            last_active_at=T0 - timedelta(days=10),
        ),
    ]
    view = build_admin_view(users, now=T0, active_window=timedelta(days=7))
    assert view.active_users_24h == 1


def test_build_admin_view_is_deterministic() -> None:
    users = [
        _make_user(user_id="u1", email="a@x.com"),
        _make_user(user_id="u2", email="b@x.com", halt_active=True),
    ]
    a = build_admin_view(users, now=T0)
    b = build_admin_view(users, now=T0)
    assert a == b


def test_admin_view_is_frozen() -> None:
    view = build_admin_view([], now=T0)
    with pytest.raises(FrozenInstanceError):
        view.total_users = 99  # type: ignore[misc]


# ------------------------ Enum string pins -----------------------------------


def test_admin_action_string_values_pinned() -> None:
    assert AdminAction.HALT_USER.value == "halt_user"
    assert AdminAction.RESUME_USER.value == "resume_user"
    assert AdminAction.OVERRIDE_TIER.value == "override_tier"
    assert AdminAction.REVOKE_SESSION.value == "revoke_session"
    assert AdminAction.SUSPEND_USER.value == "suspend_user"
    assert AdminAction.INSPECT_USER.value == "inspect_user"


# --------------------------- render_user_summary -----------------------------


def test_render_user_summary_includes_user_id_and_email() -> None:
    user = _make_user(user_id="alice", email="alice@example.com")
    out = render_user_summary(user)
    assert "alice" in out
    assert "alice@example.com" in out


def test_render_user_summary_includes_tier_and_status() -> None:
    user = _make_user(effective_tier=Tier.ENTERPRISE)
    out = render_user_summary(user)
    assert "enterprise" in out
    assert "active" in out


def test_render_user_summary_includes_llm_spend() -> None:
    user = _make_user(llm_usd_used_today=2.50, llm_usd_limit_today=10.0)
    out = render_user_summary(user)
    assert "$2.50" in out
    assert "$10.00" in out
    assert "25%" in out


def test_render_user_summary_shows_halt_marker_when_halted() -> None:
    user = _make_user(halt_active=True)
    out = render_user_summary(user)
    assert "HALTED" in out


def test_render_user_summary_omits_halt_marker_when_not_halted() -> None:
    user = _make_user(halt_active=False)
    out = render_user_summary(user)
    assert "HALTED" not in out


def test_render_user_summary_shows_never_for_no_last_active() -> None:
    user = _make_user(last_active_at=None)
    out = render_user_summary(user)
    assert "never" in out


def test_render_user_summary_no_secret_leak() -> None:
    """Pin: render never includes session tokens / API keys / hashes."""

    user = _make_user()
    out = render_user_summary(user)
    # No Stripe IDs
    assert "cus_" not in out.lower()
    assert "sub_" not in out.lower()
    # No password / hash markers
    assert "password" not in out.lower()
    assert "hash" not in out.lower()
    assert "salt" not in out.lower()
    # No session-token markers
    assert "session_" not in out.lower()
    assert "bearer" not in out.lower()


# --------------------------- render_admin_view -------------------------------


def test_render_admin_view_includes_totals() -> None:
    users = [
        _make_user(user_id="u1", email="a@x.com", halt_active=True),
        _make_user(user_id="u2", email="b@x.com"),
    ]
    view = build_admin_view(users, now=T0)
    out = render_admin_view(view)
    assert "total: 2" in out
    assert "halted: 1" in out


def test_render_admin_view_includes_tier_breakdown() -> None:
    users = [
        _make_user(user_id="u1", email="a@x.com", effective_tier=Tier.FREE),
        _make_user(user_id="u2", email="b@x.com", effective_tier=Tier.PRO),
    ]
    view = build_admin_view(users, now=T0)
    out = render_admin_view(view)
    assert "free=1" in out
    assert "pro=1" in out


def test_render_admin_view_includes_user_rows() -> None:
    users = [_make_user(user_id="alice", email="alice@x.com")]
    view = build_admin_view(users, now=T0)
    out = render_admin_view(view)
    assert "alice" in out
    assert "alice@x.com" in out


def test_render_admin_view_handles_empty_user_list() -> None:
    view = build_admin_view([], now=T0)
    out = render_admin_view(view)
    assert "total: 0" in out


def test_render_admin_view_no_secret_leak() -> None:
    users = [_make_user(user_id="u1", email="a@x.com")]
    view = build_admin_view(users, now=T0)
    out = render_admin_view(view)
    assert "cus_" not in out.lower()
    assert "sub_" not in out.lower()
    assert "password" not in out.lower()
    assert "session_" not in out.lower()
    assert "bearer" not in out.lower()


# --------------------------- render_action_record ----------------------------


def test_render_action_record_includes_action_and_target() -> None:
    admins = AdminEmailList(emails=frozenset({"admin@foo.com"}))
    req = _make_request(action=AdminAction.HALT_USER, reason="suspicious")
    record = audit_admin_action(req, admin_emails=admins)
    out = render_action_record(record)
    assert "halt_user" in out
    assert "user_1" in out


def test_render_action_record_includes_performed_by_and_reason() -> None:
    admins = AdminEmailList(emails=frozenset({"admin@foo.com"}))
    req = _make_request(action=AdminAction.HALT_USER, reason="excessive losses")
    record = audit_admin_action(req, admin_emails=admins)
    out = render_action_record(record)
    assert "admin@foo.com" in out
    assert "excessive losses" in out


def test_render_action_record_includes_target_tier_when_set() -> None:
    admins = AdminEmailList(emails=frozenset({"admin@foo.com"}))
    req = _make_request(
        action=AdminAction.OVERRIDE_TIER,
        reason="beta upgrade",
        target_tier=Tier.ENTERPRISE,
    )
    record = audit_admin_action(req, admin_emails=admins)
    out = render_action_record(record)
    assert "enterprise" in out


def test_render_action_record_omits_target_tier_when_unset() -> None:
    admins = AdminEmailList(emails=frozenset({"admin@foo.com"}))
    req = _make_request(action=AdminAction.RESUME_USER, reason="")
    record = audit_admin_action(req, admin_emails=admins)
    out = render_action_record(record)
    assert "target tier" not in out


def test_render_action_record_omits_reason_when_empty() -> None:
    """Resume + inspect actions don't require reason; render skips empty."""

    admins = AdminEmailList(emails=frozenset({"admin@foo.com"}))
    req = _make_request(action=AdminAction.RESUME_USER, reason="")
    record = audit_admin_action(req, admin_emails=admins)
    out = render_action_record(record)
    assert "reason:" not in out


# ----------------------------- e2e flows -------------------------------------


def test_e2e_admin_halts_then_resumes() -> None:
    admins = AdminEmailList(emails=frozenset({"admin@foo.com"}))

    halt_req = AdminActionRequest(
        action_id="act_halt",
        action=AdminAction.HALT_USER,
        target_user_id="user_99",
        performed_by="admin@foo.com",
        reason="user reported suspicious activity",
        performed_at=T0,
    )
    halt_record = audit_admin_action(halt_req, admin_emails=admins)
    assert halt_record.action is AdminAction.HALT_USER

    resume_req = AdminActionRequest(
        action_id="act_resume",
        action=AdminAction.RESUME_USER,
        target_user_id="user_99",
        performed_by="admin@foo.com",
        reason="",
        performed_at=T0 + timedelta(hours=2),
    )
    resume_record = audit_admin_action(resume_req, admin_emails=admins)
    assert resume_record.action is AdminAction.RESUME_USER


def test_e2e_non_admin_attempt_blocked_at_audit_boundary() -> None:
    """Pin: a non-admin caller never gets past audit_admin_action."""

    admins = AdminEmailList(emails=frozenset({"admin@foo.com"}))
    bad_req = AdminActionRequest(
        action_id="evil",
        action=AdminAction.HALT_USER,
        target_user_id="victim",
        performed_by="attacker@evil.com",
        reason="i'm not actually an admin",
        performed_at=T0,
    )
    with pytest.raises(AdminAuthorizationError):
        audit_admin_action(bad_req, admin_emails=admins)


def test_e2e_full_admin_view_render() -> None:
    users = [
        _make_user(
            user_id="u_pro",
            email="pro@x.com",
            effective_tier=Tier.PRO,
            llm_usd_used_today=5.50,
            llm_usd_limit_today=10.0,
        ),
        _make_user(
            user_id="u_free",
            email="free@x.com",
            effective_tier=Tier.FREE,
            subscription_status=SubscriptionStatus.EXPIRED,
            llm_usd_used_today=0.10,
            llm_usd_limit_today=0.50,
            halt_active=True,
        ),
        _make_user(
            user_id="u_ent",
            email="ent@x.com",
            effective_tier=Tier.ENTERPRISE,
            llm_usd_used_today=12.30,
            llm_usd_limit_today=100.0,
            joined_at=T0 - timedelta(days=60),
        ),
    ]
    view = build_admin_view(users, now=T0)
    out = render_admin_view(view)
    # Totals + per-user
    assert "total: 3" in out
    assert "halted: 1" in out
    assert "u_pro" in out
    assert "u_free" in out
    assert "u_ent" in out
    # Tier breakdown
    assert "free=1" in out
    assert "pro=1" in out
    assert "enterprise=1" in out
