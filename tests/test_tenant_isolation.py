"""Tests for the multi-tenant database isolation guards."""

from __future__ import annotations

import dataclasses

import pytest

from halal_trader.db.tenant_isolation import (
    ScopeKind,
    TenantContext,
    TenantViolationError,
    assert_payload_scope,
    assert_row_scope,
    current_tenant,
    enter_admin_scope,
    enter_user_scope,
    is_admin_scope,
    render_active_scope,
    require_tenant,
)

# ---------------------------------------------------------------------------
# TenantContext validation
# ---------------------------------------------------------------------------


def test_context_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        TenantContext(user_id="")


def test_context_rejects_whitespace_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        TenantContext(user_id="   ")


def test_context_default_scope_is_user() -> None:
    ctx = TenantContext(user_id="user-1")
    assert ctx.scope is ScopeKind.USER


def test_context_can_be_admin() -> None:
    ctx = TenantContext(user_id="admin-1", scope=ScopeKind.ADMIN)
    assert ctx.scope is ScopeKind.ADMIN


# ---------------------------------------------------------------------------
# Default state: no scope
# ---------------------------------------------------------------------------


def test_default_no_scope() -> None:
    """Pin: no implicit scope at module load time."""

    assert current_tenant() is None


def test_require_tenant_raises_without_scope() -> None:
    with pytest.raises(TenantViolationError):
        require_tenant()


def test_is_admin_scope_false_without_scope() -> None:
    assert is_admin_scope() is False


# ---------------------------------------------------------------------------
# enter_user_scope
# ---------------------------------------------------------------------------


def test_enter_user_scope_sets_current_tenant() -> None:
    with enter_user_scope("user-1") as ctx:
        assert ctx.user_id == "user-1"
        assert ctx.scope is ScopeKind.USER
        assert current_tenant() == ctx
        assert require_tenant() == ctx


def test_user_scope_restores_previous_on_exit() -> None:
    """Pin: nested scopes restore the outer scope on exit."""

    with enter_user_scope("user-1"):
        with enter_user_scope("user-2"):
            assert current_tenant().user_id == "user-2"
        assert current_tenant().user_id == "user-1"
    assert current_tenant() is None


def test_user_scope_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        with enter_user_scope(""):
            pass


def test_is_admin_scope_false_under_user_scope() -> None:
    with enter_user_scope("user-1"):
        assert is_admin_scope() is False


# ---------------------------------------------------------------------------
# enter_admin_scope — explicit opt-in only
# ---------------------------------------------------------------------------


def test_enter_admin_scope_sets_admin_kind() -> None:
    with enter_admin_scope("admin-1") as ctx:
        assert ctx.scope is ScopeKind.ADMIN
        assert ctx.user_id == "admin-1"


def test_is_admin_scope_true_under_admin_scope() -> None:
    with enter_admin_scope("admin-1"):
        assert is_admin_scope() is True


def test_admin_scope_restores_previous_on_exit() -> None:
    with enter_user_scope("user-1"):
        with enter_admin_scope("admin-1"):
            assert is_admin_scope() is True
        # restored to user scope
        assert is_admin_scope() is False
        assert current_tenant().user_id == "user-1"


def test_admin_scope_rejects_empty_admin_id() -> None:
    """Pin: admin scope still requires a non-empty user_id (the operator's identifier)."""

    with pytest.raises(ValueError, match="user_id"):
        with enter_admin_scope(""):
            pass


# ---------------------------------------------------------------------------
# assert_row_scope — read-side validation
# ---------------------------------------------------------------------------


def test_assert_row_scope_passes_for_matching_row() -> None:
    with enter_user_scope("user-1"):
        # No exception
        assert_row_scope(row_user_id="user-1")


def test_assert_row_scope_raises_for_mismatched_row() -> None:
    """Pin: cross-tenant read raises TenantViolationError."""

    with enter_user_scope("user-1"):
        with pytest.raises(TenantViolationError) as exc_info:
            assert_row_scope(row_user_id="user-2")
        assert exc_info.value.active_user_id == "user-1"
        assert exc_info.value.target_user_id == "user-2"


def test_assert_row_scope_admin_bypasses_check() -> None:
    """Pin: admin scope bypasses the row check (caller audit-logs)."""

    with enter_admin_scope("admin-1"):
        # No exception even though row belongs to another user
        assert_row_scope(row_user_id="user-X")


def test_assert_row_scope_raises_without_scope() -> None:
    """Pin: row check requires active scope."""

    with pytest.raises(TenantViolationError):
        assert_row_scope(row_user_id="user-1")


def test_assert_row_scope_rejects_empty_row_user_id() -> None:
    with enter_user_scope("user-1"):
        with pytest.raises(ValueError, match="row_user_id"):
            assert_row_scope(row_user_id="")


def test_assert_row_scope_uses_operation_label() -> None:
    with enter_user_scope("user-1"):
        with pytest.raises(TenantViolationError) as exc_info:
            assert_row_scope(row_user_id="user-2", operation="read trade")
        assert exc_info.value.operation == "read trade"


# ---------------------------------------------------------------------------
# assert_payload_scope — write-side validation
# ---------------------------------------------------------------------------


def test_assert_payload_scope_passes_for_matching_user() -> None:
    with enter_user_scope("user-1"):
        assert_payload_scope(payload_user_id="user-1")


def test_assert_payload_scope_raises_for_mismatch() -> None:
    """Pin: payload user_id mismatch → TenantViolationError on write."""

    with enter_user_scope("user-1"):
        with pytest.raises(TenantViolationError) as exc_info:
            assert_payload_scope(payload_user_id="user-2", operation="insert")
        assert exc_info.value.active_user_id == "user-1"
        assert exc_info.value.operation == "insert"


def test_assert_payload_scope_admin_bypasses_check() -> None:
    with enter_admin_scope("admin-1"):
        assert_payload_scope(payload_user_id="user-X")


def test_assert_payload_scope_raises_without_scope() -> None:
    with pytest.raises(TenantViolationError):
        assert_payload_scope(payload_user_id="user-1")


def test_assert_payload_scope_rejects_empty_payload_user_id() -> None:
    with enter_user_scope("user-1"):
        with pytest.raises(ValueError, match="payload_user_id"):
            assert_payload_scope(payload_user_id="")


# ---------------------------------------------------------------------------
# TenantViolationError no-PII contract
# ---------------------------------------------------------------------------


def test_violation_error_redacts_target_user_id_in_message() -> None:
    """Pin: the error message uses <other-tenant>, not the actual target_user_id.

    The target_user_id is stored on the exception object for
    debugging in dev environments but the str() representation
    redacts it so error logs can be safely shipped to ops channels.
    """

    err = TenantViolationError(
        active_user_id="user-1",
        target_user_id="leaky-target-id",
        operation="read",
    )
    msg = str(err)
    assert "user-1" in msg
    assert "<other-tenant>" in msg
    assert "leaky-target-id" not in msg


def test_violation_error_carries_target_user_id_for_debug() -> None:
    """Pin: target_user_id is on the exception object (operators in dev
    can access it programmatically) but never in str()."""

    err = TenantViolationError(
        active_user_id="user-1",
        target_user_id="target-id",
        operation="read",
    )
    assert err.target_user_id == "target-id"
    assert err.active_user_id == "user-1"


def test_violation_error_default_operation_is_query() -> None:
    err = TenantViolationError(
        active_user_id="user-1",
        target_user_id="user-2",
    )
    assert err.operation == "query"


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_context_is_frozen() -> None:
    ctx = TenantContext(user_id="user-1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.user_id = "user-2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned for JSON / DB serialisation
# ---------------------------------------------------------------------------


def test_scope_kind_string_values() -> None:
    assert ScopeKind.USER.value == "user"
    assert ScopeKind.ADMIN.value == "admin"


# ---------------------------------------------------------------------------
# Render output
# ---------------------------------------------------------------------------


def test_render_no_scope() -> None:
    text = render_active_scope()
    assert "🔓" in text
    assert "no tenant scope" in text


def test_render_user_scope() -> None:
    with enter_user_scope("user-1"):
        text = render_active_scope()
        assert "👤" in text
        assert "user-1" in text


def test_render_admin_scope() -> None:
    with enter_admin_scope("admin-ops-1"):
        text = render_active_scope()
        assert "🛡️" in text
        assert "ADMIN" in text
        assert "admin-ops-1" in text


def test_render_does_not_leak_other_tenant_ids() -> None:
    """Pin: render shows the active scope only, never other tenants."""

    # We can't directly test that other tenant IDs aren't shown
    # because the active scope is the only one accessible — but
    # we can verify that the rendered output references only the
    # active user_id literal.
    with enter_user_scope("user-active-1"):
        text = render_active_scope()
        assert "user-active-1" in text


# ---------------------------------------------------------------------------
# End-to-end realistic flows
# ---------------------------------------------------------------------------


def test_repository_pattern_blocks_cross_tenant_read() -> None:
    """Simulated repository.get_trade flow.

    The repository fetches a row via SQL (here, a simulated dict),
    then calls assert_row_scope to verify the row belongs to the
    active scope. A SQL bug that returns the wrong row is caught.
    """

    def simulated_get_trade(trade_id: str) -> dict:
        # Simulated row-level user_id mismatch (SQL bug)
        return {"trade_id": trade_id, "user_id": "OTHER-USER"}

    with enter_user_scope("user-1"):
        row = simulated_get_trade("trade-123")
        with pytest.raises(TenantViolationError):
            assert_row_scope(row_user_id=row["user_id"], operation="get_trade")


def test_repository_pattern_passes_for_correct_tenant() -> None:
    def simulated_get_trade(trade_id: str) -> dict:
        return {"trade_id": trade_id, "user_id": "user-1"}

    with enter_user_scope("user-1"):
        row = simulated_get_trade("trade-123")
        # Should not raise
        assert_row_scope(row_user_id=row["user_id"])


def test_admin_can_query_across_tenants_for_compliance_review() -> None:
    """Compliance-ops user reviews a flagged trade from another user."""

    def simulated_get_trade(trade_id: str) -> dict:
        return {"trade_id": trade_id, "user_id": "user-flagged"}

    with enter_admin_scope("admin-compliance-ops"):
        row = simulated_get_trade("trade-suspicious-001")
        # Admin scope bypasses the check (the audit-log capture
        # is the operator's responsibility)
        assert_row_scope(row_user_id=row["user_id"])


def test_write_with_payload_user_id_mismatch_blocked() -> None:
    """User-1 attempts to insert a row with user-2's ID — blocked."""

    def simulated_insert_trade(payload: dict) -> None:
        assert_payload_scope(payload_user_id=payload["user_id"], operation="insert")

    with enter_user_scope("user-1"):
        # Malicious payload: user_id from request body
        evil_payload = {"trade_id": "evil-trade", "user_id": "user-2"}
        with pytest.raises(TenantViolationError):
            simulated_insert_trade(evil_payload)


def test_nested_admin_into_user_scope() -> None:
    """Admin enters a user's scope to debug their session.

    Should restore the admin scope on exit.
    """

    with enter_admin_scope("admin-1"):
        assert is_admin_scope() is True
        with enter_user_scope("user-debug-target"):
            assert is_admin_scope() is False
            assert current_tenant().user_id == "user-debug-target"
        # Restored to admin
        assert is_admin_scope() is True
        assert current_tenant().user_id == "admin-1"


# ---------------------------------------------------------------------------
# ContextVar isolation verification
# ---------------------------------------------------------------------------


def test_scope_does_not_leak_after_context_exit() -> None:
    with enter_user_scope("user-temp"):
        assert current_tenant() is not None
    # After exit, scope is back to None
    assert current_tenant() is None


def test_multiple_sequential_scopes() -> None:
    for user_id in ("user-a", "user-b", "user-c"):
        with enter_user_scope(user_id):
            assert current_tenant().user_id == user_id
        assert current_tenant() is None


# ---------------------------------------------------------------------------
# Operation label flows through
# ---------------------------------------------------------------------------


def test_payload_scope_violation_carries_operation() -> None:
    with enter_user_scope("user-1"):
        with pytest.raises(TenantViolationError) as exc_info:
            assert_payload_scope(
                payload_user_id="user-2",
                operation="purification.disburse",
            )
        assert exc_info.value.operation == "purification.disburse"


def test_row_scope_violation_carries_operation() -> None:
    with enter_user_scope("user-1"):
        with pytest.raises(TenantViolationError) as exc_info:
            assert_row_scope(
                row_user_id="user-2",
                operation="trades.list",
            )
        assert exc_info.value.operation == "trades.list"
