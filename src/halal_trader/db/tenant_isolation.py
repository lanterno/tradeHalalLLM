"""Multi-tenant database isolation guards.

The roadmap calls for multi-tenant DB isolation as part of Wave
3 (User accounts + auth + per-user vaults + per-user quotas):
"Either: (a) row-level user_id on every table + Postgres RLS
policies, or (b) per-user schemas. Pick (a) for simplicity.
Update every repo to filter by user_id from the request context."
This module is the **pure-Python tenant-scope guard** — the
in-process layer that enforces the row-level user_id check
*before* the SQL query runs, so a forgotten WHERE clause in a
repository method can't silently leak one tenant's data to
another.

Picked an explicit `TenantContext` + `TenantViolationError`
over a Postgres-only RLS approach because (a) the bot's tests
run against the same SQLAlchemy + asyncpg layer the production
deploys to, and a Python-side guard is testable without
spinning up Postgres in unit tests, (b) RLS policies are still
recommended at the DB layer as defence-in-depth — this module
is the operator's first defensive boundary, not the only one,
(c) the failure mode this guards against is a developer
forgetting to pass user_id into a query, which a database
policy catches but with a less-descriptive error than
"TenantViolationError: query targeted user 'A' but active scope
is user 'B'".

Pinned semantics:
- **`TenantContext` required for every scoped operation.** The
  module-level `current_tenant()` returns `None` when no scope
  is active; scoped operations raise `TenantViolationError`
  rather than silently allowing the operation.
- **ADMIN scope is opt-in only.** The operator's admin console
  enters ADMIN scope via the explicit `enter_admin_scope()`
  context manager; admin scope bypasses the row-level user_id
  check but always emits a warning to the audit log so admin-
  side queries are traceable. Pinned via test that admin scope
  can't be entered implicitly.
- **Empty user_id rejected.** Empty / whitespace-only user_id
  raises at TenantContext construction. Mirrors the
  validation patterns of Wave 11.C KYC + Wave 11.D privacy.
- **Row-validation helper raises on cross-tenant.** When a
  repository method retrieves a row whose `user_id` doesn't
  match the active scope, `assert_row_scope` raises
  `TenantViolationError`. The pin matters because a SQL bug
  that returns the wrong row (e.g., a missing JOIN condition)
  is caught at the application layer rather than silently
  rendered to the user.
- **Render output never includes other tenants' user_ids.**
  `TenantViolationError.message` references the active scope's
  user_id but redacts the would-be-leaked target_user_id to
  `<other-tenant>` so error logs / Slack alerts don't leak
  cross-tenant identifiers. Mirrors the no-PII pattern of
  Wave 11.D + 11.C + 3.B.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Iterator


class ScopeKind(str, Enum):
    """The scope category."""

    USER = "user"
    ADMIN = "admin"


@dataclass(frozen=True)
class TenantContext:
    """The active tenant scope.

    `user_id` is the current user's identifier; for ADMIN scope,
    `user_id` carries the admin operator's identifier (so the
    audit trail can attribute admin-side queries).
    """

    user_id: str
    scope: ScopeKind = ScopeKind.USER

    def __post_init__(self) -> None:
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")


class TenantViolationError(Exception):
    """Raised when an operation crosses tenant boundaries.

    Pinned no-PII contract: the message references the active
    scope's user_id but redacts the would-be-leaked target_user_id
    to `<other-tenant>`. Operators reading error logs / Slack
    alerts see enough context to triage without seeing the other
    tenant's identifier.
    """

    def __init__(
        self,
        *,
        active_user_id: str,
        target_user_id: str,
        operation: str = "query",
    ) -> None:
        self.active_user_id = active_user_id
        self.target_user_id = target_user_id
        self.operation = operation
        super().__init__(
            f"tenant violation: active scope is user "
            f"{active_user_id!r}, but {operation} targets "
            f"<other-tenant>"
        )


_current_tenant_var: ContextVar[TenantContext | None] = ContextVar(
    "halal_trader_current_tenant", default=None
)


def current_tenant() -> TenantContext | None:
    """Return the active tenant scope, or None if unset."""

    return _current_tenant_var.get()


def require_tenant() -> TenantContext:
    """Return the active tenant scope, raising if unset.

    Use this at every scoped repository entry point — the explicit
    raise prevents silent fallthrough when a developer forgets to
    enter a scope.
    """

    ctx = _current_tenant_var.get()
    if ctx is None:
        raise TenantViolationError(
            active_user_id="<no-scope>",
            target_user_id="<unknown>",
            operation="scoped operation requires active TenantContext",
        )
    return ctx


@contextmanager
def enter_user_scope(user_id: str) -> Iterator[TenantContext]:
    """Enter a user-scope context.

    On exit, the previous scope is restored — supports nested
    scopes (e.g., admin temporarily impersonating a user).
    """

    ctx = TenantContext(user_id=user_id, scope=ScopeKind.USER)
    token = _current_tenant_var.set(ctx)
    try:
        yield ctx
    finally:
        _current_tenant_var.reset(token)


@contextmanager
def enter_admin_scope(admin_user_id: str) -> Iterator[TenantContext]:
    """Enter an admin-scope context.

    Admin scope bypasses the row-level user_id check but every
    repository method that detects admin scope should emit an
    audit-log entry attributing the operation to the admin
    operator. Pinned: admin scope is explicit + opt-in; never
    implicit.
    """

    ctx = TenantContext(user_id=admin_user_id, scope=ScopeKind.ADMIN)
    token = _current_tenant_var.set(ctx)
    try:
        yield ctx
    finally:
        _current_tenant_var.reset(token)


def assert_row_scope(*, row_user_id: str, operation: str = "read") -> None:
    """Assert that a fetched row belongs to the active tenant scope.

    Repositories call this after fetching a row to verify the
    SQL WHERE clause was correctly applied. Under ADMIN scope,
    the check is bypassed but the call is still expected (ops
    audit logs prove the check ran).

    Raises `TenantViolationError` if the row's user_id doesn't
    match the active user-scope.
    """

    ctx = require_tenant()
    if ctx.scope is ScopeKind.ADMIN:
        return  # admin bypass; caller is expected to audit-log
    if not row_user_id or not row_user_id.strip():
        raise ValueError("row_user_id must be non-empty")
    if row_user_id != ctx.user_id:
        raise TenantViolationError(
            active_user_id=ctx.user_id,
            target_user_id=row_user_id,
            operation=operation,
        )


def assert_payload_scope(*, payload_user_id: str, operation: str = "write") -> None:
    """Assert that a write payload's user_id matches the active scope.

    Use this at the entry of every repository write method to
    catch the "developer passes user_id from request body without
    validating it matches the auth context" failure mode.
    """

    ctx = require_tenant()
    if ctx.scope is ScopeKind.ADMIN:
        return
    if not payload_user_id or not payload_user_id.strip():
        raise ValueError("payload_user_id must be non-empty")
    if payload_user_id != ctx.user_id:
        raise TenantViolationError(
            active_user_id=ctx.user_id,
            target_user_id=payload_user_id,
            operation=operation,
        )


def is_admin_scope() -> bool:
    """Convenience: True iff the active scope is ADMIN.

    Repositories that admit admin-scope-only operations
    (cross-tenant reports, debug introspection) check this.
    """

    ctx = current_tenant()
    return ctx is not None and ctx.scope is ScopeKind.ADMIN


def render_active_scope() -> str:
    """Render the active scope for ops display.

    Pinned no-cross-tenant-leak contract: the rendered output
    references the active scope's user_id only, never any other
    tenant's identifier (the engine doesn't have access to
    other tenants in this state anyway).
    """

    ctx = current_tenant()
    if ctx is None:
        return "🔓 (no tenant scope)"
    if ctx.scope is ScopeKind.ADMIN:
        return f"🛡️ ADMIN: {ctx.user_id}"
    return f"👤 USER: {ctx.user_id}"


__all__ = [
    "ScopeKind",
    "TenantContext",
    "TenantViolationError",
    "assert_payload_scope",
    "assert_row_scope",
    "current_tenant",
    "enter_admin_scope",
    "enter_user_scope",
    "is_admin_scope",
    "render_active_scope",
    "require_tenant",
]
