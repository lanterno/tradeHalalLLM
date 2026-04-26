"""Mutation audit writer — wraps every state-changing request in a DB row.

The auth middleware (``web/middleware/auth.py``) decides *whether* a
request is allowed; this audit middleware writes *that it happened*.
Both run on every mutation; the audit row is created BEFORE the
handler executes (so a handler that crashes still leaves a "pending"
trace) and updated to ``ok`` / ``error`` when the handler returns.

The audit row's ``actor`` is the request_id ContextVar value so a
trace can be correlated against the JSON log file by request_id. We
deliberately do NOT capture the API token in the row — secrets stay
in headers, not in the audit table.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from fastapi import Request
from fastapi.responses import Response

from halal_trader.core.observability import request_id_var

logger = logging.getLogger(__name__)


# Methods + path prefixes that get an audit row written.
_AUDIT_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_PAYLOAD_TRUNCATE_BYTES = 4_000  # cap so a huge body can't blow up the table


def _is_audit_request(request: Request) -> bool:
    """Mirror the auth middleware's decision so the two stay in lockstep."""
    if request.method.upper() not in _AUDIT_METHODS:
        return False
    return request.url.path.startswith("/api/")


async def audit_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Wrap a mutation in an open/close pair of audit-table writes."""
    if not _is_audit_request(request):
        return await call_next(request)

    # Pull the repo from app_state — same shape every other route module
    # uses. If the app didn't initialise the engine yet (cold-start race)
    # we just log and pass through; missing audit is preferable to
    # blocking the request.
    app_state = getattr(request.app.state, "ext_state", None) or _get_app_state(request)
    repo = (app_state or {}).get("repo")

    body_bytes = b""
    try:
        body_bytes = await request.body()
    except Exception:
        body_bytes = b""

    payload = _truncate_payload(body_bytes)
    actor = request_id_var.get() or "anon"
    action_id: int | None = None

    if repo is not None:
        try:
            action_id = await repo.begin_web_action(
                actor=actor,
                method=request.method.upper(),
                path=request.url.path,
                payload=payload,
            )
        except Exception as e:
            logger.debug("audit begin_web_action failed: %s", e)

    # Re-inject the body so downstream handlers can still read it.
    async def _receive():
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    request._receive = _receive  # type: ignore[attr-defined]

    error: str | None = None
    status: int = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception as e:  # noqa: BLE001 — recorded then re-raised
        error = repr(e)
        raise
    finally:
        if repo is not None and action_id is not None:
            try:
                await repo.complete_web_action(action_id, status_code=status, error=error)
            except Exception as e:
                logger.debug("audit complete_web_action failed: %s", e)


def _get_app_state(request: Request) -> dict | None:
    """Reach into the module-level app_state used by web/app.py.

    We can't store it on ``request.app.state`` without changing the
    bootstrap; instead, every existing route looks up the module-level
    dict, so we do the same.
    """
    try:
        from halal_trader.web.app import app_state

        return app_state
    except Exception:
        return None


def _truncate_payload(body: bytes) -> str | None:
    if not body:
        return None
    if len(body) > _PAYLOAD_TRUNCATE_BYTES:
        body = body[:_PAYLOAD_TRUNCATE_BYTES]
        return body.decode("utf-8", errors="replace") + "…[truncated]"
    return body.decode("utf-8", errors="replace")
