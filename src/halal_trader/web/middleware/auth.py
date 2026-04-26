"""Header-based auth gate for dashboard mutation endpoints.

The dashboard's read endpoints stay open behind the localhost binding
the project has always relied on. State-changing endpoints — anything
under ``/api/admin/*`` or any non-GET request — additionally require a
matching ``X-Trader-Token`` header.

Why a single shared secret rather than full RBAC: this is a single-
operator product. A second user is one of the explicit non-goals from
the W6 "deprioritise" list. A shared secret is enough to gate the
attack surface (an exposed dashboard tunnel) without burning weeks on
a roles model nobody asked for.

The gate is permissive on configuration: an empty token in
``WebSettings.api_token`` puts the dashboard in *read-only* mode (every
mutation 503s with ``"mutations disabled — set WEB_API_TOKEN to enable"``).
That's the safe default for fresh deployments and CI.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from halal_trader.config import get_settings

logger = logging.getLogger(__name__)


# Methods + path prefixes that are considered "mutations" and require auth.
_MUTATION_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_ADMIN_PREFIX = "/api/admin"


def _is_mutation_request(request: Request) -> bool:
    """Treat any non-GET hit on /api/admin/* as a mutation, plus any non-GET API call.

    ``/api/admin`` is the dedicated mutation namespace, but we also gate
    *any* non-idempotent verb across the whole API. That way a future
    PATCH on a non-admin route can't accidentally bypass the gate.
    """
    method = request.method.upper()
    path = request.url.path
    if path.startswith(_ADMIN_PREFIX):
        return True
    if method in _MUTATION_METHODS and path.startswith("/api/"):
        return True
    return False


async def auth_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Reject unauthenticated mutations with 401/503.

    * No token configured → 503, mutations disabled
    * Token configured + no header → 401
    * Token configured + wrong header → 401
    * Token matches → continue
    """
    if not _is_mutation_request(request):
        return await call_next(request)

    settings = get_settings()
    expected = settings.web.api_token

    if not expected:
        logger.warning(
            "mutation rejected: WEB_API_TOKEN unset — dashboard is in read-only mode "
            "(method=%s path=%s)",
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": "mutations_disabled",
                "detail": (
                    "Dashboard mutations are disabled. Set WEB_API_TOKEN in the "
                    "environment to enable the control surface."
                ),
            },
        )

    presented = request.headers.get("X-Trader-Token", "")
    if not _constant_time_eq(presented, expected):
        logger.warning(
            "mutation rejected: bad/missing X-Trader-Token (method=%s path=%s)",
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=401,
            content={"error": "unauthorized", "detail": "X-Trader-Token missing or invalid"},
        )

    return await call_next(request)


def _constant_time_eq(a: str, b: str) -> bool:
    """Constant-time comparison to avoid leaking the secret via response timing.

    ``hmac.compare_digest`` is the stdlib answer; we route through it so
    a timing attacker on the loopback can't string-prefix-match the
    token byte by byte.
    """
    import hmac

    if len(a) != len(b):
        # ``compare_digest`` already runs constant-time on a length
        # mismatch, but spelling it out keeps the intent obvious.
        return False
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
