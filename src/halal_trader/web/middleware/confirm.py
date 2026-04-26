"""Typed-confirmation requirement for destructive admin endpoints.

The auth gate (``middleware/auth.py``) keeps anonymous callers out;
this dependency makes sure that even an authenticated operator can't
fat-finger an irreversible action. Each destructive endpoint declares
``Depends(require_confirmation)``; FastAPI then refuses any request
that doesn't carry an ``X-Trader-Confirm: true`` header.

The dashboard's React side renders a modal that asks the operator to
type a known string (e.g. the symbol being closed) and only then sends
the confirmation header. The header is the single boundary the server
checks — it doesn't enforce any particular client-side typing UI.

Tests can disable the requirement by setting ``WEB_REQUIRE_CONFIRMATION
=false`` so the runner doesn't have to forge headers in every call.
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from halal_trader.config import get_settings


def require_confirmation(request: Request) -> None:
    """FastAPI dependency: 412 if the confirm header isn't present.

    Returns ``None`` on success — used purely for its side effect.
    """
    if not get_settings().web.require_confirmation:
        return None

    header = request.headers.get("X-Trader-Confirm", "").strip().lower()
    if header != "true":
        raise HTTPException(
            status_code=412,
            detail=(
                "Destructive action requires X-Trader-Confirm: true header. "
                "The dashboard's confirm-modal will set this automatically once "
                "you type the confirmation string."
            ),
        )
    return None
