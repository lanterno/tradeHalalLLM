"""FastAPI dependency-injection seam for the typed dashboard context.

Routes used to read ``app_state["..."]`` directly. They now take a
:class:`DashboardContext` via ``Depends(get_ctx)``; the context is
attached to the FastAPI app at startup and pulled out on each request
via ``request.app.state.ctx``.

This is the only file in ``halal_trader.web`` that touches the global
``app_state`` shim — everything else is strictly typed.
"""

from __future__ import annotations

from fastapi import Request

from halal_trader.core.context import DashboardContext


def get_ctx(request: Request) -> DashboardContext:
    """Resolve the request-scoped dashboard context.

    Raises ``RuntimeError`` if the lifespan didn't install a context
    on the app — a programming error we want to surface loudly rather
    than silently 500.
    """
    ctx = getattr(request.app.state, "ctx", None)
    if ctx is None:
        raise RuntimeError("DashboardContext not attached to FastAPI app — did the lifespan run?")
    return ctx
