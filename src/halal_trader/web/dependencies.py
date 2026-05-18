"""FastAPI dependency-injection seam for the typed dashboard context.

Every route takes a :class:`DashboardContext` via ``Depends(get_ctx)``;
the context is attached to the FastAPI app at startup (lifespan) and
pulled out on each request via ``request.app.state.ctx``. The legacy
global-dict shim was removed in Wave A — there is no
app_state dict anywhere in the route layer.
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
    ctx: DashboardContext | None = getattr(request.app.state, "ctx", None)
    if ctx is None:
        raise RuntimeError("DashboardContext not attached to FastAPI app — did the lifespan run?")
    return ctx
