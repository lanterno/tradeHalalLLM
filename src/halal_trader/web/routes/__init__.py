"""Route registry — each module exports ``register(app)``.

Routes pull dependencies via ``Depends(get_ctx)`` from
``web.dependencies``; the registry just walks the module list.
"""

from __future__ import annotations

from fastapi import FastAPI

from halal_trader.web.routes import (
    activity,
    admin,
    admin_config,
    admin_halal,
    admin_trades,
    analytics,
    config,
    decisions,
    halal_explain,
    insights,
    metrics,
    mobile,
    pnl,
    positions,
    prometheus,
    prompts,
    research,
    research_jobs,
    risk,
    sentiment,
    streaming,
    system,
    trades,
)

_MODULES = (
    trades,
    pnl,
    analytics,
    positions,
    decisions,
    sentiment,
    config,
    system,
    risk,
    metrics,
    streaming,
    research,
    research_jobs,
    prometheus,
    activity,
    admin,
    admin_config,
    admin_trades,
    admin_halal,
    mobile,
    insights,
    prompts,
    halal_explain,
)


def register_all(app: FastAPI) -> None:
    """Register every route module with the FastAPI app."""
    for mod in _MODULES:
        mod.register(app)
