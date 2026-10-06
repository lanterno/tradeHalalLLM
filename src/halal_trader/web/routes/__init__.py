"""Route registry — each module exports ``register(app)``.

Routes pull dependencies via ``Depends(get_ctx)`` from
``web.dependencies``; the registry just walks the module list.
"""

from __future__ import annotations

from fastapi import FastAPI

from halal_trader.web.routes import (
    activity,
    admin_halal,
    analytics,
    config,
    core,
    decisions,
    halabot_beliefs,
    halal_compliance,
    halal_zakat,
    home,
    insights,
    metrics,
    mobile,
    pnl,
    positions,
    prometheus,
    recommendation,
    research,
    risk,
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
    config,
    system,
    risk,
    metrics,
    streaming,
    research,
    prometheus,
    activity,
    admin_halal,
    mobile,
    insights,
    halal_compliance,
    halal_zakat,
    core,
    home,
    recommendation,
    halabot_beliefs,
)


def register_all(app: FastAPI) -> None:
    """Register every route module with the FastAPI app."""
    for mod in _MODULES:
        mod.register(app)
