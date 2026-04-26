"""Route registry — each module exports register(app, app_state) -> None."""

from __future__ import annotations

from typing import Any

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
    insights,
    metrics,
    mobile,
    pnl,
    positions,
    prometheus,
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
)


def register_all(app: FastAPI, app_state: dict[str, Any]) -> None:
    """Register every route module with the FastAPI app."""
    for mod in _MODULES:
        mod.register(app, app_state)
