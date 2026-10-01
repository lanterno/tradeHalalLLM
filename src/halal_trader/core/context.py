"""Typed context that replaces the ``app_state: dict[str, Any]`` bag.

:class:`DashboardContext` is what the FastAPI app needs (engine, repos,
settings) plus a small mutable :class:`RuntimeView`. Routes take it via
DI; nothing reaches into a global dict.

The web runs in its own process, so nothing in the bot writes this
``RuntimeView``: only ``started_at`` is ever set. The bot-side
``BotContext`` / ``attach_to_app`` co-host path that would have filled
it was never wired and was deleted on 2026-10-01.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from halal_trader.config import Settings
    from halal_trader.core.event_bus import EventBus
    from halal_trader.db.repository import Repository
    from halal_trader.portfolio.analytics import PerformanceAnalytics


@dataclass
class RuntimeView:
    """The few fields the cycle / monitor pushes during a live run.

    Mutable on purpose — these are what the dashboard polls /
    streams to show "what is the bot doing right now". Each field is
    optional because the dashboard can run without a live bot in the
    same process.
    """

    bot_running: bool = False
    started_at: datetime | None = None
    last_cycle: dict[str, Any] | None = None
    risk_state: dict[str, Any] | None = None
    account_snapshot: dict[str, Any] | None = None
    stock_equity: float | None = None
    stock_positions: list[dict[str, Any]] = field(default_factory=list)
    open_positions_by_asset: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    llm_cost_today_usd: float | None = None
    # Stocks news-momentum reactor — populated by the trading scheduler
    # when the bot is co-hosted with the web app, so /api/system/status
    # can surface classifier health (provider rotation, quota state,
    # daily call volume) without grepping JSON logs.
    stocks_news_reactor: Any = None
    # Broker handle for the operator-intervention endpoints
    # (force-close, cancel-orders). Only populated when the bot is
    # co-hosted with the web app.
    stock_broker: Any = None


@dataclass(frozen=True, slots=True)
class DashboardContext:
    """Read-only deps + a mutable :class:`RuntimeView` window.

    Routes take this via FastAPI ``Depends`` (see ``web/dependencies.py``)
    instead of reaching into ``app_state``. The frozen fields are the
    long-lived deps; ``runtime`` is the only place that mutates.
    """

    engine: "AsyncEngine"
    repo: "Repository"
    analytics: "PerformanceAnalytics"
    settings: "Settings"
    bus: "EventBus"
    runtime: RuntimeView
