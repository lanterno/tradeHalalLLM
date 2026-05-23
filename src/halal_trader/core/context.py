"""Typed contexts that replace the ``app_state: dict[str, Any]`` bag.

The bot and the dashboard each carry a single immutable container of
their long-lived dependencies. Routes / cycle / monitor / CLI commands
take it via DI; nothing else reaches into a global dict.

* :class:`DashboardContext` — what the FastAPI app needs (engine,
  repos, hub, plus mutable runtime fields the cycle pushes into).
* :class:`BotContext` — superset for the trading bot itself
  (broker, LLM, settings, …).

Both are frozen dataclasses for the static slice and carry a small
mutable :class:`RuntimeView` for the few fields the cycle has to
update at runtime (last cycle id, latest risk-state summary,
account snapshot, etc).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from halal_trader.config import Settings
    from halal_trader.core.event_bus import EventBus
    from halal_trader.core.insights_hub import InsightsHub
    from halal_trader.crypto.analytics import PerformanceAnalytics
    from halal_trader.db.repository import Repository


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
    # Optional WS / sentiment refs for live-only routes (price stream,
    # sentiment recent feed). The dashboard-only process leaves them None.
    ws_manager: Any = None
    sentiment_manager: Any = None
    # Stocks news-momentum reactor — populated by the trading scheduler
    # when the bot is co-hosted with the web app, so /api/system/status
    # can surface classifier health (provider rotation, quota state,
    # daily call volume) without grepping JSON logs.
    stocks_news_reactor: Any = None
    # Broker handles for the operator-intervention endpoints
    # (force-close, cancel-orders). Only populated when the bot is
    # co-hosted with the web app.
    crypto_broker: Any = None
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
    hub: "InsightsHub"
    analytics: "PerformanceAnalytics"
    settings: "Settings"
    bus: "EventBus"
    runtime: RuntimeView


@dataclass(frozen=True, slots=True)
class BotContext:
    """Same primitives as the dashboard plus the bot-only deps.

    Built once by ``crypto/components.py:build_components`` (and the
    stocks counterpart). Passed into every cycle / monitor /
    background loop so nothing has to reach for ``get_settings()``
    ad-hoc.
    """

    engine: "AsyncEngine"
    repo: "Repository"
    hub: "InsightsHub"
    analytics: "PerformanceAnalytics"
    settings: "Settings"
    bus: "EventBus"
    runtime: RuntimeView

    def to_dashboard_context(self) -> DashboardContext:
        """Project the bot's context onto the dashboard's narrower shape."""
        return DashboardContext(
            engine=self.engine,
            repo=self.repo,
            hub=self.hub,
            analytics=self.analytics,
            settings=self.settings,
            bus=self.bus,
            runtime=self.runtime,
        )
