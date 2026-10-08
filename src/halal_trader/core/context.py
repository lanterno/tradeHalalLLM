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

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from halal_trader.config import Settings
    from halal_trader.core.event_bus import EventBus
    from halal_trader.db.repository import Repository
    from halal_trader.portfolio.analytics import PerformanceAnalytics


@dataclass
class RuntimeView:
    """What the web process knows about itself: when it started.

    The bot runs in another container, so nothing it holds in memory
    reaches the web; its state comes from the database (heartbeats,
    snapshots, the ledger).
    """

    started_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class DashboardContext:
    """Read-only deps + a mutable :class:`RuntimeView` window.

    Routes take this via FastAPI ``Depends`` (see ``web/dependencies.py``)
    instead of reaching into ``app_state``. The frozen fields are the
    long-lived deps; ``runtime`` is the only place that mutates.
    """

    engine: AsyncEngine
    repo: Repository
    analytics: PerformanceAnalytics
    settings: Settings
    bus: EventBus
    runtime: RuntimeView
