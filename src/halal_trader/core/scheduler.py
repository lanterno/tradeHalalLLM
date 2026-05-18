"""Base trading bot — shared lifecycle for stock and crypto bots."""

from __future__ import annotations

import abc
import logging
from datetime import timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.config import get_settings
from halal_trader.db.models import init_db
from halal_trader.db.repos import RepoBundle
from halal_trader.db.repository import Repository

logger = logging.getLogger(__name__)


class BaseTradingBot(abc.ABC):
    """Abstract base providing the shared lifecycle: init → run → shutdown."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._running = False
        self._engine: AsyncEngine | None = None
        self._repo: Repository | None = None
        self._bundle: RepoBundle | None = None

    async def initialize(self) -> None:
        """Set up the database and delegate component creation to the subclass."""
        engine = await init_db(self.settings.database_url)
        self._engine = engine
        self._repo = Repository(engine)
        # Typed per-table bundle — subclasses can hand narrow protocols
        # to components instead of the full ``Repository``.
        self._bundle = RepoBundle.from_engine(engine)
        await self._create_components()

    @abc.abstractmethod
    async def _create_components(self) -> None:
        """Create domain-specific components (broker, strategy, executor, etc.)."""

    @abc.abstractmethod
    async def _daily_start(self) -> None:
        """Pre-market / daily start routine."""

    @abc.abstractmethod
    async def _daily_end(self) -> None:
        """End-of-day routine."""

    @abc.abstractmethod
    def _get_cycle_service(self) -> Any:
        """Return the cycle service whose ``run_cycle`` drives one iteration."""

    async def run_once(self) -> None:
        """Run a single trading cycle (useful for testing)."""
        await self.initialize()
        try:
            await self._daily_start()
            await self._get_cycle_service().run_cycle()
            await self._daily_end()
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Dispose the database engine and mark the bot as stopped."""
        self._running = False
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None

    @abc.abstractmethod
    async def run(self) -> None:
        """Start the bot's main loop (subclass-specific)."""

    async def _prune_audit_log(self) -> None:
        """Delete ``web_actions`` rows older than the retention window.

        Subclasses call this from their ``_daily_end`` so the audit
        table doesn't grow unbounded on long-running deployments. A
        retention of ``0`` disables the prune.
        """
        retention = int(getattr(self.settings.web, "audit_retention_days", 0) or 0)
        if retention <= 0 or self._bundle is None:
            return
        try:
            deleted = await self._bundle.web_audit.delete_old_web_actions(
                older_than=timedelta(days=retention)
            )
            if deleted:
                logger.info("Pruned %d web_actions row(s) older than %d days", deleted, retention)
        except Exception as exc:  # noqa: BLE001
            logger.warning("web_actions prune failed: %s", exc)
