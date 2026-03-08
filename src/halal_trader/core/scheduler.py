"""Base trading bot — shared lifecycle for stock and crypto bots."""

from __future__ import annotations

import abc
import logging

from halal_trader.config import get_settings
from halal_trader.db.models import init_db
from halal_trader.db.repository import Repository

logger = logging.getLogger(__name__)


class BaseTradingBot(abc.ABC):
    """Abstract base providing the shared lifecycle: init → run → shutdown."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._running = False
        self._engine = None
        self._repo: Repository | None = None

    async def initialize(self) -> None:
        """Set up the database and delegate component creation to the subclass."""
        engine = await init_db(str(self.settings.resolve_db_path()))
        self._engine = engine
        self._repo = Repository(engine)
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
    def _get_cycle_service(self):
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
