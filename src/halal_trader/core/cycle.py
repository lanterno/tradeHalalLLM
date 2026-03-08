"""Base class for trading cycle services."""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseCycleService(ABC):
    """Thin base for stock and crypto trading cycle services.

    Provides the run_cycle() template: pre-checks → halt check →
    market-specific work → post-cycle hook.  Subclasses fill in the
    abstract hooks with their own data flows.
    """

    def __init__(self) -> None:
        pass

    async def run_cycle(self) -> None:
        """Execute one complete trading cycle (template method)."""
        logger.info("=== TRADING CYCLE START ===")
        try:
            if not await self._pre_cycle_checks():
                return

            if await self._should_halt():
                return

            await self._run_cycle_impl()
            await self._post_cycle()

        except Exception as e:
            logger.error("Trading cycle failed: %s", e, exc_info=True)

    @abstractmethod
    async def _pre_cycle_checks(self) -> bool:
        """Return True if the cycle should proceed, False to skip."""
        ...

    @abstractmethod
    async def _should_halt(self) -> bool:
        """Return True if trading should be halted (e.g. loss limits)."""
        ...

    @abstractmethod
    async def _run_cycle_impl(self) -> None:
        """Market-specific cycle logic: data gathering, analysis, execution."""
        ...

    async def _post_cycle(self) -> None:
        """Optional post-cycle hook (no-op by default)."""
