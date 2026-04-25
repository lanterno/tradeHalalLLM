"""Base class for trading cycle services."""

import logging
import time
from abc import ABC, abstractmethod

from halal_trader.core import events
from halal_trader.core.observability import cycle_context

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
        with cycle_context() as cid:
            logger.info(
                "=== TRADING CYCLE START === (%s)",
                cid,
                extra={"event": events.CYCLE_START},
            )
            t0 = time.monotonic()
            try:
                if not await self._pre_cycle_checks():
                    logger.info(
                        "Cycle skipped (pre-cycle gate)",
                        extra={
                            "event": events.CYCLE_SKIPPED,
                            "reason": "pre_cycle",
                            "elapsed_ms": int((time.monotonic() - t0) * 1000),
                        },
                    )
                    return

                if await self._should_halt():
                    logger.info(
                        "Cycle halted (loss limit / kill-switch)",
                        extra={
                            "event": events.CYCLE_HALTED,
                            "elapsed_ms": int((time.monotonic() - t0) * 1000),
                        },
                    )
                    return

                await self._run_cycle_impl()
                await self._post_cycle()

                logger.info(
                    "=== TRADING CYCLE COMPLETE ===",
                    extra={
                        "event": events.CYCLE_COMPLETE,
                        "elapsed_ms": int((time.monotonic() - t0) * 1000),
                    },
                )

            except Exception as e:
                logger.error(
                    "Trading cycle failed: %s",
                    e,
                    exc_info=True,
                    extra={
                        "event": events.CYCLE_FAILED,
                        "elapsed_ms": int((time.monotonic() - t0) * 1000),
                    },
                )

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
