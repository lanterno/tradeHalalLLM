"""Base class for trading cycle services."""

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from halal_trader.core import events
from halal_trader.core.observability import cycle_context

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from halal_trader.notifications.telegram import AlertSink

logger = logging.getLogger(__name__)


class BaseCycleService(ABC):
    """Thin base for stock and crypto trading cycle services.

    Provides the run_cycle() template: kill-switch → pre-checks →
    halt check → market-specific work → post-cycle hook.  Subclasses
    fill in the abstract hooks with their own data flows.
    """

    def __init__(
        self,
        alerts: "AlertSink | None" = None,
        engine: "AsyncEngine | None" = None,
    ) -> None:
        self._alerts = alerts
        # The engine is only needed for the kill-switch lookup. Tests that
        # exercise the cycle template can leave it None — the check skips.
        self._engine = engine
        # Subclasses may set ``self._bus`` to publish cycle / stage events.
        self._bus: object | None = None

    async def run_cycle(self) -> None:
        """Execute one complete trading cycle (template method)."""
        with cycle_context() as cid:
            logger.info(
                "=== TRADING CYCLE START === (%s)",
                cid,
                extra={"event": events.CYCLE_START},
            )
            t0 = time.monotonic()
            # Publish to the event bus so /ws/cycle subscribers see the start.
            await self._publish_event("cycle.start", {"cycle_id": cid})
            try:
                if await self._kill_switch_engaged():
                    logger.warning(
                        "Cycle halted by operator kill-switch",
                        extra={
                            "event": events.CYCLE_HALTED,
                            "reason": "kill_switch",
                            "elapsed_ms": int((time.monotonic() - t0) * 1000),
                        },
                    )
                    return

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
                        "Cycle halted (loss limit)",
                        extra={
                            "event": events.CYCLE_HALTED,
                            "reason": "loss_limit",
                            "elapsed_ms": int((time.monotonic() - t0) * 1000),
                        },
                    )
                    return

                await self._run_cycle_impl()
                await self._post_cycle()

                elapsed_ms = int((time.monotonic() - t0) * 1000)
                logger.info(
                    "=== TRADING CYCLE COMPLETE ===",
                    extra={
                        "event": events.CYCLE_COMPLETE,
                        "elapsed_ms": elapsed_ms,
                    },
                )
                await self._publish_event(
                    "cycle.complete",
                    {"cycle_id": cid, "elapsed_ms": elapsed_ms},
                )

            except Exception as e:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                logger.error(
                    "Trading cycle failed: %s",
                    e,
                    exc_info=True,
                    extra={
                        "event": events.CYCLE_FAILED,
                        "elapsed_ms": elapsed_ms,
                    },
                )
                await self._publish_event(
                    "cycle.failed",
                    {"cycle_id": cid, "elapsed_ms": elapsed_ms, "error": repr(e)},
                )
                if self._alerts is not None:
                    await self._alerts.notify(
                        events.CYCLE_FAILED,
                        f"{type(e).__name__}: {e}",
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

    async def _publish_event(self, topic: str, payload: "dict[str, object] | None" = None) -> None:
        """Best-effort publish to ``self._bus`` if a bus is wired."""
        bus = getattr(self, "_bus", None)
        if bus is None:
            return
        try:
            await bus.publish(topic, payload or {})
        except Exception:  # noqa: BLE001
            pass

    async def _kill_switch_engaged(self) -> bool:
        """Return True if the operator kill-switch is set."""
        if self._engine is None:
            return False
        from halal_trader.core.halt import is_halted

        return await is_halted(self._engine)
