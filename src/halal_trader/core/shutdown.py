"""Graceful-shutdown helpers — cancel in-flight orders before disposing brokers.

When the bot exits (operator SIGTERM, container restart, host reboot)
we want to leave the broker in a clean state. Specifically: any *open
orders we placed* — sit-on-book limits, in-flight market orders awaiting
fill — should be cancelled so the next process can rebuild state from
balances without inheriting half-filled phantom positions.

This module is intentionally tiny — a pure helper that takes any broker
exposing ``get_open_orders`` + ``cancel_order`` and walks them. No
concrete broker imports, so it's testable with a stub.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class CancellableBroker(Protocol):
    """Minimum surface needed to cancel orders during shutdown."""

    async def get_open_orders(self) -> list[dict[str, Any]]: ...
    async def cancel_order(self, symbol: str, order_id: str) -> Any: ...


@dataclass(frozen=True)
class CancelResult:
    """Outcome of one shutdown cancellation pass."""

    cancelled: list[str]
    failed: list[tuple[str, str]]  # (order_id, error_message)


async def cancel_all_open_orders(
    broker: CancellableBroker,
    *,
    timeout: float = 10.0,
) -> CancelResult:
    """Cancel every open order on the broker, swallowing per-order failures.

    The shutdown path is best-effort: if cancelling one order fails (the
    exchange is down, the order is already filling), we log and continue
    so the rest still get cancelled. The whole pass is wrapped in a
    timeout so a stuck broker can't block process exit indefinitely.
    """
    try:
        orders = await asyncio.wait_for(broker.get_open_orders(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("get_open_orders timed out during shutdown")
        return CancelResult(cancelled=[], failed=[("?", "get_open_orders timed out")])
    except Exception as e:
        logger.warning("get_open_orders failed during shutdown: %s", e)
        return CancelResult(cancelled=[], failed=[("?", str(e))])

    cancelled: list[str] = []
    failed: list[tuple[str, str]] = []

    async def _cancel_one(order: dict[str, Any]) -> None:
        oid = str(order.get("orderId") or order.get("order_id") or order.get("id") or "")
        symbol = str(order.get("symbol") or order.get("pair") or "")
        if not oid:
            failed.append(("?", "no order id in payload"))
            return
        try:
            await asyncio.wait_for(
                broker.cancel_order(symbol=symbol, order_id=oid), timeout=timeout
            )
            cancelled.append(oid)
        except asyncio.TimeoutError:
            failed.append((oid, "cancel timed out"))
        except Exception as e:
            failed.append((oid, str(e)))

    if orders:
        await asyncio.gather(*[_cancel_one(o) for o in orders], return_exceptions=False)

    if cancelled:
        logger.info("Cancelled %d open orders during shutdown", len(cancelled))
    if failed:
        logger.warning(
            "Failed to cancel %d order(s) during shutdown: %s",
            len(failed),
            ", ".join(f"{oid}({err})" for oid, err in failed[:5]),
        )
    return CancelResult(cancelled=cancelled, failed=failed)
