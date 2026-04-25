"""Order-fill confirmation — turn broker order responses into FillResult.

Both bots persist `submitted_at`, `filled_at`, `filled_price`, and
`filled_quantity` so reconciliation, alerts, and metrics can rely on
"submitted" vs "filled" being distinct rather than conflated as
``status='pending'`` forever.

For Binance: market orders return immediately with the fill data already
populated, so :func:`confirm_binance` parses the response.

For Alpaca (stocks): MCP tool calls return a submission ack — we have to
poll ``get_orders`` until the order is ``filled``, ``partially_filled``,
``rejected``, ``canceled``, or the timeout elapses.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable

from halal_trader.core import events

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FillResult:
    """Outcome of a fill confirmation.

    ``status`` is one of:
      * ``filled`` – the order completely executed.
      * ``partially_filled`` – some quantity executed, polling timed out.
      * ``rejected`` – exchange refused the order outright.
      * ``canceled`` – the order was canceled before/during fill.
      * ``pending`` – submission acked but no fill information yet
        (timeout reached, or broker doesn't surface fills).
    """

    status: str
    order_id: str
    filled_quantity: float
    filled_price: float | None
    submitted_at: datetime
    filled_at: datetime | None
    raw: dict[str, Any]


_TERMINAL_STATES = {"filled", "rejected", "canceled", "expired"}


def confirm_binance(order_response: dict[str, Any], submitted_at: datetime) -> FillResult:
    """Translate a Binance order response into a :class:`FillResult`.

    Binance MARKET orders come back with ``status``, ``executedQty``, and
    either ``fills`` (newer) or ``cumulativeQuoteQty`` (older). Only
    ``status='FILLED'`` rows count as fully filled; ``PARTIALLY_FILLED``
    falls through as the same so callers can decide.
    """
    order_id = str(order_response.get("orderId", ""))
    raw_status = str(order_response.get("status", "")).lower() or "pending"

    fills = order_response.get("fills") or []
    if fills:
        total_qty = sum(float(f.get("qty", 0)) for f in fills)
        total_cost = sum(float(f.get("price", 0)) * float(f.get("qty", 0)) for f in fills)
        filled_qty = total_qty
        filled_price: float | None = (total_cost / total_qty) if total_qty > 0 else None
    else:
        executed = float(order_response.get("executedQty", 0))
        cumulative = float(order_response.get("cumulativeQuoteQty", 0))
        filled_qty = executed
        filled_price = (cumulative / executed) if executed > 0 and cumulative > 0 else None

    status_map = {
        "filled": "filled",
        "partially_filled": "partially_filled",
        "rejected": "rejected",
        "canceled": "canceled",
        "expired": "rejected",
        "new": "pending",
        "pending_new": "pending",
    }
    status = status_map.get(raw_status, raw_status)

    filled_at = datetime.now(UTC) if status == "filled" and filled_qty > 0 else None

    return FillResult(
        status=status,
        order_id=order_id,
        filled_quantity=filled_qty,
        filled_price=filled_price,
        submitted_at=submitted_at,
        filled_at=filled_at,
        raw=order_response,
    )


async def confirm_alpaca(
    poll: Callable[[], "asyncio.Future[dict[str, Any]] | dict[str, Any]"],
    *,
    order_id: str,
    submitted_at: datetime,
    timeout: float = 30.0,
    interval: float = 2.0,
) -> FillResult:
    """Poll a fetch-order callback until the order reaches a terminal state.

    Args:
        poll: callable that returns the current order dict (with at least a
            ``status`` field). Awaitable or sync.
        order_id: the broker's order id (echoed onto the result).
        submitted_at: the submission timestamp captured at place_order time.
        timeout: max total polling time in seconds.
        interval: seconds between polls.

    Returns ``status="partially_filled"`` if the loop times out with some
    quantity already executed, ``status="pending"`` if no fill data was
    available, or whatever terminal state the broker reported.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    last: dict[str, Any] = {}
    while True:
        result = poll()
        if asyncio.iscoroutine(result):
            last = await result  # type: ignore[assignment]
        else:
            last = result  # type: ignore[assignment]
        status = str(last.get("status", "")).lower() or "pending"
        if status in _TERMINAL_STATES:
            return _build_alpaca_result(order_id, status, last, submitted_at)
        if asyncio.get_event_loop().time() >= deadline:
            qty = float(last.get("filled_qty", 0) or last.get("filled_quantity", 0) or 0)
            timeout_status = "partially_filled" if qty > 0 else "pending"
            logger.warning(
                "Order %s did not reach terminal state within %.0fs (last=%s)",
                order_id,
                timeout,
                status,
                extra={
                    "event": events.TRADE_FILL_PARTIAL if qty > 0 else "trade.fill.timeout",
                    "order_id": order_id,
                    "status": timeout_status,
                    "filled_quantity": qty,
                },
            )
            return _build_alpaca_result(order_id, timeout_status, last, submitted_at)
        await asyncio.sleep(interval)


def _build_alpaca_result(
    order_id: str,
    status: str,
    raw: dict[str, Any],
    submitted_at: datetime,
) -> FillResult:
    qty = float(raw.get("filled_qty", 0) or raw.get("filled_quantity", 0) or 0)
    fill_price = raw.get("filled_avg_price") or raw.get("avg_price")
    fill_price_f = float(fill_price) if fill_price is not None else None
    filled_at = datetime.now(UTC) if status == "filled" and qty > 0 else None
    return FillResult(
        status=status,
        order_id=order_id,
        filled_quantity=qty,
        filled_price=fill_price_f,
        submitted_at=submitted_at,
        filled_at=filled_at,
        raw=raw,
    )
