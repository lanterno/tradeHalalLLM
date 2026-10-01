"""Panic-button auto-liquidation for the operator kill-switch.

Used by `halal-trader halt --close-all stocks`. Alpaca exposes
`close_all_positions(cancel_orders=True)`, which is exactly what we want:
one call, then a result list the CLI can display. Best-effort: a failure
is surfaced as an error row rather than raised.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LiquidationResult:
    market: str  # 'stocks'
    symbol: str
    quantity: float
    status: str  # 'closed' | 'skipped' | 'error'
    detail: str = ""


async def liquidate_stocks(broker: Any) -> list[LiquidationResult]:
    """Close every open Alpaca position via the broker's batch endpoint."""
    try:
        await broker.close_all_positions()
    except Exception as exc:
        return [
            LiquidationResult(
                market="stocks",
                symbol="*",
                quantity=0.0,
                status="error",
                detail=str(exc),
            )
        ]

    try:
        positions = await broker.get_all_positions()
    except Exception:
        positions = []

    return [
        LiquidationResult(
            market="stocks",
            symbol=p.symbol,
            quantity=float(p.qty),
            status="closed",
            detail="batch close_all_positions",
        )
        for p in positions
    ]
