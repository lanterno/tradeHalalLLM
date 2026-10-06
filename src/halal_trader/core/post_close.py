"""Post-close analytics fan-out — one call hooks every recorder.

The monitor's close path fires the post-close recorders (RAG rationale,
round-trip purification) through one call instead of inline at the
close-site.

This module exposes a single :func:`record_close` that takes a small
:class:`CloseEvent` describing the closed trade plus an optional
context (reasoning) and dispatches to:

* :class:`DBRationaleStore` (RAG over reasoning + outcome)
* :class:`RoundTripLedger` purification accrual

Each step is best-effort: a failure in one recorder does not prevent
the others from running. Errors are logged at debug level; the call
to :func:`record_close` never raises.

The stock bot wires only the RAG store (``trading/scheduler.py``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


# ── Close event ──────────────────────────────────────────────────


@dataclass
class CloseEvent:
    """Minimum data needed by every post-close recorder."""

    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell' — almost always 'buy' for closing entries
    entry_price: float
    exit_price: float
    exit_reason: str
    realized_pnl_usd: float
    return_pct: float
    quantity: float = 0.0
    reasoning: str = ""
    setup_type: str | None = None
    hold_seconds: int = 0
    closed_at: datetime | None = None


# ── Recorder bundle ──────────────────────────────────────────────


@dataclass
class CloseRecorders:
    """Holds the writers used by :func:`record_close`.

    Each field is optional; ``None`` means that recorder is skipped.
    All store fields are async — DB-backed.
    """

    purification_ledger: Any | None = None  # RoundTripLedger
    purification_rules: Mapping[str, Any] | None = None
    rag_store: Any | None = None  # DBRationaleStore


# ── Main entry point ─────────────────────────────────────────────


async def record_close(event: CloseEvent, recorders: CloseRecorders) -> dict[str, Any]:
    """Dispatch a close event to every configured recorder.

    Returns a dict summarising what fired, suitable for INFO logging.
    Errors are caught per-recorder so partial failure is observable
    but never blocks the close-path.
    """
    summary: dict[str, Any] = {
        "trade_id": event.trade_id,
        "symbol": event.symbol,
        "return_pct": event.return_pct,
    }

    # RAG store — embed the rationale + outcome for later retrieval.
    if recorders.rag_store is not None and event.reasoning:
        try:
            await recorders.rag_store.add(
                trade_id=event.trade_id,
                symbol=event.symbol,
                text=event.reasoning,
                outcome_pnl_pct=event.return_pct,
                setup_type=event.setup_type,
                timestamp=(event.closed_at or datetime.now(UTC)).isoformat(),
            )
            summary["rag_added"] = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("rag store add failed: %s", exc)

    # Round-trip purification.
    if (
        recorders.purification_ledger is not None
        and recorders.purification_rules is not None
        and event.realized_pnl_usd > 0
    ):
        try:
            from halal_trader.halal.round_trip_purification import (
                record_round_trip,
            )

            entry = await record_round_trip(
                recorders.purification_ledger,
                recorders.purification_rules,
                trade_id=event.trade_id,
                symbol=event.symbol,
                gain_usd=event.realized_pnl_usd,
            )
            if entry is not None:
                summary["purification_due_usd"] = entry.purification_due_usd
        except Exception as exc:  # noqa: BLE001
            logger.debug("purification recorder failed: %s", exc)

    return summary
