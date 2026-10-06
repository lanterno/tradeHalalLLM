"""Stable event-name constants used in structured log records.

Every emit of `logger.info(..., extra={"event": EVENT_NAME, ...})` should use
one of these constants so log-grep / metrics endpoints / dashboards can rely
on a fixed namespace. Add new events here before using them.
"""

from __future__ import annotations

from typing import Final

# ── Cycle lifecycle ─────────────────────────────────────────────
CYCLE_START: Final[str] = "cycle.start"
CYCLE_COMPLETE: Final[str] = "cycle.complete"
CYCLE_SKIPPED: Final[str] = "cycle.skipped"
CYCLE_HALTED: Final[str] = "cycle.halted"
CYCLE_FAILED: Final[str] = "cycle.failed"
CYCLE_NO_ACTION: Final[str] = "cycle.no_action"

# ── Quant band maintenance (advisory) ──────────────────────────
BAND_COVERAGE_DRIFT: Final[str] = "band.coverage_drift"

# ── Trades ──────────────────────────────────────────────────────
TRADE_BUY_PLACED: Final[str] = "trade.buy.placed"
TRADE_SELL_PLACED: Final[str] = "trade.sell.placed"
# A BUY refused at the order boundary because the symbol is not (provably)
# halal -- the screen said no, or could not be read (fail closed).
HALAL_GATE_REJECTED: Final[str] = "trade.rejected.halal"
TRADE_FILL_PARTIAL: Final[str] = "trade.fill.partial"
TRADE_EXIT_SL: Final[str] = "trade.exit.stop_loss"
TRADE_EXIT_TP: Final[str] = "trade.exit.take_profit"

# ── LLM ─────────────────────────────────────────────────────────
LLM_CALL_COMPLETE: Final[str] = "llm.call.complete"
LLM_CHAIN_BACKOFF: Final[str] = "llm.chain.backoff"

# ── Risk / Reconciliation ──────────────────────────────────────
RECONCILE_DRIFT: Final[str] = "reconcile.drift"
