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
TRADE_REJECTED: Final[str] = "trade.rejected"
TRADE_FILL_PARTIAL: Final[str] = "trade.fill.partial"
TRADE_FILLED: Final[str] = "trade.filled"
TRADE_EXIT_SL: Final[str] = "trade.exit.stop_loss"
TRADE_EXIT_TP: Final[str] = "trade.exit.take_profit"
TRADE_EXIT_FORCED: Final[str] = "trade.exit.forced"

# ── LLM ─────────────────────────────────────────────────────────
LLM_CALL_COMPLETE: Final[str] = "llm.call.complete"
LLM_CALL_FAILED: Final[str] = "llm.call.failed"
LLM_FALLBACK_TRIGGERED: Final[str] = "llm.fallback.triggered"
LLM_CHAIN_BACKOFF: Final[str] = "llm.chain.backoff"

# ── Risk / Reconciliation ──────────────────────────────────────
RISK_HALT: Final[str] = "risk.halt"
RECONCILE_DRIFT: Final[str] = "reconcile.drift"

# ── Operational ────────────────────────────────────────────────
HALT_SET: Final[str] = "halt.set"
HALT_CLEARED: Final[str] = "halt.cleared"
NEWS_EVENT: Final[str] = "news.event"
