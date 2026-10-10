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
# The monitor closed a position on another rule (trailing stop, trend break, ...).
TRADE_EXIT_OTHER: Final[str] = "trade.exit.other"
# What became of a reactor entry the news asked for (placed, skipped, refused).
REACTOR_ENTRY: Final[str] = "reactor.entry"

# ── LLM ─────────────────────────────────────────────────────────
LLM_CALL_COMPLETE: Final[str] = "llm.call.complete"
LLM_CHAIN_BACKOFF: Final[str] = "llm.chain.backoff"
LLM_INSUFFICIENT_QUOTA: Final[str] = "llm.insufficient_quota"

# ── Risk / Reconciliation ──────────────────────────────────────
RECONCILE_DRIFT: Final[str] = "reconcile.drift"
SAFEGUARD_VIOLATION: Final[str] = "safeguards.violation"

# ── Scheduler ───────────────────────────────────────────────────
# Every scheduled job's run (TradingBot._job), under one job_id.
JOB_START: Final[str] = "scheduler.job.start"
JOB_COMPLETE: Final[str] = "scheduler.job.complete"
JOB_SKIPPED: Final[str] = "scheduler.job.skipped"
JOB_FAILED: Final[str] = "scheduler.job.failed"
# A daily job whose scheduled run a restart skipped, run late on startup.
JOB_CATCH_UP: Final[str] = "scheduler.catch_up"

# ── Tracing ─────────────────────────────────────────────────────
TRACE_SPAN: Final[str] = "trace.span"

# ── Research simulator (halabot/playbooks: research runs, never trades) ──
# A sim.run: its start, each batch of outcomes written, and its summary.
SIM_RUN_START: Final[str] = "playbooks.sim.run.start"
SIM_RUN_BATCH: Final[str] = "playbooks.sim.run.batch"
SIM_RUN_DONE: Final[str] = "playbooks.sim.run.done"

# ── News research (events/: descriptive and pre-registered runs, never trades) ──
# The atlas's rebuilt stories that differ from their news_stories rows (one per run).
ATLAS_STORIES_MISMATCH: Final[str] = "events.atlas.stories_mismatch"
# A Phase 0 gate's run, written to the ledger (events/sim_gate.py), or its refusal.
SIM_GATE_RUN: Final[str] = "research.sim_gate.run"
SIM_GATE_REFUSED: Final[str] = "research.sim_gate.refused"
