"""Click CLI entrypoint with Rich terminal output.

Sub-command modules each export a Click command (or group) and are
attached to the top-level ``cli`` group below. Keep new commands as new
modules; do not add inline command definitions here.
"""

from __future__ import annotations

import click

from halal_trader.cli import books as books_cmd
from halal_trader.cli import broker as broker_cmd
from halal_trader.cli import compliance as compliance_cmd
from halal_trader.cli import dashboard as dashboard_cmd
from halal_trader.cli import data as data_cmd
from halal_trader.cli import db as db_cmd
from halal_trader.cli import events as events_cmd
from halal_trader.cli import halal as halal_cmd
from halal_trader.cli import halt as halt_cmd
from halal_trader.cli import insights as insights_cmd
from halal_trader.cli import ledger as ledger_cmd
from halal_trader.cli import llm_decisions as llm_decisions_cmd
from halal_trader.cli import purify as purify_cmd
from halal_trader.cli import quant as quant_cmd
from halal_trader.cli import recommend as recommend_cmd
from halal_trader.cli import reconcile as reconcile_cmd
from halal_trader.cli import research as research_cmd
from halal_trader.cli import stocks as stocks_cmd
from halal_trader.cli import watchdog as watchdog_cmd


@click.group()
@click.option(
    "--log-level",
    default=None,
    help="Override log level (DEBUG, INFO, WARNING, ERROR)",
)
def cli(log_level: str | None) -> None:
    """Halal Trader - LLM-powered halal day-trading bot."""
    from halal_trader.config import get_settings
    from halal_trader.logging import setup_logging

    settings = get_settings()
    setup_logging(settings, cli_log_level=log_level)


# ── Stocks ─────────────────────────────────────────────────────
cli.add_command(stocks_cmd.start)
cli.add_command(stocks_cmd.status)
cli.add_command(stocks_cmd.history)
cli.add_command(stocks_cmd.config)

# ── Database ────────────────────────────────────────────────────
cli.add_command(db_cmd.db_group)

# ── Operator (kill-switch) ─────────────────────────────────────
cli.add_command(halt_cmd.halt)
cli.add_command(halt_cmd.resume)
cli.add_command(halt_cmd.halt_status)

# ── In-house Shariah screen (research-grade until validated) ──
cli.add_command(compliance_cmd.compliance)

# ── Strategy research (backtests) ──────────────────────────────
cli.add_command(research_cmd.research)

# ── Event store (S2: news and filings, scored and labelled) ──────
cli.add_command(events_cmd.events)

# ── Purification of impermissible income ─────────────────────────
cli.add_command(purify_cmd.purify)

# ── Research market data ───────────────────────────────────────
cli.add_command(data_cmd.data)

# ── Paper-forward books (no orders) ────────────────────────────
cli.add_command(books_cmd.books)

# ── Broker ledger (the books of truth) ─────────────────────────
cli.add_command(ledger_cmd.ledger)

# ── Broker adapters, side by side ──────────────────────────────
cli.add_command(broker_cmd.broker)

# ── Reconciliation ─────────────────────────────────────────────
# Now a Click group: `reconcile check {market}` + `reconcile fix-orphans`
cli.add_command(reconcile_cmd.reconcile)

# ── LLM Decision Audit ─────────────────────────────────────────
cli.add_command(llm_decisions_cmd.llm_decisions)

# ── Dashboard ──────────────────────────────────────────────────
cli.add_command(dashboard_cmd.dashboard)

# ── Insights (purification / catalysts / RAG / receipts) ───────
cli.add_command(insights_cmd.insights)

# ── Daily halal recommendation (advisory stock-of-the-day) ─────
cli.add_command(recommend_cmd.recommend)

# ── Quantitative range-model tools (advisory) ──────────────────
cli.add_command(quant_cmd.quant)

# ── Halal compliance explainer (Wave L) ────────────────────────
cli.add_command(halal_cmd.halal_group)

# ── Dead-man-switch watchdog (out-of-process via launchd) ──────
cli.add_command(watchdog_cmd.watchdog)


__all__ = ["cli"]
