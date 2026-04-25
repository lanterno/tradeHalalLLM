"""Click CLI entrypoint with Rich terminal output.

Sub-command modules each export a Click command (or group) and are
attached to the top-level ``cli`` group below. Keep new commands as new
modules; do not add inline command definitions here.
"""

from __future__ import annotations

import click

from halal_trader.cli import backup as backup_cmd
from halal_trader.cli import crypto as crypto_cmd
from halal_trader.cli import dashboard as dashboard_cmd
from halal_trader.cli import db as db_cmd
from halal_trader.cli import halt as halt_cmd
from halal_trader.cli import reconcile as reconcile_cmd
from halal_trader.cli import stocks as stocks_cmd


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

# ── Reconciliation + Backup ────────────────────────────────────
cli.add_command(reconcile_cmd.reconcile)
cli.add_command(backup_cmd.backup)

# ── Crypto ─────────────────────────────────────────────────────
cli.add_command(crypto_cmd.crypto)

# ── Dashboard ──────────────────────────────────────────────────
cli.add_command(dashboard_cmd.dashboard)


__all__ = ["cli"]
