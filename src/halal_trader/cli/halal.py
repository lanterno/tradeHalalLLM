"""Halal-compliance CLI — operator-side explainer.

Mirrors the dashboard's ``/api/halal/explain/{trade_id}`` endpoint for
terminal use. Pulls the trade + its screening receipt and renders the
criteria blob as Markdown with citations to
``docs/halal_jurisprudence.md``.
"""

from __future__ import annotations

from typing import Any

import click
from rich.markdown import Markdown

from halal_trader.cli._run import fail, run_db
from halal_trader.logging import console


@click.group("halal")
def halal_group() -> None:
    """Halal-compliance operations."""


@halal_group.command("explain")
@click.argument("trade_id", type=int)
def explain(trade_id: int) -> None:
    """Render the Sharia-compliance explanation for one trade."""
    run_db(lambda engine, settings: _run_explain(engine, trade_id=trade_id))


async def _run_explain(engine: Any, *, trade_id: int) -> None:
    from halal_trader.halal.audit import export_receipt
    from halal_trader.halal.explainer import explain_screening

    receipt = await export_receipt(engine, trade_id=trade_id)
    if receipt is None:
        fail(f"trade {trade_id} not found")
    explanation = explain_screening(receipt.payload)
    console.print(Markdown(explanation.body_md))
    if explanation.sources:
        console.print()
        console.print("[dim]Sources:[/dim]")
        for src in explanation.sources:
            console.print(f"  • {src}")
