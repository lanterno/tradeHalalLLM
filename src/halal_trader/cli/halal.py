"""Halal-compliance CLI (Wave L) — operator-side explainer.

Mirrors the dashboard's ``/api/halal/explain/{trade_id}`` endpoint for
terminal use. Pulls the trade + its screening receipt and renders the
criteria blob as Markdown with citations to
``docs/halal_jurisprudence.md``.
"""

from __future__ import annotations

import asyncio

import click
from rich.markdown import Markdown

from halal_trader.logging import console


@click.group("halal")
def halal_group() -> None:
    """Halal-compliance operations."""


@halal_group.command("explain")
@click.argument("trade_id", type=int)
@click.option(
    "--asset-class",
    type=click.Choice(["crypto", "stock"]),
    default="crypto",
    show_default=True,
)
def explain(trade_id: int, asset_class: str) -> None:
    """Render the Sharia-compliance explanation for one trade."""
    asyncio.run(_run_explain(trade_id=trade_id, asset_class=asset_class))


async def _run_explain(*, trade_id: int, asset_class: str) -> None:
    from halal_trader.config import get_settings
    from halal_trader.db import init_db
    from halal_trader.halal.audit import export_receipt
    from halal_trader.halal.explainer import explain_screening

    settings = get_settings()
    engine = await init_db(settings.database_url)
    receipt = await export_receipt(engine, trade_id=trade_id, asset_class=asset_class)
    if receipt is None:
        console.print(f"[red]Trade {trade_id} ({asset_class}) not found.[/red]")
        return
    explanation = explain_screening(receipt.payload)
    console.print(Markdown(explanation.body_md))
    if explanation.sources:
        console.print()
        console.print("[dim]Sources:[/dim]")
        for src in explanation.sources:
            console.print(f"  • {src}")
