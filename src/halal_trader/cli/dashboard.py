"""Dashboard launcher command."""

from __future__ import annotations

import click
from rich.panel import Panel

from halal_trader.logging import console


@click.command("dashboard")
@click.option("--port", default=8082, help="Dashboard port")
@click.option("--host", default="0.0.0.0", help="Dashboard host")
def dashboard(port: int, host: str) -> None:
    """Launch the FastAPI + React web dashboard."""
    try:
        import uvicorn

        from halal_trader.web.app import create_app

        console.print(
            Panel(
                f"[bold green]Halal Trader Dashboard[/bold green]\n[dim]http://{host}:{port}[/dim]",
                title="Starting",
                border_style="green",
            )
        )
        app = create_app()
        uvicorn.run(app, host=host, port=port, log_level="info")
    except ImportError:
        console.print(
            "[red]Dashboard requires fastapi and uvicorn. "
            "Install with: pip install fastapi uvicorn[/red]"
        )
