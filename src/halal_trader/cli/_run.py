"""What every database-backed command shares.

``run_db`` opens a checked engine, runs the work and disposes the engine
whatever happens; ``fail`` ends a command with one error style (red
"Error:", exit 1). Heavy modules stay imported lazily so ``--help`` is fast.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, NoReturn, TypeVar

import click

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from halal_trader.config import Settings

T = TypeVar("T")


def run_db(work: Callable[[AsyncEngine, Settings], Awaitable[T]]) -> T:
    """Run ``work(engine, settings)`` to completion on a fresh engine."""

    async def _run() -> T:
        from halal_trader.config import get_settings
        from halal_trader.db.models import open_db

        settings = get_settings()
        async with open_db(settings.database_url) as engine:
            return await work(engine, settings)

    return asyncio.run(_run())


def fail(message: str) -> NoReturn:
    """Stop the command: ``Error: <message>`` on stderr, exit status 1."""
    raise click.ClickException(message)
