"""Test helper: write heartbeat rows the way the bot does, from sync tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import create_async_engine


def write_beat(database_url: str, component: str, detail: dict[str, Any] | None = None) -> None:
    from halal_trader.core.heartbeat import beat

    async def go() -> None:
        engine = create_async_engine(database_url)
        try:
            await beat(engine, component, detail, now=datetime.now(UTC))
        finally:
            await engine.dispose()

    asyncio.run(go())
