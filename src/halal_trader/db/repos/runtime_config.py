"""Runtime config overlay repository — operator-set values that override Settings.

Wave D extraction. Values are JSONB so any JSON-shaped overlay works
(``crypto.max_position_pct: 0.05``, ``llm.shadow_enabled: true``, …).
The matching ``RuntimeConfigRepo`` Protocol lives in ``protocols.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import RuntimeConfig


class RuntimeConfigRepoImpl:
    """Concrete implementation of :class:`RuntimeConfigRepo`."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def set_runtime_config(self, key: str, value: Any, *, set_by: str | None = None) -> None:
        """Insert/update a runtime overlay value (JSONB — any JSON shape)."""
        async with AsyncSession(self._engine) as session:
            row = await session.get(RuntimeConfig, key.upper())
            if row is None:
                row = RuntimeConfig(key=key.upper(), value=value, set_by=set_by)
            else:
                row.value = value
                row.set_by = set_by
                row.set_at = datetime.now(UTC)
            session.add(row)
            await session.commit()

    async def delete_runtime_config(self, key: str) -> bool:
        async with AsyncSession(self._engine) as session:
            row = await session.get(RuntimeConfig, key.upper())
            if row is None:
                return False
            await session.delete(row)
            await session.commit()
            return True

    async def list_runtime_config(self) -> dict[str, Any]:
        async with AsyncSession(self._engine) as session:
            results = await session.exec(select(RuntimeConfig))
            return {r.key: r.value for r in results.all()}
