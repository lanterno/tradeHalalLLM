"""Web audit repository — pending/completed audit rows for dashboard mutations.

First mini-repo extracted from the monolithic ``Repository`` class as
part of Wave D of ``docs/cleanup_roadmap.md``. Each table's data access
moves into its own ≤80-line module; the existing ``Repository`` class
delegates to these so call sites can migrate incrementally.

The matching ``WebAuditRepo`` Protocol lives in ``protocols.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import WebAction


class WebAuditRepoImpl:
    """Concrete implementation of :class:`WebAuditRepo`."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def begin_web_action(
        self, *, actor: str, method: str, path: str, payload: str | None = None
    ) -> int:
        """Insert a 'pending' web_actions row before the handler runs."""
        row = WebAction(actor=actor, method=method, path=path, payload=payload)
        async with AsyncSession(self._engine) as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
            assert row.id is not None
            return row.id

    async def complete_web_action(
        self, action_id: int, *, status_code: int, error: str | None = None
    ) -> None:
        """Update a pending row with the final outcome."""
        async with AsyncSession(self._engine) as session:
            row = await session.get(WebAction, action_id)
            if row is None:
                return
            row.status_code = status_code
            row.outcome = "ok" if 200 <= status_code < 400 and error is None else "error"
            row.error = error
            session.add(row)
            await session.commit()

    async def get_recent_web_actions(self, limit: int = 50) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = select(WebAction).order_by(col(WebAction.id).desc()).limit(limit)
            results = await session.exec(statement)
            return [r.model_dump() for r in results.all()]

    async def delete_old_web_actions(self, *, older_than: timedelta) -> int:
        """Prune ``web_actions`` rows older than ``older_than``.

        Returns the number of rows deleted. The daily-end scheduler hook
        calls this so a long-running deployment doesn't accumulate
        unbounded mutation-audit rows.
        """
        cutoff = datetime.now(UTC) - older_than
        async with AsyncSession(self._engine) as session:
            result = await session.exec(
                sa_delete(WebAction).where(col(WebAction.timestamp) < cutoff)
            )
            await session.commit()
            return int(result.rowcount or 0)
