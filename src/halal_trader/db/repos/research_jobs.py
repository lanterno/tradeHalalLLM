"""Research jobs queue repository — backtest jobs and their results.

Wave D extraction. Each row tracks one async backtest run: parameters,
status (pending → running → ok/error), serialized result blob, and an
operator-driven ``pinned`` flag for keeping noteworthy runs around.
The matching ``ResearchJobRepo`` Protocol lives in ``protocols.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import ResearchJob


class ResearchJobRepoImpl:
    """Concrete implementation of :class:`ResearchJobRepo`."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def enqueue_research_job(
        self, *, kind: str, params: dict[str, Any], name: str | None = None
    ) -> int:
        row = ResearchJob(kind=kind, name=name, params=params)
        async with AsyncSession(self._engine) as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
            assert row.id is not None
            return row.id

    async def update_research_job(
        self,
        job_id: int,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        async with AsyncSession(self._engine) as session:
            row = await session.get(ResearchJob, job_id)
            if row is None:
                return
            row.status = status
            if result is not None:
                row.result = result
            if error is not None:
                row.error = error
            if status in ("ok", "error"):
                row.finished_at = datetime.now(UTC)
            session.add(row)
            await session.commit()

    async def get_research_job(self, job_id: int) -> dict[str, Any] | None:
        async with AsyncSession(self._engine) as session:
            row = await session.get(ResearchJob, job_id)
            if row is None:
                return None
            return row.model_dump()

    async def list_research_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = select(ResearchJob).order_by(col(ResearchJob.id).desc()).limit(limit)
            results = await session.exec(statement)
            return [r.model_dump() for r in results.all()]

    async def pin_research_job(self, job_id: int, pinned: bool) -> bool:
        async with AsyncSession(self._engine) as session:
            row = await session.get(ResearchJob, job_id)
            if row is None:
                return False
            row.pinned = pinned
            session.add(row)
            await session.commit()
            return True
