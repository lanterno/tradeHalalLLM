"""Database-backed RAG store — same interface as ``RationaleStore``.

Lives next to the JSON-backed :class:`RationaleStore` so the cycle can
opt into either one. Production wires the DB variant via
``CryptoComponents.build_components``; tests use the JSON variant
because they bypass the engine entirely.

Storage shape
=============
Each rationale lands as one ``rag_rationales`` row with the embedding
serialised as a JSON list[float] (portable across SQLite-test and
Postgres-prod). Switching to a ``vector(512)`` column with an HNSW
index is one alembic migration away — the public methods here don't
change.

API mirrors :class:`RationaleStore` but is fully async, since every
call site already runs inside an async context.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlmodel import select

from halal_trader.core.llm.rag import (
    Embedder,
    HashingEmbedder,
    cosine,
)
from halal_trader.core.llm.rag import (
    RationaleRow as RationaleRowDC,
)
from halal_trader.db.models import RationaleRow

logger = logging.getLogger(__name__)


@dataclass
class DBRationaleStore:
    """Async DB-backed rationale store.

    Same retrieval semantics as :class:`RationaleStore`: linear-scan
    cosine over JSON-encoded vectors. Linear scan is fine up to a few
    thousand rows; past that, swap the body of :meth:`query` for a
    pgvector index without touching the public API.
    """

    engine: AsyncEngine
    embedder: Embedder = field(default_factory=lambda: HashingEmbedder())
    capacity: int = 10_000

    @property
    def _sm(self) -> "async_sessionmaker[Any]":
        return async_sessionmaker(self.engine, expire_on_commit=False)

    async def size(self) -> int:
        from sqlalchemy import func

        async with self._sm() as s:
            r = await s.execute(select(func.count()).select_from(RationaleRow))
            return int(r.scalar_one())

    async def add(
        self,
        *,
        trade_id: str,
        symbol: str,
        text: str,
        outcome_pnl_pct: float,
        setup_type: str | None = None,
        timestamp: str = "",
        tags: Iterable[str] | None = None,
    ) -> RationaleRowDC:
        """Embed + persist one rationale. Idempotent on ``trade_id``."""
        async with self._sm() as s:
            existing = await s.get(RationaleRow, trade_id)
            if existing is not None:
                return _row_to_dc(existing)

            vec = self.embedder.embed(text)
            row = RationaleRow(
                trade_id=trade_id,
                symbol=symbol,
                text=text,
                vector=json.dumps(vec),
                outcome_pnl_pct=outcome_pnl_pct,
                outcome_win=outcome_pnl_pct > 0,
                setup_type=setup_type,
                timestamp=_parse_timestamp(timestamp),
            )
            s.add(row)
            await s.commit()

        await self._enforce_capacity()
        return _row_to_dc(row)

    async def _enforce_capacity(self) -> None:
        from sqlalchemy import delete, func

        async with self._sm() as s:
            count_r = await s.execute(select(func.count()).select_from(RationaleRow))
            count = int(count_r.scalar_one())
            if count <= self.capacity:
                return
            # Drop the oldest rows down to capacity.
            to_drop = count - self.capacity
            from sqlalchemy import inspect as sa_inspect

            ts_col = sa_inspect(RationaleRow).columns["timestamp"]
            cutoff_q = select(ts_col).order_by(ts_col).limit(to_drop).offset(to_drop - 1)
            cutoff = (await s.execute(cutoff_q)).scalar_one_or_none()
            if cutoff is None:
                return
            await s.execute(delete(RationaleRow).where(ts_col <= cutoff))
            await s.commit()

    async def query(
        self,
        text: str,
        *,
        k: int = 5,
        min_similarity: float = 0.1,
        symbol: str | None = None,
    ) -> list[tuple[RationaleRowDC, float]]:
        if not text:
            return []
        q = self.embedder.embed(text)
        async with self._sm() as s:
            stmt = select(RationaleRow)
            if symbol is not None:
                stmt = stmt.where(RationaleRow.symbol == symbol)
            rows = (await s.execute(stmt)).scalars().all()
        scored: list[tuple[RationaleRowDC, float]] = []
        for r in rows:
            try:
                vec = json.loads(r.vector)
            except (TypeError, ValueError) as _exc:  # noqa: F841 — keep parens, ruff format strips them otherwise
                continue
            score = cosine(q, vec)
            if score >= min_similarity:
                scored.append((_row_to_dc(r), score))
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:k]

    async def aggregate(self, hits: Iterable[tuple[RationaleRowDC, float]]) -> dict[str, Any]:
        """Mirror of ``RationaleStore.aggregate``."""
        hits = list(hits)
        if not hits:
            return {"n": 0, "weighted_pnl_pct": 0.0, "weighted_win_rate": 0.0}
        total_w = 0.0
        wp = 0.0
        ww = 0.0
        for row, sim in hits:
            w = max(0.0, sim)
            total_w += w
            wp += w * row.outcome_pnl_pct
            ww += w * (1.0 if row.outcome_win else 0.0)
        if total_w == 0:
            return {"n": len(hits), "weighted_pnl_pct": 0.0, "weighted_win_rate": 0.0}
        return {
            "n": len(hits),
            "weighted_pnl_pct": wp / total_w,
            "weighted_win_rate": ww / total_w,
        }


# ── Helpers ──────────────────────────────────────────────────────


def _row_to_dc(row: RationaleRow) -> RationaleRowDC:
    """SQLModel row → public dataclass shape."""
    try:
        vec = json.loads(row.vector)
    except (TypeError, ValueError) as _exc:  # noqa: F841 — keep parens, ruff format strips them otherwise
        vec = []
    return RationaleRowDC(
        trade_id=row.trade_id,
        symbol=row.symbol,
        text=row.text,
        vector=vec,
        outcome_pnl_pct=row.outcome_pnl_pct,
        outcome_win=row.outcome_win,
        setup_type=row.setup_type,
        timestamp=row.timestamp.isoformat() if row.timestamp else "",
        tags=[],
    )


def _parse_timestamp(raw: str) -> datetime:
    if not raw:
        return datetime.now(UTC)
    try:
        ts = datetime.fromisoformat(raw)
    except ValueError:
        return datetime.now(UTC)
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)
