"""Database-backed RAG store over `rag_rationales`.

Storage shape
=============
Each rationale lands as one ``rag_rationales`` row with a native
pgvector ``embedding`` column (vector(512)). Cosine similarity
queries run as ``ORDER BY embedding <=> :q LIMIT k`` against an
HNSW index — ~1ms even at 100k rows.

API is fully async since every call site already runs inside an
async context.
"""

from __future__ import annotations

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
)
from halal_trader.core.llm.rag import (
    RationaleRow as RationaleRowDC,
)
from halal_trader.db.models import RationaleRow

logger = logging.getLogger(__name__)


@dataclass
class DBRationaleStore:
    """Async DB-backed rationale store with pgvector similarity."""

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
                embedding=list(vec),
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
        """Top-k cosine similarity match via pgvector + HNSW.

        ``embedding <=> :q`` returns cosine *distance* (1 - similarity);
        we convert back to similarity for the public API and apply the
        ``min_similarity`` filter in Python because pgvector's HNSW
        operator can't filter on a derived expression.
        """
        if not text:
            return []
        q = list(self.embedder.embed(text))
        distance = RationaleRow.embedding.cosine_distance(q)  # type: ignore[attr-defined]
        async with self._sm() as s:
            stmt = select(RationaleRow, distance.label("distance"))
            if symbol is not None:
                stmt = stmt.where(RationaleRow.symbol == symbol)
            stmt = stmt.order_by(distance).limit(max(k * 2, k))
            result = await s.execute(stmt)
            rows = result.all()
        scored: list[tuple[RationaleRowDC, float]] = []
        for row, dist in rows:
            sim = 1.0 - float(dist)
            if sim >= min_similarity:
                scored.append((_row_to_dc(row), sim))
            if len(scored) >= k:
                break
        return scored

    async def aggregate(self, hits: Iterable[tuple[RationaleRowDC, float]]) -> dict[str, Any]:
        """Similarity-weighted stats over a query result."""
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
    embedding = row.embedding
    if embedding is None:
        vec: list[float] = []
    else:
        vec = list(embedding)
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
