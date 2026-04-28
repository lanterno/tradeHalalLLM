"""DB-backed regret recorder — append-only over ``regret_records``."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlmodel import select

from halal_trader.db.models import RegretRecordRow

logger = logging.getLogger(__name__)


@dataclass
class DBRegretRecorder:
    """Persistent regret record store.

    Rows live in ``regret_records`` so aggregate queries (mean, p99,
    by symbol / setup_type) run as proper SQL.
    """

    engine: AsyncEngine

    @property
    def _sm(self) -> "async_sessionmaker[Any]":
        return async_sessionmaker(self.engine, expire_on_commit=False)

    async def append(self, record: dict[str, Any]) -> None:
        """Idempotent on ``trade_id`` — replays don't duplicate."""
        async with self._sm() as s:
            existing = await s.get(RegretRecordRow, str(record["trade_id"]))
            if existing is not None:
                return
            ts_raw = record.get("ts", "")
            ts = _parse_ts(ts_raw)
            s.add(
                RegretRecordRow(
                    trade_id=str(record["trade_id"]),
                    symbol=str(record.get("symbol", "")),
                    regret=float(record.get("regret", 0.0)),
                    optimal_size_pct=float(record.get("optimal_size_pct", 0.0)),
                    actual_size_pct=float(record.get("actual_size_pct", 0.0)),
                    pnl_pct=float(record.get("pnl_pct", 0.0)),
                    note=str(record.get("note", "")),
                    setup_type=record.get("setup_type"),
                    closed_at=ts,
                )
            )
            await s.commit()

    async def all(self) -> list[dict[str, Any]]:
        async with self._sm() as s:
            rows = (await s.execute(select(RegretRecordRow))).scalars().all()
            return [
                {
                    "trade_id": r.trade_id,
                    "symbol": r.symbol,
                    "regret": r.regret,
                    "optimal_size_pct": r.optimal_size_pct,
                    "actual_size_pct": r.actual_size_pct,
                    "pnl_pct": r.pnl_pct,
                    "note": r.note,
                    "setup_type": r.setup_type,
                    "ts": r.closed_at.isoformat() if r.closed_at else "",
                }
                for r in rows
            ]


def _parse_ts(raw: str) -> datetime:
    if not raw:
        return datetime.now(UTC)
    try:
        ts = datetime.fromisoformat(raw)
    except ValueError:
        return datetime.now(UTC)
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)
