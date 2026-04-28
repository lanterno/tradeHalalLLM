"""Embedding-based regime memory.

Once you have a few months of trades you have something most retail bots
don't: a labelled history of how *your* strategy behaved in each regime.
The cheap way to exploit it is per-cycle retrieval — compute a small
feature vector summarising "what kind of day is this?", look up the K
most similar past days, and surface their P&L in the prompt.

This module keeps the abstraction tiny:

* :class:`RegimeFeatures` — the day's snapshot vector (volatility, trend,
  breadth, sentiment, drawdown, etc.). Hand-engineered, not learned —
  the goal is interpretability and stability, not raw accuracy.
* :class:`RegimeMemory` — DB-backed store over ``regime_snapshots``;
  ``add()`` appends a snapshot + outcome, ``query()`` returns top-K
  cosine matches.
* :func:`format_for_prompt` — render a query result into the plain
  text block the LLM reads.

No external embedding model. Cosine over the standardised feature vector
is enough to find genuinely similar days, and the feature vector is
already what the bot computes per cycle anyway.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlmodel import col, select

from halal_trader.db.models import RegimeSnapshotRow

logger = logging.getLogger(__name__)


# ── Feature vector ────────────────────────────────────────────────


@dataclass
class RegimeFeatures:
    """Daily market-state snapshot used as the embedding input.

    All fields default to 0; missing inputs degrade gracefully (the
    cosine similarity just treats absent dimensions as neutral). Keep
    the shape stable — adding a field invalidates older snapshots.
    """

    volatility: float = 0.0  # ATR / price (decimal)
    trend: float = 0.0  # -1..+1 multi-tf alignment
    breadth: float = 0.0  # -1..+1 share of universe up
    sentiment: float = 0.0  # -1..+1 composite news/social
    drawdown: float = 0.0  # 0..1 from peak
    volume_ratio: float = 1.0  # current / 20d avg
    correlation: float = 0.0  # 0..1 portfolio internal corr
    realized_return_1d: float = 0.0  # past day's market return
    rsi: float = 50.0  # 0..100
    spread_bps: float = 0.0  # microstructure cost proxy

    def to_vector(self) -> list[float]:
        return [
            self.volatility * 50.0,  # scale ATR/price (~0.02) to ~1
            self.trend,
            self.breadth,
            self.sentiment,
            self.drawdown * 5.0,  # 0..1 -> 0..5
            self.volume_ratio - 1.0,  # center
            self.correlation,
            self.realized_return_1d * 50.0,  # ~5% ⇒ ~2.5
            (self.rsi - 50.0) / 50.0,  # -1..+1
            self.spread_bps / 10.0,
        ]


# ── Snapshot record ───────────────────────────────────────────────


@dataclass
class RegimeSnapshot:
    """One day's regime + downstream outcome."""

    date: str  # ISO date "YYYY-MM-DD"
    features: RegimeFeatures
    outcome_pnl_pct: float = 0.0
    outcome_win_rate: float = 0.0
    outcome_n_trades: int = 0
    note: str = ""

    def label(self) -> str:
        sign = "+" if self.outcome_pnl_pct >= 0 else ""
        return (
            f"{self.date}: P&L {sign}{self.outcome_pnl_pct:.2%}, "
            f"win {self.outcome_win_rate:.0%} on {self.outcome_n_trades} trades"
        )


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def _row_to_snapshot(row: RegimeSnapshotRow) -> RegimeSnapshot:
    features = RegimeFeatures(**row.features_json)
    return RegimeSnapshot(
        date=row.date,
        features=features,
        outcome_pnl_pct=row.outcome_pnl_pct,
        outcome_win_rate=row.outcome_win_rate,
        outcome_n_trades=row.outcome_n_trades,
        note=row.note,
    )


# ── DB-backed memory store ────────────────────────────────────────


@dataclass
class RegimeMemory:
    """Postgres-backed daily regime store.

    Queries are O(N) over the in-memory rowset; N stays small (months
    of trading days, capped at ``capacity``), so explicit cosine in
    Python is cheaper than round-tripping a vector op to the DB.
    pgvector adoption is a one-line migration when N grows past ~10k.
    """

    engine: AsyncEngine
    capacity: int = 730  # ~2 trading years

    @property
    def _sm(self) -> "async_sessionmaker[Any]":
        return async_sessionmaker(self.engine, expire_on_commit=False)

    async def size(self) -> int:
        async with self._sm() as s:
            from sqlalchemy import func

            result = await s.execute(select(func.count()).select_from(RegimeSnapshotRow))
            return int(result.scalar_one())

    async def add(self, snapshot: RegimeSnapshot) -> None:
        """Upsert by date — same date overwrites."""
        features_json = asdict(snapshot.features)
        vector_json = snapshot.features.to_vector()
        async with self._sm() as s:
            existing = await s.get(RegimeSnapshotRow, snapshot.date)
            if existing is None:
                s.add(
                    RegimeSnapshotRow(
                        date=snapshot.date,
                        features_json=features_json,
                        vector_json=vector_json,
                        outcome_pnl_pct=snapshot.outcome_pnl_pct,
                        outcome_win_rate=snapshot.outcome_win_rate,
                        outcome_n_trades=snapshot.outcome_n_trades,
                        note=snapshot.note,
                    )
                )
            else:
                existing.features_json = features_json
                existing.vector_json = vector_json
                existing.outcome_pnl_pct = snapshot.outcome_pnl_pct
                existing.outcome_win_rate = snapshot.outcome_win_rate
                existing.outcome_n_trades = snapshot.outcome_n_trades
                existing.note = snapshot.note
                s.add(existing)
            await s.commit()
        await self._enforce_capacity()

    async def add_today(
        self,
        features: RegimeFeatures,
        *,
        today: str | date | None = None,
        outcome_pnl_pct: float = 0.0,
        outcome_win_rate: float = 0.0,
        outcome_n_trades: int = 0,
        note: str = "",
    ) -> RegimeSnapshot:
        if today is None:
            today = date.today()
        date_str = today.isoformat() if isinstance(today, date) else str(today)
        snap = RegimeSnapshot(
            date=date_str,
            features=features,
            outcome_pnl_pct=outcome_pnl_pct,
            outcome_win_rate=outcome_win_rate,
            outcome_n_trades=outcome_n_trades,
            note=note,
        )
        await self.add(snap)
        return snap

    async def query(
        self,
        features: RegimeFeatures,
        *,
        k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[tuple[RegimeSnapshot, float]]:
        """Top-K similar snapshots by cosine of feature vectors."""
        q = features.to_vector()
        async with self._sm() as s:
            result = await s.execute(select(RegimeSnapshotRow))
            rows = result.scalars().all()
        if not rows:
            return []
        scored: list[tuple[RegimeSnapshot, float]] = []
        for row in rows:
            vec = row.vector_json or []
            scored.append((_row_to_snapshot(row), _cosine(q, vec)))
        scored.sort(key=lambda p: p[1], reverse=True)
        out = [p for p in scored if p[1] >= min_similarity]
        return out[:k]

    async def recent(self, limit: int = 10) -> list[RegimeSnapshot]:
        """Most recently inserted snapshots (for the dashboard)."""
        async with self._sm() as s:
            stmt = (
                select(RegimeSnapshotRow)
                .order_by(col(RegimeSnapshotRow.created_at).desc())
                .limit(limit)
            )
            result = await s.execute(stmt)
            rows = result.scalars().all()
        return [_row_to_snapshot(r) for r in rows]

    async def _enforce_capacity(self) -> None:
        """Trim the oldest rows if we're past capacity."""
        if self.capacity <= 0:
            return
        async with self._sm() as s:
            from sqlalchemy import func

            count_r = await s.execute(select(func.count()).select_from(RegimeSnapshotRow))
            total = int(count_r.scalar_one())
            if total <= self.capacity:
                return
            excess = total - self.capacity
            stmt = (
                select(RegimeSnapshotRow.date)
                .order_by(col(RegimeSnapshotRow.created_at).asc())
                .limit(excess)
            )
            result = await s.execute(stmt)
            stale_dates = [row[0] for row in result.all()]
            if not stale_dates:
                return
            from sqlalchemy import delete

            await s.execute(
                delete(RegimeSnapshotRow).where(col(RegimeSnapshotRow.date).in_(stale_dates))
            )
            await s.commit()

    @staticmethod
    def aggregate_outcome(hits: Iterable[tuple[RegimeSnapshot, float]]) -> dict[str, float]:
        """Similarity-weighted outcome aggregate from a query result."""
        hits = list(hits)
        if not hits:
            return {"weighted_pnl_pct": 0.0, "weighted_win_rate": 0.0, "n": 0}
        total_w = 0.0
        weighted_pnl = 0.0
        weighted_win = 0.0
        for s, sim in hits:
            w = max(0.0, sim)
            total_w += w
            weighted_pnl += w * s.outcome_pnl_pct
            weighted_win += w * s.outcome_win_rate
        if total_w == 0:
            return {"weighted_pnl_pct": 0.0, "weighted_win_rate": 0.0, "n": len(hits)}
        return {
            "weighted_pnl_pct": weighted_pnl / total_w,
            "weighted_win_rate": weighted_win / total_w,
            "n": len(hits),
        }


# ── Prompt rendering ──────────────────────────────────────────────


def format_for_prompt(
    today: RegimeFeatures,
    hits: Sequence[tuple[RegimeSnapshot, float]],
    *,
    max_lines: int = 5,
) -> str:
    """Render the top-K analogous days as a compact prompt block.

    Empty result returns a stable placeholder string the prompt
    builder can recognise and elide.
    """
    if not hits:
        return "No analogous past regime data."
    lines = [
        "Top historical regimes most similar to today",
        f"  (today: vol={today.volatility:.4f}, trend={today.trend:+.2f}, "
        f"sent={today.sentiment:+.2f}, dd={today.drawdown:.2%})",
    ]
    for snap, sim in hits[:max_lines]:
        lines.append(f"  · sim={sim:+.2f} — {snap.label()}")
        if snap.note:
            lines.append(f"      note: {snap.note[:100]}")
    return "\n".join(lines)
