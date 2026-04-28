"""Bit-perfect cycle replay.

To debug a bad trade or honestly score a prompt change, you need to be
able to recreate the *exact* inputs the bot saw on that cycle and re-run
its decision pipeline against them. Without that, "this prompt is
better" reduces to handwaving over noisy live results.

This module is the seam:

* :class:`CycleSnapshot` — the input bundle (klines, account, sentiment,
  positions, indicators, regime, etc.) for one cycle. Pure data, JSON
  serializable, versioned by ``schema_version``.
* :class:`ReplayStore` — write/read cycle snapshots from a Postgres
  ``replay_snapshots`` table. The full snapshot lives in a JSONB
  ``payload`` column; the dataclass below is the schema authority.
* :func:`replay_cycle` — given a cycle_id and a "decision callable", load
  the snapshot, call the callable on the inputs, and return its plan.

The cycle code captures snapshots once via :func:`record_snapshot` (no
behavior change to the cycle besides one INSERT per cycle). The replay
harness reconstructs ``CycleSnapshot`` objects exactly — the LLM output
won't be bit-identical because LLMs aren't deterministic, but
*everything the LLM saw* is.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlmodel import col, select

from halal_trader.db.models import ReplaySnapshotRow
from halal_trader.domain.models import Kline

logger = logging.getLogger(__name__)


SCHEMA_VERSION = 1


# ── Snapshot ──────────────────────────────────────────────────────


@dataclass
class CycleSnapshot:
    """Frozen inputs of one cycle. Pass back through ``replay_cycle``."""

    cycle_id: str
    cycle_started_at: str  # ISO timestamp
    market: str = "crypto"  # "crypto" | "stocks"
    schema_version: int = SCHEMA_VERSION

    halal_pairs: list[str] = field(default_factory=list)
    klines_by_symbol: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    indicators_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    orderbooks: dict[str, dict[str, Any]] = field(default_factory=dict)

    account: dict[str, Any] = field(default_factory=dict)
    positions_text: str = ""
    open_position_count: int = 0
    today_pnl: float = 0.0

    sentiment_text: str = ""
    timeframe_text: str = ""
    ml_signals_text: str = ""
    regime_text: str = ""
    risk_text: str = ""
    microstructure_text: str = ""
    news_text: str = ""
    performance_text: str = ""
    exchange_rules_text: str = ""
    active_adjustments: str = ""

    extra: dict[str, Any] = field(default_factory=dict)

    # ── helpers ──────────────────────────────────────────────────

    @classmethod
    def from_inputs(
        cls,
        *,
        cycle_id: str,
        klines_by_symbol: dict[str, list[Kline]] | None = None,
        **kwargs: Any,
    ) -> "CycleSnapshot":
        """Build from native inputs — flattens Pydantic ``Kline`` objects."""
        flat: dict[str, list[dict[str, Any]]] = {}
        for sym, ks in (klines_by_symbol or {}).items():
            flat[sym] = [_kline_to_dict(k) for k in ks]
        return cls(
            cycle_id=cycle_id,
            cycle_started_at=datetime.now(UTC).isoformat(),
            klines_by_symbol=flat,
            **kwargs,
        )

    def klines_native(self) -> dict[str, list[Kline]]:
        """Materialise the flat klines back into ``Kline`` objects."""
        out: dict[str, list[Kline]] = {}
        for sym, rows in self.klines_by_symbol.items():
            out[sym] = [Kline(**r) for r in rows]
        return out


def _kline_to_dict(k: Kline | dict[str, Any]) -> dict[str, Any]:
    if isinstance(k, dict):
        return dict(k)
    if hasattr(k, "model_dump"):
        return k.model_dump()
    if hasattr(k, "dict"):
        return k.dict()  # pydantic v1 fallback
    return asdict(k)


# ── DB-backed store ───────────────────────────────────────────────


@dataclass
class ReplayStore:
    """Postgres-backed snapshot store keyed by ``cycle_id``."""

    engine: AsyncEngine

    @property
    def _sm(self) -> "async_sessionmaker[Any]":
        return async_sessionmaker(self.engine, expire_on_commit=False)

    async def write(self, snapshot: CycleSnapshot) -> None:
        """Upsert by cycle_id — replays of the same cycle overwrite."""
        payload = asdict(snapshot)
        async with self._sm() as s:
            existing = await s.get(ReplaySnapshotRow, snapshot.cycle_id)
            if existing is None:
                s.add(
                    ReplaySnapshotRow(
                        cycle_id=snapshot.cycle_id,
                        market=snapshot.market,
                        schema_version=snapshot.schema_version,
                        payload=payload,
                    )
                )
            else:
                existing.market = snapshot.market
                existing.schema_version = snapshot.schema_version
                existing.payload = payload
                s.add(existing)
            await s.commit()

    async def read(self, cycle_id: str) -> CycleSnapshot:
        async with self._sm() as s:
            row = await s.get(ReplaySnapshotRow, cycle_id)
            if row is None:
                raise KeyError(f"no replay snapshot for cycle_id {cycle_id!r}")
            version = int(row.schema_version)
            if version > SCHEMA_VERSION:
                raise ValueError(
                    f"snapshot {cycle_id} schema_version {version} > supported {SCHEMA_VERSION}"
                )
            return CycleSnapshot(**row.payload)

    async def list_cycle_ids(self, limit: int = 50) -> list[str]:
        """Most-recent ``limit`` cycles by ``created_at`` desc."""
        async with self._sm() as s:
            stmt = (
                select(ReplaySnapshotRow.cycle_id)
                .order_by(col(ReplaySnapshotRow.created_at).desc())
                .limit(limit)
            )
            result = await s.execute(stmt)
            return [row[0] for row in result.all()]


# ── Recording / replay ────────────────────────────────────────────


async def record_snapshot(store: ReplayStore, snapshot: CycleSnapshot) -> None:
    """Persist a snapshot — failure logged, never raised to the cycle."""
    try:
        await store.write(snapshot)
    except Exception as exc:  # noqa: BLE001
        logger.warning("replay snapshot write failed (%s): %s", snapshot.cycle_id, exc)


async def replay_cycle(
    store: ReplayStore,
    cycle_id: str,
    decide: Callable[[CycleSnapshot], Awaitable[Any]],
) -> Any:
    """Load the snapshot, hand it to ``decide``, return the result.

    The harness is intentionally agnostic about *what* ``decide`` does —
    it can be:
      * a real strategy instance (re-runs the LLM call),
      * a stub that just inspects the inputs (debugging),
      * the adversarial-co-bot critic alone,
      * the prompt-evolution GA fitness function.
    """
    snap = await store.read(cycle_id)
    return await decide(snap)


def diff_snapshots(a: CycleSnapshot, b: CycleSnapshot) -> dict[str, Any]:
    """Shallow diff of two snapshots — useful when reproducing bugs.

    Reports which top-level fields differ (and a count of klines that
    differ per symbol). Not a deep semantic diff; intentionally cheap.
    """
    diff: dict[str, Any] = {}
    a_d = asdict(a)
    b_d = asdict(b)
    for key in set(a_d) | set(b_d):
        if a_d.get(key) != b_d.get(key):
            if key == "klines_by_symbol":
                ksum: dict[str, int] = {}
                for sym in set(a.klines_by_symbol) | set(b.klines_by_symbol):
                    a_ks = a.klines_by_symbol.get(sym, [])
                    b_ks = b.klines_by_symbol.get(sym, [])
                    if a_ks != b_ks:
                        ksum[sym] = max(len(a_ks), len(b_ks))
                diff[key] = ksum
            else:
                diff[key] = {"a": a_d.get(key), "b": b_d.get(key)}
    return diff
