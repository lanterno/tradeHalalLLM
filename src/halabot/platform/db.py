"""Engine + schema for the new engine's own tables.

The re-architecture tables (``hb_event_log``, ``hb_belief_state``) live in their
OWN SQLAlchemy ``MetaData`` — deliberately NOT in ``halal_trader``'s
``SQLModel.metadata`` and NOT in its Alembic chain. Two reasons:

* ``init_db`` derives its expected-table set from ``SQLModel.metadata`` and its
  revision check from ``alembic_version``; keeping these tables out of both
  means standing them up never trips the legacy bot's startup check (the live
  bot, on ``main``, is unaffected — fix R-07 hazard avoided).
* During migration the two systems share ONE Postgres. Isolated metadata +
  ``hb_`` prefix guarantees no collision with legacy tables.

``bootstrap_schema`` is additive and idempotent (``create_all`` with
``checkfirst``). The formal Alembic migration that folds these into the single
chain is the deliberate Phase-4 cutover step (REARCHITECTURE Part IV), with the
downgrade scripts the runbook requires.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

metadata = MetaData()

event_log = Table(
    "hb_event_log",
    metadata,
    Column("id", PgUUID(as_uuid=True), primary_key=True),
    Column("type", Text, nullable=False),
    Column("asset", Text, nullable=True),
    Column("ts", DateTime(timezone=True), nullable=False),
    Column("ingested_at", DateTime(timezone=True), nullable=False),
    Column("source", Text, nullable=False),
    Column("payload", JSONB, nullable=False),
    Column("causation_id", PgUUID(as_uuid=True), nullable=True),
    Column("correlation_id", PgUUID(as_uuid=True), nullable=False),
    Column("schema_version", Integer, nullable=False),
)

belief_state = Table(
    "hb_belief_state",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("asset", Text, nullable=False),
    Column("version", Integer, nullable=False),
    Column("regime", Text, nullable=False),
    Column("regime_confidence", Float, nullable=False),
    Column("direction", Text, nullable=False),
    Column("conviction", Float, nullable=False),
    Column("conviction_raw", Float, nullable=False),
    Column("horizon", Text, nullable=False),
    Column("thesis", Text, nullable=True),
    Column("levels", JSONB, nullable=False),
    Column("catalysts", JSONB, nullable=False),
    Column("evidence", JSONB, nullable=False),
    Column("halal_verdict", JSONB, nullable=True),
    Column("opened_trade_ids", JSONB, nullable=False),
    Column("last_thesis_refresh", DateTime(timezone=True), nullable=True),
    Column("last_updated", DateTime(timezone=True), nullable=True),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("asset", "version", name="uq_hb_belief_asset_version"),
)

# Hypothetical (shadow) outcomes — one row per closed/reduced shadow position,
# marked to price. Feeds the A/B P&L comparison and the conviction calibrator
# (entry_belief is the at-entry snapshot — no mid-trade leakage).
outcome = Table(
    "hb_outcome",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("asset", Text, nullable=False),
    Column("entry_ts", DateTime(timezone=True), nullable=False),
    Column("exit_ts", DateTime(timezone=True), nullable=False),
    Column("entry_price", Float, nullable=False),
    Column("exit_price", Float, nullable=False),
    Column("closed_weight", Float, nullable=False),
    Column("return_pct", Float, nullable=False),
    Column("hold_seconds", Integer, nullable=False),
    Column("belief_version", Integer, nullable=False),
    Column("entry_belief", JSONB, nullable=True),
    Column("label", Integer, nullable=False),  # win=1 if return_pct > threshold else 0
    Column("reason", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

# Replay + lookup indexes (created by create_all alongside the tables).
Index("ix_hb_event_type_ts", event_log.c.type, event_log.c.ts)
Index("ix_hb_event_asset_ts", event_log.c.asset, event_log.c.ts)
Index("ix_hb_event_corr", event_log.c.correlation_id)
Index("ix_hb_belief_asset_version", belief_state.c.asset, belief_state.c.version.desc())
Index("ix_hb_outcome_asset_ts", outcome.c.asset, outcome.c.exit_ts)


async def bootstrap_schema(engine: AsyncEngine) -> None:
    """Create the engine's tables if missing (additive, idempotent).

    Safe to call against the shared DB: it touches only ``hb_*`` tables and
    never ``alembic_version``, so the legacy bot's head check is unaffected.
    """
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)


def make_engine(database_url: str) -> AsyncEngine:
    """Async engine over the (shared) Postgres URL — same DB as the legacy bot."""
    return create_async_engine(database_url)


__all__ = [
    "metadata",
    "event_log",
    "belief_state",
    "outcome",
    "bootstrap_schema",
    "make_engine",
]
