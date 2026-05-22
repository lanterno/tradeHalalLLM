"""SQLModel table definitions and database initialization."""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import Field, SQLModel

# Embedding dimensions are pinned constants; if they change the
# corresponding pgvector column needs an alembic migration that
# REINDEXes the HNSW index.
RAG_EMBEDDING_DIM = 512
REGIME_EMBEDDING_DIM = 10

logger = logging.getLogger(__name__)


# ── Stock Tables ────────────────────────────────────────────────


class Trade(SQLModel, table=True):
    """Record of a single trade (buy or sell)."""

    __tablename__ = "trades"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float | None = None
    order_id: str | None = None
    status: str = Field(default="pending")
    llm_reasoning: str | None = None

    # Fill confirmation (populated by FillConfirmer after place_order).
    submitted_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True))
    filled_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True))
    filled_price: float | None = None
    filled_quantity: float | None = None

    # Halal audit FK — links to the screening decision that gated this trade.
    halal_screening_id: int | None = Field(default=None, foreign_key="halal_screenings.id")

    # SL/TP + close lifecycle — mirrors CryptoTrade so the shared
    # monitor/analytics surface can treat both asset classes uniformly.
    stop_loss: float | None = None
    target_price: float | None = None

    # How the trade entered the book. Used by the slow-out discipline
    # (memory: strategy-fast-in-slow-out): "reactor_momentum" trades
    # are LLM-untouchable on the SELL side — only the monitor's
    # rule-based exit can close them. None = legacy / "scheduled"
    # (default for cron-cycle entries).
    entry_type: str | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    closed_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True))

    # Paper-vs-live divergence — both sides record realized slippage
    # (signed, in fraction of price) so the operator can sanity-check
    # the backtester's assumptions against live fills.
    paper_slippage_pct: float | None = None
    live_slippage_pct: float | None = None
    # Wave G: replay-fitted slippage prediction stamped at fill time.
    # See CryptoTrade.predicted_slippage_pct.
    predicted_slippage_pct: float | None = None


class DailyPnl(SQLModel, table=True):
    """Daily profit-and-loss snapshot."""

    __tablename__ = "daily_pnl"

    id: int | None = Field(default=None, primary_key=True)
    date: str = Field(unique=True)
    starting_equity: float
    ending_equity: float | None = None
    realized_pnl: float = Field(default=0)
    return_pct: float | None = None
    trades_count: int = Field(default=0)


class HalalCache(SQLModel, table=True):
    """Cached Shariah-compliance status for a stock symbol."""

    __tablename__ = "halal_cache"

    symbol: str = Field(primary_key=True)
    compliance: str  # 'halal', 'not_halal', 'doubtful'
    detail: str | None = None
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )


class LlmDecision(SQLModel, table=True):
    """Audit log entry for an LLM trading decision."""

    __tablename__ = "llm_decisions"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )
    provider: str
    model: str
    prompt_summary: str | None = None
    raw_response: str | None = None
    parsed_action: dict | None = Field(
        default=None, sa_column=sa.Column("parsed_action", JSONB, nullable=True)
    )
    symbols: list | None = Field(default=None, sa_column=sa.Column("symbols", JSONB, nullable=True))
    execution_ms: int | None = None
    thinking: str | None = None  # reasoning chain from thinking-mode LLMs

    # Cost / cache attribution. Lets us cap daily spend, measure cache
    # hit rate, and replay any decision against its exact prompt version.
    prompt_version: str | None = None  # registry "name@hash", e.g. "crypto.strategy.system@abc123"
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost_usd: float | None = None  # rounded float — Decimal aggregation done in code

    # Wave H — agentic mode persists each tool call's name/args/result
    # so the dashboard can render a tree of the model's chain-of-thought.
    # None when the cycle ran in single-prompt mode.
    tool_transcript: list | None = Field(
        default=None,
        sa_column=sa.Column("tool_transcript", JSONB, nullable=True),
    )


# ── Crypto Tables ──────────────────────────────────────────────


class CryptoTrade(SQLModel, table=True):
    """Record of a single crypto trade."""

    __tablename__ = "crypto_trades"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )
    pair: str  # e.g. 'BTCUSDT'
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float | None = None
    order_id: str | None = None
    exchange: str = Field(default="binance")
    status: str = Field(default="pending")
    llm_reasoning: str | None = None

    entry_price: float | None = None
    stop_loss: float | None = None
    target_price: float | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    closed_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True))

    # Fill confirmation (populated by FillConfirmer after place_order).
    submitted_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True))
    filled_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True))
    filled_price: float | None = None
    filled_quantity: float | None = None

    # Halal audit FK — links to the screening decision that gated this trade.
    halal_screening_id: int | None = Field(default=None, foreign_key="halal_screenings.id")

    # Paper-vs-live divergence — same fields as Trade.
    paper_slippage_pct: float | None = None
    live_slippage_pct: float | None = None
    # Wave G: replay-fitted slippage prediction stamped at fill time.
    # The backtester reads the same model so backtest and live converge;
    # this column lets us score the model's calibration after the fact
    # (predicted vs realised live_slippage_pct).
    predicted_slippage_pct: float | None = None


class CryptoDailyPnl(SQLModel, table=True):
    """Daily crypto profit-and-loss snapshot."""

    __tablename__ = "crypto_daily_pnl"

    id: int | None = Field(default=None, primary_key=True)
    date: str = Field(unique=True)
    starting_equity: float
    ending_equity: float | None = None
    realized_pnl: float = Field(default=0)
    return_pct: float | None = None
    trades_count: int = Field(default=0)


class CryptoHalalCache(SQLModel, table=True):
    """Cached Shariah-compliance status for a crypto token."""

    __tablename__ = "crypto_halal_cache"

    symbol: str = Field(primary_key=True)  # e.g. 'BTC', 'ETH'
    compliance: str  # 'halal', 'not_halal', 'doubtful'
    category: str | None = None  # e.g. 'layer-1', 'defi', 'meme'
    market_cap: float | None = None
    screening_criteria: dict | None = Field(
        default=None, sa_column=sa.Column("screening_criteria", JSONB, nullable=True)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )


class IndicatorSnapshot(SQLModel, table=True):
    """Indicator feature vector captured at trade entry time for ML training."""

    __tablename__ = "indicator_snapshots"

    id: int | None = Field(default=None, primary_key=True)
    trade_id: int = Field(index=True)
    pair: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )
    rsi_14: float | None = None
    macd_histogram: float | None = None
    volume_ratio: float | None = None
    atr_14: float | None = None
    bb_position: float | None = None
    price_change_5m: float | None = None
    ema_9: float | None = None
    ema_21: float | None = None
    vwap: float | None = None
    label: int | None = None  # 1=profitable, 0=unprofitable (set after close)
    return_pct: float | None = None  # actual return % (set after close)


class StrategyAdjustment(SQLModel, table=True):
    """Audit log for LLM self-improvement parameter changes."""

    __tablename__ = "strategy_adjustments"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )
    parameter: str
    old_value: float | None = None
    new_value: float
    reasoning: str | None = None


class ReconciliationLog(SQLModel, table=True):
    """Append-only log of DB-vs-broker drift events surfaced by the Reconciler."""

    __tablename__ = "reconciliation_log"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )
    market: str  # 'stocks' | 'crypto'
    symbol: str  # asset/ticker affected
    db_quantity: float
    broker_quantity: float
    drift_pct: float  # |db - broker| / max(db, broker, 1e-9)
    drift_usd: float | None = None
    notes: str | None = None


class ResearchJob(SQLModel, table=True):
    """One backtest / walk-forward / Monte Carlo job, queued or completed."""

    __tablename__ = "research_jobs"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )
    kind: str  # 'backtest' | 'walk_forward' | 'monte_carlo'
    name: str | None = None  # operator-supplied label
    params: dict = Field(sa_column=sa.Column("params", JSONB, nullable=False))
    status: str = Field(default="queued")  # 'queued' | 'running' | 'ok' | 'error'
    result: dict | None = Field(default=None, sa_column=sa.Column("result", JSONB, nullable=True))
    error: str | None = None
    finished_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True))
    pinned: bool = Field(default=False)


class RuntimeConfig(SQLModel, table=True):
    """Runtime overlay for ``Settings`` knobs.

    The bot reads these on each cycle as an overlay over the .env-derived
    values, so an operator can tune ``CRYPTO_MAX_POSITION_PCT`` from the
    dashboard and see the effect on the next tick — no restart required.
    Removing a row reverts to the .env value.
    """

    __tablename__ = "runtime_config"

    key: str = Field(primary_key=True)  # uppercase env-var name
    # Raw scalar / list / dict — JSONB so we can round-trip any
    # ``Settings`` value without a parse-on-read step.
    value: Any = Field(sa_column=sa.Column("value", JSONB, nullable=False))
    set_by: str | None = None
    set_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )


class PairPause(SQLModel, table=True):
    """Per-pair operator pause toggle.

    The cycle's tradeable-pair filter excludes any symbol present here.
    One row per paused pair; deleting the row resumes it. Audit fields
    (set_by, set_at, reason) are kept for the activity feed.
    """

    __tablename__ = "pair_pauses"

    pair: str = Field(primary_key=True)
    set_by: str | None = None
    set_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )
    reason: str | None = None


class WebAction(SQLModel, table=True):
    """Audit log row for one dashboard mutation request.

    Written by ``web/audit.py`` *before* the underlying handler runs so
    even a mutation that crashes mid-execution leaves a trace. The
    ``outcome`` column gets updated to "ok"/"error" once the handler
    returns; rows that stay "pending" point at handlers that crashed
    without cleanup.
    """

    __tablename__ = "web_actions"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )
    actor: str  # the request_id ContextVar value, or "anon" if missing
    method: str  # POST | DELETE | PATCH | PUT
    path: str  # e.g. "/api/admin/halt"
    payload: str | None = None  # JSON-serialised request body (truncated)
    outcome: str = Field(default="pending")  # 'pending' | 'ok' | 'error'
    status_code: int | None = None
    error: str | None = None


class PurificationEntry(SQLModel, table=True):
    """Persistent record of a dividend's haram-portion purification obligation.

    One row per received dividend. ``paid_at`` stays NULL until the
    operator records the donation; ``outstanding_total`` queries filter
    on ``paid_at IS NULL``.
    """

    __tablename__ = "purification_entries"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )
    symbol: str = Field(index=True)
    dividend_usd: float
    haram_pct: float
    purification_usd: float
    notes: str | None = None
    paid_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True))


class HalalScreening(SQLModel, table=True):
    """Per-decision audit row for a Shariah-compliance screening.

    Every trade should reference one of these via ``halal_screening_id`` so
    we can prove, after the fact, *why* a position was deemed compliant —
    which source said so, with what criteria, at what time. Cache hits
    record a row too (with ``cache_hit=True``) so the audit trail never
    has gaps even when the underlying provider isn't queried.
    """

    __tablename__ = "halal_screenings"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )
    symbol: str = Field(index=True)
    asset_class: str  # 'stock' | 'crypto'
    source: str  # 'zoya' | 'coingecko_rules' | 'override' | 'cache' | …
    decision: str  # 'halal' | 'not_halal' | 'doubtful'
    criteria: dict | None = Field(
        default=None, sa_column=sa.Column("criteria", JSONB, nullable=True)
    )
    cache_hit: bool = Field(default=False)


class ThesisTagRow(SQLModel, table=True):
    """One thesis tag attached to a closed trade."""

    __tablename__ = "thesis_tags"

    trade_id: str = Field(primary_key=True)
    tag: str  # one of THESIS_TAGS
    confidence: float = 0.0
    reason: str | None = None
    method: str = Field(default="heuristic")  # heuristic | llm
    set_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )


class RoundTripPurificationRow(SQLModel, table=True):
    """One round-trip purification accrual on a closed-and-realised gain.

    Capital-gains side of the purification accounting (the dividend-side
    lives in :class:`PurificationEntry`). Idempotent on
    ``(symbol, source_ref)`` so the close hook can call freely.
    """

    __tablename__ = "round_trip_purification"

    entry_id: str = Field(primary_key=True)
    symbol: str = Field(index=True)
    gain_amount_usd: float
    impure_ratio: float
    purification_due_usd: float
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )
    source_ref: str = ""
    note: str = ""
    disbursed: bool = Field(default=False)
    disbursed_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True))
    disbursed_to: str = ""


class RegretRecordRow(SQLModel, table=True):
    """Hindsight regret record for one closed trade.

    Aggregate queries (mean, p99, by symbol/setup_type) run as proper
    SQL against this table.
    """

    __tablename__ = "regret_records"

    trade_id: str = Field(primary_key=True)
    symbol: str = Field(index=True)
    regret: float
    optimal_size_pct: float
    actual_size_pct: float
    pnl_pct: float
    note: str = ""
    setup_type: str | None = None
    closed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )


class RationaleRow(SQLModel, table=True):
    """RAG store row — one closed-trade rationale + outcome.

    Vector column is JSON-serialised today; a pgvector(512) column
    with an HNSW index is one alembic migration away — the public
    storage API doesn't change.
    """

    __tablename__ = "rag_rationales"

    trade_id: str = Field(primary_key=True)
    symbol: str = Field(index=True)
    text: str
    embedding: list[float] = Field(
        sa_column=sa.Column("embedding", Vector(RAG_EMBEDDING_DIM), nullable=False)
    )
    outcome_pnl_pct: float
    outcome_win: bool
    setup_type: str | None = None
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )


class ReplaySnapshotRow(SQLModel, table=True):
    """One cycle's frozen input bundle.

    The full ``CycleSnapshot`` lives in the ``payload`` JSONB column —
    treating it as opaque keeps schema churn out of the cycle path
    (snapshot fields can come and go via the dataclass). The top-level
    columns are extracted from the snapshot for cheap listing /
    filtering by the dashboard.
    """

    __tablename__ = "replay_snapshots"

    cycle_id: str = Field(primary_key=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=sa.DateTime(timezone=True),
        index=True,
    )
    market: str
    schema_version: int
    payload: dict = Field(sa_column=sa.Column("payload", JSONB, nullable=False))


class ShariaExceptionRow(SQLModel, table=True):
    """One pending Sharia ruling for an ambiguous instrument.

    The screener writes a row when an instrument is doubtful or
    unknown; the operator decides via the dashboard. Keyed by
    ``(instrument, kind)`` (composed into ``entry_id``) so re-screening
    the same pair updates the same row instead of spamming the queue.
    """

    __tablename__ = "sharia_exceptions"

    entry_id: str = Field(primary_key=True)
    instrument: str
    kind: str
    reasoning: str
    status: str = Field(default="pending")  # pending | approved | rejected | deferred
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )
    decided_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True))
    decided_by: str = ""
    operator_note: str = ""


class RegimeSnapshotRow(SQLModel, table=True):
    """One day's regime snapshot (features + outcome).

    Features are JSON-serialised; the embedding vector lives next to
    them so cosine similarity queries don't have to recompute it.
    The pgvector(N) promotion is one alembic migration away — the
    stored JSON list[float] format ports cleanly to a vector column.
    """

    __tablename__ = "regime_snapshots"

    date: str = Field(primary_key=True)
    features_json: dict = Field(sa_column=sa.Column("features_json", JSONB, nullable=False))
    embedding: list[float] = Field(
        sa_column=sa.Column("embedding", Vector(REGIME_EMBEDDING_DIM), nullable=False)
    )
    outcome_pnl_pct: float = 0.0
    outcome_win_rate: float = 0.0
    outcome_n_trades: int = 0
    note: str = ""
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )


class PromptGenome(SQLModel, table=True):
    """One candidate prompt produced by the prompt-evolution GA.

    Each row is a slot→allele mapping (``genome``) with its measured
    fitness over a panel of replay snapshots, plus optional lineage
    pointers for the dashboard's evolution tree view. The dashboard
    can promote a row to live by writing its ``short`` to the
    ``ACTIVE_PROMPT_VERSION`` runtime-config key.
    """

    __tablename__ = "prompt_genomes"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=sa.DateTime(timezone=True),
        index=True,
    )
    name: str = Field(index=True)  # which prompt slot is being evolved
    genome: dict = Field(sa_column=sa.Column("genome", JSONB, nullable=False))
    fitness: float = 0.0
    n_cycles: int = 0  # how many replay snapshots the fitness was measured over
    parent_ids: list = Field(
        default_factory=list,
        sa_column=sa.Column("parent_ids", JSONB, nullable=False, default="[]"),
    )
    promoted_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True))
    notes: str = ""


class MlArtefact(SQLModel, table=True):
    """Versioned ML model blob.

    Wave K replaces ``models/*.pkl`` with this table so the bot's
    state replicates with the DB and rolls back atomically alongside
    the schema. Each row is one (name, version) — the loader picks
    the highest version for a given name; the retrainer inserts a
    new row with version+1.

    The payload stores either a sklearn pickle (BYTEA) or a small
    JSON blob (slippage model, calibration curve), keyed by
    ``payload_format``. HuggingFace caches (~GBs of Chronos
    weights) intentionally stay on the filesystem.
    """

    __tablename__ = "ml_artefacts"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    version: int
    payload_format: str  # "json" | "pickle"
    payload_bytes: bytes | None = Field(
        default=None, sa_column=sa.Column("payload_bytes", sa.LargeBinary, nullable=True)
    )
    payload_json: dict | None = Field(
        default=None, sa_column=sa.Column("payload_json", JSONB, nullable=True)
    )
    sklearn_version: str = ""
    feature_hash: str = ""
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )


class KillSwitch(SQLModel, table=True):
    """Single-row operator kill-switch.

    Both bots check this row at the top of every cycle and refuse to
    enter new positions while ``enabled`` is True. The monitor still
    enforces SL/TP exits — closing risk is *less* dangerous than holding
    overnight under unknown failure.
    """

    __tablename__ = "kill_switch"

    id: int = Field(default=1, primary_key=True)
    enabled: bool = Field(default=False)
    reason: str | None = None
    set_by: str | None = None
    set_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True))


# ── Schema authority ────────────────────────────────────────────
#
# Alembic is the single source of truth for schema. `init_db` opens the
# engine and verifies the DB is on the expected revision; it never runs
# DDL itself. Apply migrations explicitly with `halal-trader db migrate`.


class SchemaError(RuntimeError):
    """Raised when the DB schema is not at the expected Alembic revision."""


async def init_db(database_url: str) -> AsyncEngine:
    """Open the async engine after verifying Alembic is at head.

    Behavior:
      * If `alembic_version` is missing AND any expected table exists, the DB
        was populated by a pre-Alembic `create_all` codepath. Raise
        `SchemaError` directing the operator to `halal-trader db stamp head`.
      * If `alembic_version` is missing AND the DB is empty, raise
        `SchemaError` directing the operator to `halal-trader db migrate`.
      * If `alembic_version` is present but != head, raise `SchemaError`
        directing the operator to `halal-trader db migrate`.
      * Otherwise, return the engine.
    """
    import sqlalchemy as sa

    engine = create_async_engine(database_url)

    expected_head = _alembic_head_revision()
    expected_tables = set(SQLModel.metadata.tables.keys())

    async with engine.connect() as conn:
        existing_tables = set(await conn.run_sync(lambda c: sa.inspect(c).get_table_names()))
        alembic_table_present = "alembic_version" in existing_tables

        current_revision: str | None = None
        if alembic_table_present:
            row = await conn.execute(sa.text("SELECT version_num FROM alembic_version"))
            first = row.first()
            current_revision = first[0] if first else None

    if current_revision == expected_head:
        return engine

    await engine.dispose()
    adopted = expected_tables.intersection(existing_tables)

    if current_revision is None:
        if adopted:
            raise SchemaError(
                f"Database has tables {sorted(adopted)} but no recorded Alembic "
                f"revision. This DB pre-dates Alembic-managed schema. "
                f"Run `halal-trader db stamp head` once to adopt it."
            )
        raise SchemaError(
            "Database is empty and not initialized. "
            "Run `halal-trader db migrate` to create the schema."
        )

    raise SchemaError(
        f"Database is at revision {current_revision!r}, expected {expected_head!r}. "
        f"Run `halal-trader db migrate`."
    )


def _alembic_head_revision() -> str:
    """Return the head revision id from the local alembic config."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config(str(_alembic_ini_path()))
    script = ScriptDirectory.from_config(cfg)
    head = script.get_current_head()
    if head is None:
        raise SchemaError("Alembic has no head revision — migration tree is empty.")
    return head


def _alembic_ini_path() -> Path:
    """Locate alembic.ini at the project root."""
    return Path(__file__).resolve().parent.parent.parent.parent / "alembic.ini"
