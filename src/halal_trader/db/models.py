"""SQLModel table definitions and database initialization."""

import logging
from datetime import UTC, date, datetime
from decimal import Decimal
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

    # SL/TP + close lifecycle, read by the position monitor and analytics.
    stop_loss: float | None = None
    target_price: float | None = None

    # How the trade entered the book. Used by the slow-out discipline
    # (memory: strategy-fast-in-slow-out): "reactor_momentum" trades
    # are LLM-untouchable on the SELL side — only the monitor's
    # rule-based exit can close them. None = legacy / "scheduled"
    # (default for cron-cycle entries).
    entry_type: str | None = None
    # Highest price seen since entry, persisted whenever the trailing stop
    # ratchets. The monitor's bogus-stop repair compares it with stop_loss;
    # held only in memory, a restart made a ratcheted, profit-locking stop
    # look "never reached" and repaired it DOWN instead of exiting.
    high_water_price: float | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    closed_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True))

    # Realized slippage (signed, in fraction of price) so the operator can
    # sanity-check the backtester's assumptions against fills. Only
    # paper_slippage_pct is written today; live_slippage_pct and
    # predicted_slippage_pct (a slippage-model forecast) have no writer.
    paper_slippage_pct: float | None = None
    live_slippage_pct: float | None = None
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
    prompt_version: str | None = None  # registry "name@hash", e.g. "trading.strategy.system@abc123"
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost_usd: float | None = None  # rounded float — Decimal aggregation done in code

    # Written by the agentic tool-calling mode, deleted 2026-10-01; the
    # column stays until a migration drops it. Always None for new rows.
    tool_transcript: list | None = Field(
        default=None,
        sa_column=sa.Column("tool_transcript", JSONB, nullable=True),
    )


class IndicatorSnapshot(SQLModel, table=True):
    """Indicator feature vector captured at trade entry time for ML training."""

    __tablename__ = "indicator_snapshots"

    id: int | None = Field(default=None, primary_key=True)
    trade_id: int = Field(index=True)
    pair: str  # the stock symbol (column named for the crypto bot)
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
    market: str  # 'stocks' ('crypto' on rows from before 2026-10-01)
    symbol: str  # asset/ticker affected
    db_quantity: float
    broker_quantity: float
    drift_pct: float  # |db - broker| / max(db, broker, 1e-9)
    drift_usd: float | None = None
    notes: str | None = None


class DailyRecommendation(SQLModel, table=True):
    """LLM-picked "stock of the day" — the single most promising halal stock.

    Advisory only: surfaced on the dashboard / CLI / API, never auto-traded.
    One row per generation; the most recent row for a given ``date`` is the
    active pick, and regenerating appends a new row so history is preserved.
    """

    __tablename__ = "daily_recommendations"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )
    date: str = Field(index=True)  # trading day, YYYY-MM-DD (US/Eastern)
    symbol: str
    conviction: float = Field(default=0.0)  # 0..1
    thesis: str = ""  # why this is the most promising buy today
    halal_note: str = ""  # Shariah-compliance justification
    suggested_entry: float | None = None
    suggested_target: float | None = None
    suggested_stop: float | None = None
    catalysts: str | None = None
    risks: str | None = None
    universe_size: int = Field(default=0)  # candidates considered
    model: str | None = None  # LLM model that produced the pick
    prompt_version: str | None = None
    # Per-symbol context the model weighed (price/trend/indicator summary),
    # kept for transparency on the dashboard.
    candidates: dict | None = Field(
        default=None, sa_column=sa.Column("candidates", JSONB, nullable=True)
    )
    # ── Outcome tracking (forward-return labeling; advisory track record) ──
    # Populated by the scorecard backfill once the pick has matured. Honest
    # measurement of whether the recommendations actually work.
    outcome_status: str = Field(default="pending")  # pending | scored | skipped
    scored_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True))
    entry_close: float | None = None  # close on/after the rec date (return baseline)
    fwd_return_1d: float | None = None  # % forward return, 1 trading day
    fwd_return_5d: float | None = None  # % forward return, 5 trading days
    fwd_return_20d: float | None = None  # % forward return, 20 trading days
    benchmark_return_5d: float | None = None  # halal benchmark (SPUS) over 5d
    # ── Path outcomes over the 5-day anchor window (bars AFTER the entry bar).
    # These score the LLM's *plan* (suggested_target/stop), not just direction.
    realized_high_5d: float | None = None  # max bar high over the 5d window
    realized_low_5d: float | None = None  # min bar low over the 5d window
    mfe_pct: float | None = None  # max favorable excursion % vs entry_close
    mae_pct: float | None = None  # max adverse excursion % vs entry_close (≤0)
    target_hit: bool | None = None  # any bar high ≥ suggested_target within 5d
    stop_hit: bool | None = None  # any bar low ≤ suggested_stop within 5d
    # Which level was touched first: target | stop | both_same_bar | none.
    # Daily bars can't sequence within a day — both_same_bar is honest ignorance.
    first_hit: str | None = None
    # ── Plan-anchored outcome: the actionable plan buys at the entry bar's
    # OPEN (the rec lands pre-market) and bracket-exits at target/stop, with
    # a time exit after 5 sessions (entry day included). Same-bar ties
    # resolve to the stop — pessimistic, honest for a long plan.
    entry_open: float | None = None  # open of the entry bar (plan fill proxy)
    plan_return_5d: float | None = None  # % outcome of the stated plan
    plan_exit: str | None = None  # target | stop | time


class QuantTrial(SQLModel, table=True):
    """One evaluated strategy/forecast variant — the anti-overfitting ledger.

    Every quant experiment (level-family validation, band calibration,
    backtest arm, …) records a row, INCLUDING failures: the Deflated Sharpe
    Ratio needs an honest trial count, and with ~10-20 tried variants the
    expected max in-sample Sharpe of pure noise already exceeds 1.0
    (docs/QUANT_PREDICTION_ROADMAP.md, honest foundations). Advisory
    bookkeeping — never read by any trading path.
    """

    __tablename__ = "quant_trials"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )
    name: str = Field(index=True)  # e.g. "levels.swing_zones.touch_hold_5d"
    kind: str  # level_family | band_calibration | backtest | signal
    config_hash: str  # short digest of the config dict (dedup/count aid)
    config: dict | None = Field(default=None, sa_column=sa.Column("config", JSONB, nullable=True))
    window: str = ""  # human description of the data window
    metrics: dict | None = Field(default=None, sa_column=sa.Column("metrics", JSONB, nullable=True))
    # Success criterion decided BEFORE looking at results (pre-registration).
    criterion: str | None = None
    verdict: str | None = None  # pass | fail | inconclusive


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
    """Legacy: persistent record of a dividend's haram-portion purification obligation.

    Never written by anything; ``purification_accruals`` (compliance/purification.py)
    is the dividend ledger since 2026-10-04. Still summed by the compliance summary
    so no row entered by hand could be lost.

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
    asset_class: str  # 'stock' ('crypto' on rows from before 2026-10-01)
    source: str  # 'zoya' | 'override' | 'cache' | … ('coingecko_rules' on old crypto rows)
    decision: str  # 'halal' | 'not_halal' | 'doubtful'
    criteria: dict | None = Field(
        default=None, sa_column=sa.Column("criteria", JSONB, nullable=True)
    )
    cache_hit: bool = Field(default=False)


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

    Nothing writes this table since ``core/regret`` was deleted on 2026-10-01;
    the model stays so the schema matches the migrations.
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


class BrokerActivity(SQLModel, table=True):
    """Alpaca's own record of every account activity -- the ledger of truth.

    Synced (insert-only, keyed by Alpaca's activity id) by
    execution/ledger.py. Fills, fees, dividends and transfers all land here
    as the broker reported them; the internal ``trades`` table is reconciled
    against this, never the other way round.
    """

    __tablename__ = "broker_activities"

    id: str = Field(primary_key=True)
    activity_type: str = Field(index=True)
    transaction_time: datetime = Field(index=True, sa_type=sa.DateTime(timezone=True))
    symbol: str | None = Field(default=None, index=True)
    side: str | None = None
    qty: float | None = None
    price: float | None = None
    net_amount: float | None = None
    order_id: str | None = None
    raw: dict[str, Any] = Field(sa_column=sa.Column("raw", JSONB, nullable=False))
    # Which Alpaca account: "paper" (the day-trader's) or "core".
    account: str = Field(default="paper", sa_column_kwargs={"server_default": "paper"})


class BrokerEquity(SQLModel, table=True):
    """Daily closing account equity as Alpaca reports it (portfolio history).

    The basis for performance measurement: returns, drawdown and Sharpe come
    from here, not from the bot's own daily_pnl bookkeeping, which disagreed
    with the broker by several points over the 2026 paper record.
    """

    __tablename__ = "broker_equity"

    account: str = Field(
        default="paper", primary_key=True, sa_column_kwargs={"server_default": "paper"}
    )
    day: date = Field(primary_key=True)
    equity: float
    profit_loss: float
    profit_loss_pct: float
    synced_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )


class LlmSpend(SQLModel, table=True):
    """LLM spend per UTC day and consumer process (core/llm/spend.py).

    One shared total across the stock bot and the shadow engine, which bill
    the same OpenRouter key; the daily cap is checked against the sum.
    """

    __tablename__ = "llm_spend"

    day: date = Field(primary_key=True)
    consumer: str = Field(primary_key=True)
    calls: int = Field(default=0)
    spent_usd: Decimal = Field(default=Decimal("0"), sa_type=sa.Numeric(14, 6))


class Heartbeat(SQLModel, table=True):
    """Last-seen time of each long-running component, one row per component.

    The bots and the dashboard run in separate containers, so "is the bot
    alive?" has to be answered from the database: the stock bot upserts a
    row per component (process, cycle, monitor) and /api/health/bot,
    `just health` and the System page read them. Before this the
    dashboard's "Bot Running" came from in-process state the web container
    never had, and /api/health was a hard-coded constant.
    """

    __tablename__ = "heartbeats"

    component: str = Field(primary_key=True)
    beat_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )
    detail: dict[str, Any] | None = Field(
        default=None, sa_column=sa.Column("detail", JSONB, nullable=True)
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

    # pool_pre_ping: liveness-check each pooled connection on checkout and
    # transparently replace dead ones — fixes the recurring
    # "connection is closed" / "ConnectionDoesNotExist" InterfaceErrors that
    # surfaced in the long-lived monitor/cycle loops when Postgres (or a proxy)
    # dropped an idle asyncpg connection out from under the pool.
    # pool_recycle: proactively retire connections older than 30 min, below
    # typical server-side idle timeouts, so stale ones are rare to begin with.
    engine = create_async_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=1800,
    )

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


# ── Research data store (data/, plan Phase 3) ─────────────────────────
# Re-derivable from Alpaca, so excluded from the nightly backup.


class MarketAsset(SQLModel, table=True):
    """Alpaca's active US-equity asset list (stocks and ETFs), as last synced."""

    __tablename__ = "market_assets"

    symbol: str = Field(primary_key=True)
    name: str
    exchange: str = Field(index=True)
    tradable: bool
    fractionable: bool
    status: str
    synced_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )


class DailyBarRow(SQLModel, table=True):
    """One daily SIP bar per (symbol, session, adjustment).

    ``adjustment`` is 'raw' (the prices that traded -- what executions and
    stops see) or 'all' (split- and dividend-adjusted -- what returns and
    factors need). Adjusted history is rewritten by every corporate action,
    so ``fetched_at`` records which vintage a row is.
    """

    __tablename__ = "daily_bars"

    symbol: str = Field(primary_key=True)
    day: date = Field(primary_key=True)
    adjustment: str = Field(primary_key=True)
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    trades: int | None = None
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )


class HalalScreenResult(SQLModel, table=True):
    """One in-house Shariah screening verdict per (as_of, symbol, method) (compliance/).

    Every run is kept, so the screening history is point-in-time: a backtest
    asks what the screen said on a given day. A re-screen under a new method
    adds rows beside the old ones rather than rewriting them; readers use the
    ``halal_screen_current`` view (the newest method per as_of and symbol).
    ``metrics`` stores the inputs and ratios behind the verdict so each
    decision can be audited.
    """

    __tablename__ = "halal_screen_results"

    as_of: date = Field(primary_key=True)
    symbol: str = Field(primary_key=True)
    cik: int | None = None
    sic_description: str = ""
    verdict: str = Field(index=True)
    reasons: list[str] = Field(sa_column=sa.Column("reasons", JSONB, nullable=False))
    metrics: dict[str, Any] = Field(sa_column=sa.Column("metrics", JSONB, nullable=False))
    method: str = Field(primary_key=True)
    screened_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )


class ForwardBook(SQLModel, table=True):
    """A strategy run forward on paper, with no orders (research/forward_book.py)."""

    __tablename__ = "forward_books"

    name: str = Field(primary_key=True)
    strategy: str
    params: dict[str, Any] = Field(sa_column=sa.Column("params", JSONB, nullable=False))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )


class ForwardBookDay(SQLModel, table=True):
    """One session of a forward book: NAV, return, turnover and end-of-day weights.

    Append-only: a session is written the evening it closes and never
    recomputed, so the record cannot be revised with hindsight.
    """

    __tablename__ = "forward_book_days"

    book: str = Field(primary_key=True, foreign_key="forward_books.name")
    day: date = Field(primary_key=True)
    nav: float
    day_return: float
    turnover: float
    weights: dict[str, float] = Field(sa_column=sa.Column("weights", JSONB, nullable=False))
    rebalance_next: bool = False
    recorded_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )


class AnnualFundamentals(SQLModel, table=True):
    """Gross profit and total assets per SEC filer per calendar-year frame: the
    quality factor's input (data/fundamentals.py)."""

    __tablename__ = "annual_fundamentals"

    cik: int = Field(primary_key=True)
    year: int = Field(primary_key=True)
    gross_profit: float | None = None
    assets: float | None = None


class Event(SQLModel, table=True):
    """One market event (a news article, a filing) about one symbol: the event
    store's append-only row (events/store.py). ``published_at`` is the source's
    timestamp; ``seen_at`` is when this system first stored it."""

    __tablename__ = "events"
    __table_args__ = (
        sa.UniqueConstraint("source", "source_id", "symbol", name="uq_events_source_item"),
    )

    id: int | None = Field(default=None, sa_column=sa.Column(sa.BigInteger(), primary_key=True))
    source: str
    source_id: str
    kind: str
    symbol: str
    published_at: datetime = Field(sa_type=sa.DateTime(timezone=True))
    seen_at: datetime = Field(sa_type=sa.DateTime(timezone=True))
    payload: dict | None = Field(default=None, sa_column=sa.Column("payload", JSONB, nullable=True))


class EventScore(SQLModel, table=True):
    """One scorer's judgement of one event; a new prompt or model is a new scorer."""

    __tablename__ = "event_scores"
    __table_args__ = (
        sa.UniqueConstraint("event_id", "scorer", name="uq_event_scores_event_scorer"),
    )

    id: int | None = Field(default=None, sa_column=sa.Column(sa.BigInteger(), primary_key=True))
    event_id: int = Field(sa_column=sa.Column(sa.BigInteger(), sa.ForeignKey("events.id")))
    scorer: str
    score: float
    tag: str | None = None
    rationale: str | None = None
    scored_at: datetime = Field(sa_type=sa.DateTime(timezone=True))


class EventFact(SQLModel, table=True):
    """A structured fact an extractor read from an event (e.g. EPS vs consensus).
    Several per event are possible (a result and its guidance)."""

    __tablename__ = "event_facts"

    id: int | None = Field(default=None, sa_column=sa.Column(sa.BigInteger(), primary_key=True))
    event_id: int = Field(sa_column=sa.Column(sa.BigInteger(), sa.ForeignKey("events.id")))
    extractor: str
    kind: str
    fields: dict = Field(sa_column=sa.Column("fields", JSONB, nullable=False))


class EventLabel(SQLModel, table=True):
    """An event's realised return over ``horizon`` sessions, raw and versus SPUS."""

    __tablename__ = "event_labels"

    event_id: int = Field(
        sa_column=sa.Column(sa.BigInteger(), sa.ForeignKey("events.id"), primary_key=True)
    )
    horizon: int = Field(primary_key=True)
    ret: float
    abn_ret: float
    labeled_at: datetime = Field(sa_type=sa.DateTime(timezone=True))


class Dividend(SQLModel, table=True):
    """A cash dividend from Alpaca's corporate actions: rate per share, by ex-date."""

    __tablename__ = "dividends"

    source_id: str = Field(primary_key=True)
    symbol: str
    ex_date: date
    payable_date: date | None = None
    record_date: date | None = None
    rate: float
    special: bool = False


class PurificationAccrual(SQLModel, table=True):
    """The share of one dividend owed to charity: dividend x the payer's impure-income
    ratio from the screen in force at the ex-date (compliance/purification.py)."""

    __tablename__ = "purification_accruals"

    account: str = Field(primary_key=True)  # "paper", or "book:<name>" (per notional)
    dividend_id: str = Field(primary_key=True)
    symbol: str
    ex_date: date
    payable_date: date | None = None
    shares: float
    dividend: float
    impure_ratio: float
    screen_as_of: date | None = None
    amount: float
    method: str
    accrued_at: datetime = Field(sa_type=sa.DateTime(timezone=True))
    paid_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True))
    paid_to: str | None = None  # the charity, as the operator records it


class ZakatAssessment(SQLModel, table=True):
    """One account's zakat for one hawl, by both of Dar al-Ifta's methods (trade
    goods: 2.5% of market value; income: 2.5% of dividends net of purification),
    the higher chosen (compliance/zakat.py)."""

    __tablename__ = "zakat_assessments"

    account: str = Field(primary_key=True)
    hawl_date: date = Field(primary_key=True)
    hawl_hijri: str
    period_start: date
    market_value: float
    trade_goods_zakat: float
    dividends: float
    purified: float
    income_zakat: float
    chosen: str
    amount: float
    holdings: dict | None = Field(
        default=None, sa_column=sa.Column("holdings", JSONB, nullable=True)
    )
    source: str
    computed_at: datetime = Field(sa_type=sa.DateTime(timezone=True))


class CoreOrder(SQLModel, table=True):
    """One order of the core portfolio (portfolio/core_executor.py), filled or
    refused, with the screen verdict it relied on: the receipt for every fill."""

    __tablename__ = "core_orders"

    id: int | None = Field(default=None, primary_key=True)
    # "core" (paper) or "core-live": portfolio/core_account.py.
    account: str = Field(default="core", sa_column_kwargs={"server_default": "core"})
    submitted_at: datetime = Field(sa_type=sa.DateTime(timezone=True))
    symbol: str
    side: str
    qty: float
    est_price: float
    notional: float
    reason: str
    screen_as_of: date | None = None
    status: str
    broker_order_id: str | None = None
    response: dict | None = Field(
        default=None, sa_column=sa.Column("response", JSONB, nullable=True)
    )


class CoreRun(SQLModel, table=True):
    """One run of the core executor: monthly or forced-sales-only, executed or a plan."""

    __tablename__ = "core_runs"

    id: int | None = Field(default=None, primary_key=True)
    account: str = Field(default="core", sa_column_kwargs={"server_default": "core"})
    run_on: date
    monthly: bool
    # Orders were placed (or the plan was legitimately empty). A halted run is
    # never executed, so it neither ends the month nor counts for the gate.
    executed: bool
    equity: float
    cash: float
    orders: int
    halted: str | None = None
    screen_as_of: date | None = None
    notes: list = Field(
        default_factory=list,
        sa_column=sa.Column("notes", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
    )
    recorded_at: datetime = Field(sa_type=sa.DateTime(timezone=True))


class MinuteBar(SQLModel, table=True):
    """One raw SIP minute bar (regular session) around an event: the intraday
    entry study's prices and, later, the slippage model's arrival prices."""

    __tablename__ = "minute_bars"

    symbol: str = Field(primary_key=True)
    ts: datetime = Field(primary_key=True, sa_type=sa.DateTime(timezone=True))
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None


class BackfillProgress(SQLModel, table=True):
    """One completed unit (a day, a quarter, a company) of a resumable backfill."""

    __tablename__ = "backfill_progress"

    task: str = Field(primary_key=True)
    unit: str = Field(primary_key=True)
    items: int
    done_at: datetime = Field(sa_type=sa.DateTime(timezone=True))


class EpsFact(SQLModel, table=True):
    """One reported earnings-per-share value from XBRL, with the date it was filed:
    the input to standardized unexpected earnings, known only from ``filed``."""

    __tablename__ = "eps_facts"

    cik: int = Field(primary_key=True)
    concept: str = Field(primary_key=True)
    start: date = Field(primary_key=True)
    end: date = Field(primary_key=True)
    accn: str = Field(primary_key=True)
    val: float
    form: str
    fp: str | None = None
    fy: int | None = None
    filed: date


class EtfHolding(SQLModel, table=True):
    """One equity holding of a halal index ETF from an N-PORT filing, dated by
    the filing (when it became public), not only by the period it reports."""

    __tablename__ = "etf_holdings"

    id: int | None = Field(default=None, primary_key=True)
    etf: str
    filed: date
    period_end: date
    ticker: str | None = None
    name: str
    cusip: str
    weight_pct: float


class TickerCik(SQLModel, table=True):
    """A ticker SEC's current ticker file does not list, matched to its filer by
    name (compliance/delisted.py). ``status`` says whether it was: mapped,
    fund (an ETF or fund, never a company), no_name, ambiguous, no_match."""

    __tablename__ = "ticker_ciks"

    symbol: str = Field(primary_key=True)
    status: str
    cik: int | None = None
    asset_name: str | None = None
    filer_name: str | None = None
    matched_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=sa.DateTime(timezone=True)
    )


class MonthlyBar(SQLModel, table=True):
    """One raw monthly SIP bar per (symbol, month), listed and delisted stocks
    alike: the input to the point-in-time liquidity universe (data/universe.py)."""

    __tablename__ = "monthly_bars"

    symbol: str = Field(primary_key=True)
    month: date = Field(primary_key=True)  # first of the month
    close: float
    volume: float
    vwap: float | None = None


class AccountSnapshot(SQLModel, table=True):
    """The newest value of one broker account (portfolio/snapshots.py): equity,
    cash, the previous close's equity and its positions. Overwritten each minute."""

    __tablename__ = "account_snapshots"

    account: str = Field(primary_key=True)
    taken_at: datetime = Field(sa_type=sa.DateTime(timezone=True))
    equity: float
    cash: float
    last_equity: float | None = None
    positions: list = Field(
        default_factory=list, sa_column=sa.Column("positions", JSONB, nullable=False)
    )


class Quote(SQLModel, table=True):
    """The newest price of a symbol and its previous close (portfolio/snapshots.py)."""

    __tablename__ = "quotes"

    symbol: str = Field(primary_key=True)
    taken_at: datetime = Field(sa_type=sa.DateTime(timezone=True))
    price: float
    prev_close: float | None = None
