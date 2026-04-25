"""SQLModel table definitions and database initialization."""

import logging
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import Field, SQLModel

logger = logging.getLogger(__name__)

# ── Stock Tables ────────────────────────────────────────────────


class Trade(SQLModel, table=True):
    """Record of a single trade (buy or sell)."""

    __tablename__ = "trades"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float | None = None
    order_id: str | None = None
    status: str = Field(default="pending")
    llm_reasoning: str | None = None


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
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LlmDecision(SQLModel, table=True):
    """Audit log entry for an LLM trading decision."""

    __tablename__ = "llm_decisions"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    provider: str
    model: str
    prompt_summary: str | None = None
    raw_response: str | None = None
    parsed_action: str | None = None  # stored as JSON string
    symbols: str | None = None  # stored as JSON string
    execution_ms: int | None = None
    thinking: str | None = None  # reasoning chain from thinking-mode LLMs


# ── Crypto Tables ──────────────────────────────────────────────


class CryptoTrade(SQLModel, table=True):
    """Record of a single crypto trade."""

    __tablename__ = "crypto_trades"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
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
    closed_at: datetime | None = None


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
    screening_criteria: str | None = None  # JSON string of criteria met/failed
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class IndicatorSnapshot(SQLModel, table=True):
    """Indicator feature vector captured at trade entry time for ML training."""

    __tablename__ = "indicator_snapshots"

    id: int | None = Field(default=None, primary_key=True)
    trade_id: int = Field(index=True)
    pair: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    parameter: str
    old_value: float | None = None
    new_value: float
    reasoning: str | None = None


# ── Schema authority ────────────────────────────────────────────
#
# Alembic is the single source of truth for schema. `init_db` opens the
# engine and verifies the DB is on the expected revision; it never runs
# DDL itself. Apply migrations explicitly with `halal-trader db migrate`.


class SchemaError(RuntimeError):
    """Raised when the DB schema is not at the expected Alembic revision."""


async def init_db(db_path: str | Path) -> AsyncEngine:
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

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    expected_head = _alembic_head_revision()
    expected_tables = set(SQLModel.metadata.tables.keys())

    async with engine.connect() as conn:
        version_row = await conn.execute(
            sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='alembic_version'")
        )
        alembic_table_present = version_row.first() is not None

        existing_tables: set[str] = set()
        result = await conn.execute(sa.text("SELECT name FROM sqlite_master WHERE type='table'"))
        existing_tables = {row[0] for row in result}

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
                f"Database at {db_path} has tables {sorted(adopted)} but no "
                f"recorded Alembic revision. This DB pre-dates Alembic-managed "
                f"schema. Run `halal-trader db stamp head` once to adopt it."
            )
        raise SchemaError(
            f"Database at {db_path} is empty and not initialized. "
            f"Run `halal-trader db migrate` to create the schema."
        )

    raise SchemaError(
        f"Database at {db_path} is at revision {current_revision!r}, "
        f"expected {expected_head!r}. Run `halal-trader db migrate`."
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
