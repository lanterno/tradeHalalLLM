"""SQLModel table definitions and database initialization."""

from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import Field, SQLModel

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


async def init_db(db_path: str) -> AsyncEngine:
    """Create the async engine and ensure all tables exist.

    Uses SQLModel create_all for new tables, then adds any missing columns
    to existing tables (SQLite ALTER TABLE ADD COLUMN).
    """
    import sqlalchemy as sa

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

        # Ensure new columns exist on pre-existing crypto_trades tables
        _NEW_CRYPTO_TRADE_COLUMNS = [
            ("entry_price", "REAL"),
            ("stop_loss", "REAL"),
            ("target_price", "REAL"),
            ("exit_price", "REAL"),
            ("exit_reason", "TEXT"),
            ("closed_at", "TIMESTAMP"),
        ]
        existing = await conn.execute(sa.text("PRAGMA table_info(crypto_trades)"))
        existing_names = {row[1] for row in existing}
        for col_name, col_type in _NEW_CRYPTO_TRADE_COLUMNS:
            if col_name not in existing_names:
                await conn.execute(
                    sa.text(f"ALTER TABLE crypto_trades ADD COLUMN {col_name} {col_type}")
                )

        _NEW_LLM_DECISION_COLUMNS = [
            ("thinking", "TEXT"),
        ]
        existing = await conn.execute(sa.text("PRAGMA table_info(llm_decisions)"))
        existing_names = {row[1] for row in existing}
        for col_name, col_type in _NEW_LLM_DECISION_COLUMNS:
            if col_name not in existing_names:
                await conn.execute(
                    sa.text(f"ALTER TABLE llm_decisions ADD COLUMN {col_name} {col_type}")
                )

    return engine
