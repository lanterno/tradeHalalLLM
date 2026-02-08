"""SQLModel table definitions and database initialization."""

from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import Field, SQLModel


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


async def init_db(db_path: str) -> AsyncEngine:
    """Create the async engine and ensure all tables exist via Alembic.

    Falls back to SQLModel.metadata.create_all if Alembic is not configured
    (e.g. in tests or first-time setup).
    """
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    # Apply schema via create_all (idempotent — only creates missing tables)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    return engine
