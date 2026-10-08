"""The broker ledger: Alpaca's record is the truth; ours is reconciled to it.

Three jobs:

* :func:`sync_broker_ledger` copies Alpaca's account activities and daily
  equity into ``broker_activities`` / ``broker_equity``. Idempotent:
  activities are insert-only by Alpaca's id, equity days are upserted.
* :func:`reconcile_fills` compares the broker's fills for one trading day
  with the fills the bot recorded in ``trades``, per (symbol, side).
* :func:`performance` measures the account from ``broker_equity`` alone.

Why: over the 2026 paper record the bot's own books disagreed with each
other and with the broker (daily_pnl compounded to +5.5% while broker equity
fell 100,000 -> 99,322), so nothing measured from them could be trusted to
decide whether a strategy earns capital.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.market_hours import today_eastern

if TYPE_CHECKING:
    from halal_trader.execution.alpaca_rest import AlpacaRestClient
    from halal_trader.portfolio.core_account import BrokerAccount

logger = logging.getLogger(__name__)

# Activities are re-requested from this far before the newest one on record:
# non-trade activities carry only a date, and inserts are idempotent anyway.
_OVERLAP = timedelta(days=2)
# Equity days are re-requested this far back: Alpaca can revise a recent day.
_EQUITY_REFRESH_DAYS = 7
# Fills are compared to this many shares; float noise is not drift.
_QTY_TOLERANCE = 1e-6
_TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True, slots=True)
class SyncResult:
    activities_fetched: int
    activities_new: int
    equity_days: int


@dataclass(frozen=True, slots=True)
class FillDrift:
    symbol: str
    side: str
    broker_qty: float
    recorded_qty: float

    @property
    def difference(self) -> float:
        return self.broker_qty - self.recorded_qty


@dataclass(frozen=True, slots=True)
class FillReconciliation:
    day: date
    broker_fills: int
    drifts: list[FillDrift] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.drifts


@dataclass(frozen=True, slots=True)
class Performance:
    first_day: date
    last_day: date
    days: int
    start_equity: float
    end_equity: float
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float | None
    max_drawdown: float
    best_day: float
    worst_day: float


async def sync_broker_ledger(
    engine: AsyncEngine, client: AlpacaRestClient, *, account: str = "paper"
) -> SyncResult:
    """Pull new activities and recent equity of one account into the ledger tables.

    ``account`` names it: "paper" for the day-trader's, "core" for the core
    portfolio's. Every reader filters on it, so the two never mix.
    """
    async with engine.connect() as conn:
        newest = (
            await conn.execute(
                text("SELECT max(transaction_time) FROM broker_activities WHERE account = :a"),
                {"a": account},
            )
        ).scalar()
        last_equity_day = (
            await conn.execute(
                text("SELECT max(day) FROM broker_equity WHERE account = :a"), {"a": account}
            )
        ).scalar()

    activities = await client.activities(after=newest - _OVERLAP if newest else None)
    new = 0
    async with engine.begin() as conn:
        for a in activities:
            result = await conn.execute(
                text(
                    """
                    INSERT INTO broker_activities (id, activity_type, transaction_time, symbol,
                        side, qty, price, net_amount, order_id, raw, account)
                    VALUES (:id, :activity_type, :transaction_time, :symbol, :side, :qty,
                        :price, :net_amount, :order_id, CAST(:raw AS JSONB), :account)
                    ON CONFLICT (id) DO NOTHING
                    """
                ),
                {
                    "id": a.id,
                    "activity_type": a.activity_type,
                    "transaction_time": a.transaction_time,
                    "symbol": a.symbol,
                    "side": a.side,
                    "qty": a.qty,
                    "price": a.price,
                    "net_amount": a.net_amount,
                    "order_id": a.order_id,
                    "raw": json.dumps(a.raw),
                    "account": account,
                },
            )
            new += result.rowcount or 0

    if last_equity_day is not None:
        start = last_equity_day - timedelta(days=_EQUITY_REFRESH_DAYS)
    else:
        async with engine.connect() as conn:
            first = (
                await conn.execute(
                    text("SELECT min(transaction_time) FROM broker_activities WHERE account = :a"),
                    {"a": account},
                )
            ).scalar()
        start = first.date() if first else today_eastern() - timedelta(days=365)
    points = await client.equity_history(start=start)
    async with engine.begin() as conn:
        for p in points:
            await conn.execute(
                text(
                    """
                    INSERT INTO broker_equity (account, day, equity, profit_loss,
                        profit_loss_pct, synced_at)
                    VALUES (:account, :day, :equity, :pl, :plp, now())
                    ON CONFLICT (account, day) DO UPDATE SET equity = EXCLUDED.equity,
                        profit_loss = EXCLUDED.profit_loss,
                        profit_loss_pct = EXCLUDED.profit_loss_pct,
                        synced_at = EXCLUDED.synced_at
                    """
                ),
                {
                    "account": account,
                    "day": p.day,
                    "equity": p.equity,
                    "pl": p.profit_loss,
                    "plp": p.profit_loss_pct,
                },
            )
    logger.info(
        "broker ledger synced: %d activities fetched (%d new), %d equity days",
        len(activities),
        new,
        len(points),
    )
    return SyncResult(len(activities), new, len(points))


async def equity_history(
    engine: AsyncEngine,
    account: str,
    *,
    since: date | None = None,
    through: date | None = None,
) -> list[tuple[date, float]]:
    """An account's daily closing equity from the broker, oldest first (days the
    account was empty, at zero, left out)."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT day, equity FROM broker_equity WHERE account = :a AND equity > 0 "
                "AND (CAST(:s AS DATE) IS NULL OR day >= :s) "
                "AND (CAST(:t AS DATE) IS NULL OR day <= :t) ORDER BY day"
            ),
            {"a": account, "s": since, "t": through},
        )
        return [(r.day, float(r.equity)) for r in rows]


async def sync_accounts(
    engine: AsyncEngine, accounts: Iterable[BrokerAccount]
) -> dict[str, SyncResult]:
    """:func:`sync_broker_ledger` for each account (portfolio/core_account.broker_accounts)."""
    from halal_trader.execution.alpaca_rest import AlpacaRestClient

    out: dict[str, SyncResult] = {}
    for a in accounts:
        client = AlpacaRestClient(a.api_key, a.secret_key, paper=a.paper)
        try:
            out[a.name] = await sync_broker_ledger(engine, client, account=a.name)
        finally:
            await client.aclose()
    return out


async def reconcile_fills(engine: AsyncEngine, day: date) -> FillReconciliation:
    """Compare the broker's fills on ``day`` (US/Eastern) with the bot's recorded fills."""
    broker: dict[tuple[str, str], float] = defaultdict(float)
    recorded: dict[tuple[str, str], float] = defaultdict(float)
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                """
                SELECT symbol, side, qty FROM broker_activities
                WHERE activity_type = 'FILL' AND account = 'paper'
                  AND (transaction_time AT TIME ZONE 'America/New_York')::date = :day
                """
            ),
            {"day": day},
        )
        broker_fills = 0
        for symbol, side, qty in rows:
            broker[(symbol, side)] += qty or 0.0
            broker_fills += 1
        rows = await conn.execute(
            text(
                """
                SELECT symbol, side, coalesce(filled_quantity, quantity) FROM trades
                WHERE status IN ('filled', 'partially_filled', 'closed')
                  AND (coalesce(filled_at, timestamp) AT TIME ZONE 'America/New_York')::date = :day
                """
            ),
            {"day": day},
        )
        for symbol, side, qty in rows:
            recorded[(symbol, side)] += qty or 0.0

    drifts = [
        FillDrift(symbol, side, broker.get((symbol, side), 0.0), recorded.get((symbol, side), 0.0))
        for symbol, side in sorted(set(broker) | set(recorded))
        if abs(broker.get((symbol, side), 0.0) - recorded.get((symbol, side), 0.0)) > _QTY_TOLERANCE
    ]
    return FillReconciliation(day=day, broker_fills=broker_fills, drifts=drifts)


async def performance(
    engine: AsyncEngine,
    *,
    start: date | None = None,
    end: date | None = None,
    account: str = "paper",
) -> Performance | None:
    """Account performance from broker equity alone. None with fewer than 2 days."""
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    """
                    SELECT day, equity, profit_loss_pct FROM broker_equity
                    WHERE account = :account
                      AND (CAST(:start AS DATE) IS NULL OR day >= :start)
                      AND (CAST(:end AS DATE) IS NULL OR day <= :end)
                    ORDER BY day
                    """
                ),
                {"start": start, "end": end, "account": account},
            )
        ).all()
    if len(rows) < 2:
        return None
    # Daily return = Alpaca's profit_loss_pct, which is net of deposits and
    # withdrawals. The first row's pct belongs to the day before the window.
    returns = [float(r.profit_loss_pct) for r in rows[1:]]
    growth = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        growth *= 1 + r
        peak = max(peak, growth)
        max_dd = min(max_dd, growth / peak - 1)
    n = len(returns)
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1) if n > 1 else 0.0
    vol = math.sqrt(var)
    return Performance(
        first_day=rows[0].day,
        last_day=rows[-1].day,
        days=n,
        start_equity=float(rows[0].equity),
        end_equity=float(rows[-1].equity),
        total_return=growth - 1,
        annualized_return=growth ** (_TRADING_DAYS_PER_YEAR / n) - 1,
        annualized_volatility=vol * math.sqrt(_TRADING_DAYS_PER_YEAR),
        sharpe=(mean / vol * math.sqrt(_TRADING_DAYS_PER_YEAR)) if vol > 0 else None,
        max_drawdown=max_dd,
        best_day=max(returns),
        worst_day=min(returns),
    )
