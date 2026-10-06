"""How well the core's orders were filled: the fill against two prices.

* **arrival:** the price the plan used when it sized the order (the latest
  trade a few seconds before submission). Slippage here is the market's
  cost of trading, spread and impact.
* **close:** the session's official close, which is what the ``core``
  forward book assumes it trades at (plus a 5 bps cost). Slippage here is
  what makes the account drift from its book, so it is the number the
  live-money gate's tracking test ultimately feels.

Signed so that positive is a cost: a buy filled above the reference, or a
sell filled below it. Averages are weighted by filled notional. Fills come
from the broker ledger (``broker_activities``, the order's own account),
joined to the order by Alpaca's order id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.market_hours import MARKET_TZ
from halal_trader.portfolio.core_account import CORE_PAPER

BOOK_COST_BPS = 5.0  # what the forward book charges per side


@dataclass(frozen=True, slots=True)
class OrderFill:
    submitted_at: datetime
    symbol: str
    side: str
    qty: float
    est_price: float
    filled_qty: float
    fill_price: float | None  # volume-weighted over the order's fills
    close: float | None

    @property
    def filled_notional(self) -> float:
        return self.filled_qty * (self.fill_price or 0.0)

    def _bps(self, reference: float | None) -> float | None:
        if not self.fill_price or not reference:
            return None
        sign = 1.0 if self.side == "buy" else -1.0
        return sign * (self.fill_price / reference - 1) * 1e4

    @property
    def vs_arrival_bps(self) -> float | None:
        return self._bps(self.est_price)

    @property
    def vs_close_bps(self) -> float | None:
        return self._bps(self.close)

    @property
    def status(self) -> str:
        if self.filled_qty <= 0:
            return "unfilled"
        return "filled" if self.filled_qty >= self.qty * 0.999 else "partial"


@dataclass
class Report:
    start: date
    end: date
    orders: list[OrderFill] = field(default_factory=list)

    def _weighted(self, attr: str) -> float | None:
        pairs = [
            (getattr(o, attr), o.filled_notional)
            for o in self.orders
            if getattr(o, attr) is not None and o.filled_notional > 0
        ]
        total = sum(w for _, w in pairs)
        return sum(v * w for v, w in pairs) / total if total else None

    @property
    def vs_arrival_bps(self) -> float | None:
        return self._weighted("vs_arrival_bps")

    @property
    def vs_close_bps(self) -> float | None:
        return self._weighted("vs_close_bps")

    @property
    def filled_notional(self) -> float:
        return sum(o.filled_notional for o in self.orders)

    @property
    def cost_vs_close_usd(self) -> float | None:
        """Dollars the fills cost against trading every order at the close."""
        bps = self.vs_close_bps
        return None if bps is None else bps / 1e4 * self.filled_notional

    def count(self, status: str) -> int:
        return sum(1 for o in self.orders if o.status == status)

    def worst(self, n: int = 5) -> list[OrderFill]:
        scored = [o for o in self.orders if o.vs_arrival_bps is not None]
        return sorted(scored, key=lambda o: -(o.vs_arrival_bps or 0.0))[:n]

    def summary(self) -> dict[str, object]:
        def r(x: float | None, d: int = 1) -> float | None:
            return None if x is None else round(x, d)

        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "orders": len(self.orders),
            "filled": self.count("filled"),
            "partial": self.count("partial"),
            "unfilled": self.count("unfilled"),
            "filled_notional": round(self.filled_notional, 2),
            "vs_arrival_bps": r(self.vs_arrival_bps),
            "vs_close_bps": r(self.vs_close_bps),
            "book_cost_bps": BOOK_COST_BPS,
            "cost_vs_close_usd": r(self.cost_vs_close_usd, 2),
            "worst": [
                {"symbol": o.symbol, "side": o.side, "vs_arrival_bps": r(o.vs_arrival_bps)}
                for o in self.worst()
            ],
        }


def _bounds(start: date, end: date) -> tuple[datetime, datetime]:
    lo = datetime.combine(start, time(), MARKET_TZ).astimezone(UTC)
    hi = datetime.combine(end + timedelta(days=1), time(), MARKET_TZ).astimezone(UTC)
    return lo, hi


async def report(
    engine: AsyncEngine, start: date, end: date | None = None, *, account: str = CORE_PAPER
) -> Report:
    """Every submitted order of the core ``account`` in [start, end] (New York
    dates) with its fills."""
    end = end or start
    lo, hi = _bounds(start, end)
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT o.submitted_at, o.symbol, o.side, o.qty, o.est_price, "
                "coalesce(sum(f.qty), 0) AS filled_qty, "
                "sum(f.qty * f.price) / nullif(sum(f.qty), 0) AS fill_price, "
                "(SELECT b.close FROM daily_bars b WHERE b.symbol = o.symbol "
                " AND b.adjustment = 'raw' "
                " AND b.day = (o.submitted_at AT TIME ZONE 'America/New_York')::date) AS close "
                "FROM core_orders o LEFT JOIN broker_activities f "
                "ON f.account = o.account AND f.activity_type = 'FILL' "
                "AND f.order_id = o.broker_order_id "
                "WHERE o.account = :a AND o.status = 'submitted' "
                "AND o.submitted_at >= :lo AND o.submitted_at < :hi "
                "GROUP BY o.id ORDER BY o.submitted_at"
            ),
            {"lo": lo, "hi": hi, "a": account},
        )
        orders = [
            OrderFill(
                r.submitted_at,
                r.symbol,
                r.side,
                float(r.qty),
                float(r.est_price),
                float(r.filled_qty),
                float(r.fill_price) if r.fill_price is not None else None,
                float(r.close) if r.close is not None else None,
            )
            for r in rows
        ]
    return Report(start, end, orders)
