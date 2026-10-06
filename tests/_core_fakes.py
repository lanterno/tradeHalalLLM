"""A fake Alpaca account for the core portfolio's tests, and screen/settings helpers.

The broker behaves like Alpaca where the core depends on it: market orders
fill at the given price (sells can be told not to), cash moves with fills,
a ``client_order_id`` the account has already used is refused, and open
orders are listed until they fill.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.domain.models import Account, MarketClock, Position


class FakeCoreBroker:
    def __init__(
        self,
        *,
        cash: float,
        positions: list[Position] | None = None,
        prices: dict[str, float] | None = None,
        margin: float = 1.0,
        shorting: bool = False,
        account_id: str = "core-acct",
        is_open: bool = True,
        sells_fill: bool = True,
        open_orders: list[dict[str, Any]] | None = None,
    ) -> None:
        self.cash = cash
        self.positions = {p.symbol: p for p in positions or []}
        self.prices = prices or {}
        self.margin, self.shorting, self.account_id = margin, shorting, account_id
        self.is_open, self.sells_fill = is_open, sells_fill
        self.open_orders = list(open_orders or [])
        self.orders: list[tuple[str, str, float]] = []
        self.client_ids: list[str] = []
        self.by_id: dict[str, dict[str, Any]] = {}
        self.polls = 0

    async def disconnect(self) -> None:
        return None

    # ── account ──
    async def get_account_info(self) -> Account:
        equity = self.cash + sum(p.qty * p.current_price for p in self.positions.values())
        return Account(
            equity=equity,
            cash=self.cash,
            buying_power=self.cash * self.margin,
            portfolio_value=equity,
            status="ACTIVE",
            account_id=self.account_id,
            account_number=f"N-{self.account_id}",
            multiplier=self.margin,
            shorting_enabled=self.shorting,
        )

    async def get_all_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if p.qty > 0]

    async def get_clock(self) -> MarketClock:
        return MarketClock(is_open=self.is_open)

    async def get_stock_snapshot(self, symbols: str) -> dict[str, Any]:
        return {
            s: {"latestTrade": {"p": self.prices[s]}}
            for s in symbols.split(",")
            if s in self.prices
        }

    # ── orders ──
    async def get_open_orders(self) -> list[dict[str, Any]]:
        working = [o for o in self.by_id.values() if o["status"] not in ("filled", "canceled")]
        return self.open_orders + working

    async def get_order_by_id(self, order_id: str) -> dict[str, Any]:
        self.polls += 1
        return dict(self.by_id[order_id])

    async def place_order(
        self, symbol: str, side: str, quantity: float, *, client_order_id: str | None = None
    ) -> dict[str, Any]:
        if client_order_id in self.client_ids:
            return {"error": {"status": 422, "detail": "client_order_id must be unique"}}
        self.client_ids.append(client_order_id or "")
        self.orders.append((symbol, side, quantity))
        oid = f"o{len(self.orders)}"
        price = self.prices.get(symbol) or (
            self.positions[symbol].current_price if symbol in self.positions else 0.0
        )
        order = {"id": oid, "symbol": symbol, "side": side, "qty": str(quantity)}
        if side == "sell" and not self.sells_fill:
            order["status"] = "new"
        else:
            order["status"] = "filled"
            self._fill(symbol, side, quantity, price)
        self.by_id[oid] = order
        return {"id": oid, "status": "accepted", "client_order_id": client_order_id}

    def _fill(self, symbol: str, side: str, qty: float, price: float) -> None:
        held = self.positions.get(symbol)
        if side == "buy":
            self.cash -= qty * price
            if held is None:
                self.positions[symbol] = Position(symbol=symbol, qty=qty, current_price=price)
            else:
                held.qty += qty
        else:
            self.cash += qty * price
            assert held is not None and held.qty >= qty - 1e-9, "sold more than held"
            held.qty -= qty


def core_settings(
    *,
    paper: bool = True,
    token: str = "",
    ceiling: float = 1_000.0,
    day_key: str = "day-key",
    core_key: str = "core-key",
) -> SimpleNamespace:
    return SimpleNamespace(
        core=SimpleNamespace(
            enabled=True,
            alpaca_api_key=core_key,
            alpaca_secret_key="core-secret",
            paper=paper,
            top_n=100,
            live_confirmation=token,
            live_max_notional=ceiling,
        ),
        alpaca=SimpleNamespace(api_key=day_key, secret_key="day-secret", paper_trade=True),
    )


def day_trader_is(account_id: str | None):
    """A ``day_trader`` lookup returning an account with ``account_id`` (None: no keys)."""

    async def lookup(_settings: Any) -> Account | None:
        if account_id is None:
            return None
        return Account(account_id=account_id, account_number=f"N-{account_id}")

    return lookup


async def screen(
    engine: AsyncEngine, as_of: date, rows: dict[str, tuple[str, float, float]]
) -> None:
    """symbol -> (verdict, price, shares outstanding)."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, cik, sic_description, verdict, "
                "reasons, metrics, method, screened_at) VALUES (:a, :s, :c, '', :v, '[]', "
                "CAST(:m AS JSONB), 't', now())"
            ),
            [
                {
                    "a": as_of,
                    "s": s,
                    "c": i + 1,
                    "v": v,
                    "m": f'{{"price": {p:f}, "shares_outstanding": {sh:f}}}',
                }
                for i, (s, (v, p, sh)) in enumerate(rows.items())
            ],
        )


async def no_sleep(_seconds: float) -> None:
    return None
