"""Pure-Python paper-trading brokers.

Round-4 wave 1.A introduced the broker plugin framework; this module
ships two adapters that need *no* external SDK:

* :class:`PaperStockBroker` — implements the :class:`Broker` Protocol.
* :class:`PaperCryptoBroker` — implements the :class:`CryptoBroker`
  Protocol.

They run a simple in-memory matching engine: market orders fill
immediately at the seed price (with optional configurable slippage),
positions / cash / P&L are tracked locally. Klines / snapshots /
bars come from the seed price (one synthetic bar per call by default;
operators can prime longer histories via the test-only
``seed_klines`` constructor arg).

This is *not* a backtester — :mod:`crypto.backtest` is. The paper
broker is for live-paper sessions where the operator wants the bot
to run end-to-end without burning real capital, real broker API keys,
or even real network. Especially useful for:

* New-user onboarding before they connect a real broker.
* CI integration tests that need a working broker contract.
* Demoing the dashboard to scholars / auditors / investors.

Halal compliance: the screener still runs; only the matching is
synthetic. Every paper trade flows through the same compliance
layer as a live trade, so paper sessions are still proof of halal
compliance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from halal_trader.domain.models import (
    Account,
    CryptoAccount,
    CryptoBalance,
    Kline,
    MarketClock,
    Position,
)

logger = logging.getLogger(__name__)


@dataclass
class _PaperFill:
    """Internal record of a single fill."""

    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime


@dataclass
class _PaperState:
    """Shared mutable state for a paper broker.

    Kept as a separate dataclass so the stocks + crypto adapters can
    each have their own knobs (decimals, dust-floor, etc.) while
    sharing the matching-engine plumbing.
    """

    cash: float = 100_000.0
    positions: dict[str, float] = field(default_factory=dict)
    avg_costs: dict[str, float] = field(default_factory=dict)
    fills: list[_PaperFill] = field(default_factory=list)
    next_order_id: int = 1
    seed_prices: dict[str, float] = field(default_factory=dict)
    slippage_bps: float = 0.0


def _apply_slippage(price: float, side: str, slippage_bps: float) -> float:
    """Apply a bps-scaled slippage in the unfavourable direction.

    Buys get a slightly higher price; sells get a slightly lower one.
    `slippage_bps=10` ≈ 0.1% adverse move.
    """
    if slippage_bps <= 0 or price <= 0:
        return price
    factor = slippage_bps / 10_000.0
    return price * (1 + factor) if side.lower() == "buy" else price * (1 - factor)


class PaperStockBroker:
    """In-memory paper-trading broker that satisfies the stocks
    :class:`halal_trader.domain.ports.Broker` Protocol.

    Initial cash defaults to $100k; operators override via the
    constructor or via :meth:`reset`. Seed prices are required: the
    matching engine has nothing to match against without them. The
    bot's prompt-context fetcher already calls
    :meth:`get_stock_snapshot` and :meth:`get_stock_bars` per cycle —
    those queries refresh the in-memory price for the symbol.
    """

    def __init__(
        self,
        *,
        starting_cash: float = 100_000.0,
        seed_prices: dict[str, float] | None = None,
        slippage_bps: float = 5.0,
    ) -> None:
        self._state = _PaperState(
            cash=starting_cash,
            seed_prices=dict(seed_prices or {}),
            slippage_bps=slippage_bps,
        )

    # ── Read-only surface ──────────────────────────────────────

    async def get_account_info(self) -> Account:
        equity = self._state.cash + sum(
            qty * self._state.seed_prices.get(sym, self._state.avg_costs.get(sym, 0.0))
            for sym, qty in self._state.positions.items()
        )
        return Account(
            equity=equity,
            buying_power=self._state.cash,
            cash=self._state.cash,
            portfolio_value=equity,
            status="ACTIVE",
        )

    async def get_clock(self) -> MarketClock:
        # Paper market is always open — operators wanting closed-market
        # behaviour should use the real broker. The clock-skip
        # short-circuit in the cycle is for live; paper sessions
        # generally want to run regardless of NYSE hours.
        return MarketClock(
            is_open=True,
            next_open="",
            next_close="",
            timestamp=datetime.now(UTC),
        )

    async def get_calendar(
        self, start: str | None = None, end: str | None = None
    ) -> list[dict[str, Any]]:
        return []

    async def get_all_positions(self) -> list[Position]:
        out: list[Position] = []
        for sym, qty in self._state.positions.items():
            if qty <= 0:
                continue
            avg = self._state.avg_costs.get(sym, 0.0)
            current = self._state.seed_prices.get(sym, avg)
            unrealized = (current - avg) * qty
            unrealized_pct = ((current / avg) - 1.0) if avg > 0 else 0.0
            out.append(
                Position(
                    symbol=sym,
                    qty=qty,
                    avg_entry_price=avg,
                    current_price=current,
                    unrealized_pl=unrealized,
                    unrealized_plpc=unrealized_pct,
                )
            )
        return out

    async def get_stock_snapshot(self, symbols: str) -> dict[str, Any]:
        """Return an Alpaca-shape snapshot for the requested symbol(s)."""
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        out: dict[str, Any] = {}
        for sym in sym_list:
            price = self._state.seed_prices.get(sym, 0.0)
            out[sym] = {
                "latest_trade": {"price": price, "timestamp": datetime.now(UTC).isoformat()},
                "latest_quote": {"bid_price": price * 0.999, "ask_price": price * 1.001},
                "daily_bar": {"close": price, "volume": 0, "high": price, "low": price},
            }
        return out

    async def get_stock_bars(
        self, symbol: str, days: int = 5, timeframe: str = "1Day"
    ) -> list[dict[str, Any]]:
        """Return a synthetic flat-price bar history for the symbol.

        Real history would need a market-data feed; paper sessions
        either prime via ``seed_klines`` (test-only) or accept that
        indicators will read flat (which conservatively short-circuits
        most strategies into HOLD).
        """
        price = self._state.seed_prices.get(symbol.upper(), 0.0)
        if price <= 0:
            return []
        return [
            {"o": price, "h": price, "l": price, "c": price, "v": 0.0} for _ in range(max(1, days))
        ]

    # ── Mutating surface ──────────────────────────────────────

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        time_in_force: str = "day",
    ) -> dict[str, Any]:
        sym = symbol.upper()
        side_lower = side.lower()
        if quantity <= 0:
            return {"status": "rejected", "reason": "non-positive quantity"}
        seed_price = self._state.seed_prices.get(sym, 0.0)
        if seed_price <= 0:
            return {"status": "rejected", "reason": f"no seed price for {sym}"}

        fill_price = _apply_slippage(seed_price, side_lower, self._state.slippage_bps)
        notional = fill_price * quantity

        if side_lower == "buy":
            if notional > self._state.cash:
                return {
                    "status": "rejected",
                    "reason": (
                        f"insufficient cash: need {notional:.2f}, have {self._state.cash:.2f}"
                    ),
                }
            self._state.cash -= notional
            current_qty = self._state.positions.get(sym, 0.0)
            current_avg = self._state.avg_costs.get(sym, 0.0)
            new_qty = current_qty + quantity
            self._state.avg_costs[sym] = (
                (current_qty * current_avg + notional) / new_qty if new_qty > 0 else 0.0
            )
            self._state.positions[sym] = new_qty
        elif side_lower == "sell":
            current_qty = self._state.positions.get(sym, 0.0)
            if quantity > current_qty:
                return {
                    "status": "rejected",
                    "reason": f"insufficient position: have {current_qty}, want to sell {quantity}",
                }
            self._state.cash += notional
            self._state.positions[sym] = current_qty - quantity
            if self._state.positions[sym] <= 0:
                self._state.positions.pop(sym, None)
                self._state.avg_costs.pop(sym, None)
        else:
            return {"status": "rejected", "reason": f"unknown side: {side}"}

        order_id = f"paper-{self._state.next_order_id}"
        self._state.next_order_id += 1
        now = datetime.now(UTC)
        self._state.fills.append(
            _PaperFill(
                symbol=sym, side=side_lower, quantity=quantity, price=fill_price, timestamp=now
            )
        )
        return {
            "id": order_id,
            "status": "filled",
            "symbol": sym,
            "side": side_lower,
            "filled_qty": str(quantity),
            "filled_avg_price": str(fill_price),
            "submitted_at": now.isoformat(),
            "filled_at": now.isoformat(),
        }

    async def get_order_by_id(self, order_id: str) -> dict[str, Any]:
        # Paper engine fills synchronously, so we don't track open
        # orders. Return a synthetic "filled" marker.
        return {"id": order_id, "status": "filled"}

    async def close_position(self, symbol: str) -> dict[str, Any]:
        sym = symbol.upper()
        qty = self._state.positions.get(sym, 0.0)
        if qty <= 0:
            return {"status": "skipped", "reason": "no position"}
        return await self.place_order(symbol=sym, side="sell", quantity=qty)

    async def close_all_positions(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for sym in list(self._state.positions):
            results.append(await self.close_position(sym))
        return results

    # ── Operator helpers (test + onboarding-only) ─────────────

    def set_seed_price(self, symbol: str, price: float) -> None:
        """Prime / update a symbol's price (called by tests and the
        onboarding flow)."""
        self._state.seed_prices[symbol.upper()] = float(price)

    def reset(self, *, starting_cash: float | None = None) -> None:
        """Reset the broker to a fresh state. Preserves seed prices so
        callers don't have to re-prime after a reset."""
        seed = dict(self._state.seed_prices)
        slip = self._state.slippage_bps
        cash = self._state.cash if starting_cash is None else starting_cash
        self._state = _PaperState(cash=cash, seed_prices=seed, slippage_bps=slip)


class PaperCryptoBroker:
    """In-memory paper-trading broker that satisfies the
    :class:`halal_trader.domain.ports.CryptoBroker` Protocol.

    Mirrors :class:`PaperStockBroker` but works in USDT-quoted pairs
    and uses crypto-shape balances + klines.
    """

    def __init__(
        self,
        *,
        starting_usdt: float = 100_000.0,
        seed_prices: dict[str, float] | None = None,
        slippage_bps: float = 5.0,
    ) -> None:
        self._state = _PaperState(
            cash=starting_usdt,
            seed_prices=dict(seed_prices or {}),
            slippage_bps=slippage_bps,
        )

    async def get_account(self) -> CryptoAccount:
        positions_value = sum(
            qty * self._state.seed_prices.get(sym, 0.0)
            for sym, qty in self._state.positions.items()
        )
        total = self._state.cash + positions_value
        return CryptoAccount(
            total_balance_usdt=total,
            available_balance_usdt=self._state.cash,
            in_order_usdt=0.0,
            usdt_free=self._state.cash,
        )

    async def get_balances(self) -> list[CryptoBalance]:
        out: list[CryptoBalance] = [CryptoBalance(asset="USDT", free=self._state.cash, locked=0.0)]
        for sym, qty in self._state.positions.items():
            if qty <= 0:
                continue
            asset = sym.removesuffix("USDT").removesuffix("BUSD")
            out.append(CryptoBalance(asset=asset, free=qty, locked=0.0))
        return out

    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return []

    async def get_klines(self, symbol: str, interval: str = "1m", limit: int = 100) -> list[Kline]:
        sym = symbol.upper()
        price = self._state.seed_prices.get(sym, 0.0)
        if price <= 0:
            return []
        now = int(datetime.now(UTC).timestamp() * 1000)
        out: list[Kline] = []
        for i in range(limit):
            ts = now - (limit - i) * 60_000
            out.append(
                Kline(
                    open_time=ts,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=0.0,
                    close_time=ts + 60_000,
                )
            )
        return out

    async def get_order_book(self, symbol: str, limit: int = 10) -> dict[str, Any]:
        sym = symbol.upper()
        price = self._state.seed_prices.get(sym, 0.0)
        if price <= 0:
            return {"bids": [], "asks": []}
        return {
            "bids": [[price * 0.999, 1.0]],
            "asks": [[price * 1.001, 1.0]],
        }

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: float | None = None,
    ) -> dict[str, Any]:
        sym = symbol.upper()
        side_lower = side.lower()
        if quantity <= 0:
            return {"status": "REJECTED", "reason": "non-positive quantity"}
        seed_price = self._state.seed_prices.get(sym, 0.0)
        if seed_price <= 0:
            return {"status": "REJECTED", "reason": f"no seed price for {sym}"}

        fill_price = _apply_slippage(seed_price, side_lower, self._state.slippage_bps)
        notional = fill_price * quantity

        if side_lower == "buy":
            if notional > self._state.cash:
                return {"status": "REJECTED", "reason": "insufficient USDT"}
            self._state.cash -= notional
            current_qty = self._state.positions.get(sym, 0.0)
            current_avg = self._state.avg_costs.get(sym, 0.0)
            new_qty = current_qty + quantity
            self._state.avg_costs[sym] = (
                (current_qty * current_avg + notional) / new_qty if new_qty > 0 else 0.0
            )
            self._state.positions[sym] = new_qty
        elif side_lower == "sell":
            current_qty = self._state.positions.get(sym, 0.0)
            if quantity > current_qty:
                return {"status": "REJECTED", "reason": "insufficient position"}
            self._state.cash += notional
            self._state.positions[sym] = current_qty - quantity
            if self._state.positions[sym] <= 0:
                self._state.positions.pop(sym, None)
                self._state.avg_costs.pop(sym, None)
        else:
            return {"status": "REJECTED", "reason": f"unknown side: {side}"}

        order_id = self._state.next_order_id
        self._state.next_order_id += 1
        now = datetime.now(UTC)
        self._state.fills.append(
            _PaperFill(
                symbol=sym, side=side_lower, quantity=quantity, price=fill_price, timestamp=now
            )
        )
        return {
            "orderId": order_id,
            "status": "FILLED",
            "symbol": sym,
            "side": side.upper(),
            "executedQty": str(quantity),
            "cumulativeQuoteQty": str(notional),
            "fills": [{"price": str(fill_price), "qty": str(quantity)}],
        }

    async def cancel_order(self, symbol: str, order_id: str) -> dict[str, Any]:
        # Paper fills synchronously — nothing to cancel.
        return {"status": "REJECTED", "reason": "no open order to cancel"}

    async def get_ticker_price(self, symbol: str) -> float:
        return self._state.seed_prices.get(symbol.upper(), 0.0)

    # ── Operator helpers ──────────────────────────────────────

    def set_seed_price(self, symbol: str, price: float) -> None:
        self._state.seed_prices[symbol.upper()] = float(price)

    def reset(self, *, starting_usdt: float | None = None) -> None:
        seed = dict(self._state.seed_prices)
        slip = self._state.slippage_bps
        cash = self._state.cash if starting_usdt is None else starting_usdt
        self._state = _PaperState(cash=cash, seed_prices=seed, slippage_bps=slip)


# ── Default registrations ─────────────────────────────────────


def _paper_stock_factory(_settings: Any) -> PaperStockBroker:
    return PaperStockBroker()


def _paper_crypto_factory(_settings: Any) -> PaperCryptoBroker:
    return PaperCryptoBroker()
