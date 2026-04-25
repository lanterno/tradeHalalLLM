"""Binance exchange client — async REST wrapper implementing CryptoBroker protocol."""

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Any

from binance import AsyncClient, BinanceAPIException

from halal_trader.domain.models import CryptoAccount, CryptoBalance, Kline

logger = logging.getLogger(__name__)

_MAX_CONCURRENT_REQUESTS = 10

# Notional value (in USDT) below which we treat a balance / order as dust.
# Shared by the executor, monitor, and liquidator so any tuning happens in
# one place. Binance enforces a per-symbol min-notional on top of this; the
# constant is the floor we use for housekeeping decisions (skip auto-exit,
# refuse to place a buy that's smaller than the floor, etc.).
DUST_NOTIONAL_USD: float = 5.0


@dataclass
class SymbolFilter:
    """Binance exchange trading rules for a single symbol."""

    min_qty: float
    max_qty: float
    step_size: float
    min_notional: float
    tick_size: float
    base_asset_precision: int
    quote_asset_precision: int


def extract_fill_price(order_result: dict[str, Any]) -> float | None:
    """Extract average fill price from a Binance order result."""
    fills = order_result.get("fills", [])
    if fills:
        total_qty = sum(float(f.get("qty", 0)) for f in fills)
        total_cost = sum(float(f.get("price", 0)) * float(f.get("qty", 0)) for f in fills)
        if total_qty > 0:
            return total_cost / total_qty
    exec_qty = float(order_result.get("executedQty", 0))
    cumulative = float(order_result.get("cumulativeQuoteQty", 0))
    if exec_qty > 0 and cumulative > 0:
        return cumulative / exec_qty
    return None


class BinanceClient:
    """Async Binance client implementing the CryptoBroker protocol.

    Supports both testnet and production via a simple toggle.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        testnet: bool = True,
        configured_pairs: list[str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._testnet = testnet
        self._client: AsyncClient | None = None
        self._configured_pairs = configured_pairs or []
        self._relevant_assets: set[str] | None = None
        self._symbol_filters: dict[str, SymbolFilter] = {}
        self._semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)
        self._latest_price_cache: dict[str, float] = {}
        self._account_cache: tuple[float, CryptoAccount] | None = None
        self._account_cache_ttl = 10.0
        self._filters_loaded_at: float = 0.0
        self._filters_refresh_interval = 3600.0
        if configured_pairs:
            self._relevant_assets = {
                p.upper().removesuffix("USDT").removesuffix("BUSD") for p in configured_pairs
            }

    async def connect(self) -> None:
        """Create and initialise the async Binance client."""
        self._client = await AsyncClient.create(
            api_key=self._api_key,
            api_secret=self._secret_key,
            testnet=self._testnet,
        )
        mode = "TESTNET" if self._testnet else "PRODUCTION"
        logger.info("Binance client connected (%s)", mode)

        await self._load_symbol_filters()

    async def disconnect(self) -> None:
        """Close the client session."""
        if self._client:
            await self._client.close_connection()
            self._client = None
            logger.info("Binance client disconnected")

    @property
    def client(self) -> AsyncClient:
        if self._client is None:
            raise RuntimeError("BinanceClient not connected — call connect() first")
        return self._client

    # ── Symbol filter helpers ──────────────────────────────────

    async def _load_symbol_filters(self) -> None:
        """Fetch exchange info for configured pairs and cache trading rules."""
        for pair in self._configured_pairs:
            try:
                info = await self.client.get_symbol_info(pair.upper())
                if info is None:
                    continue
                sf = self._parse_symbol_filters(info)
                if sf:
                    self._symbol_filters[pair.upper()] = sf
            except Exception as e:
                logger.debug("Failed to load symbol info for %s: %s", pair, e)

        self._filters_loaded_at = time.monotonic()
        if self._symbol_filters:
            logger.info("Loaded exchange filters for %d symbols", len(self._symbol_filters))

    async def refresh_symbol_filters_if_stale(self) -> None:
        """Reload symbol filters if they haven't been refreshed within the interval."""
        elapsed = time.monotonic() - self._filters_loaded_at
        if elapsed >= self._filters_refresh_interval:
            logger.info("Refreshing symbol filters (stale for %.0fs)", elapsed)
            await self._load_symbol_filters()

    @staticmethod
    def _parse_symbol_filters(info: dict[str, Any]) -> SymbolFilter | None:
        """Parse LOT_SIZE, NOTIONAL/MIN_NOTIONAL, and PRICE_FILTER from exchange info."""
        min_qty = 0.0
        max_qty = 0.0
        step_size = 0.0
        min_notional = 5.0
        tick_size = 0.01

        for f in info.get("filters", []):
            ft = f.get("filterType", "")
            if ft == "LOT_SIZE":
                min_qty = float(f.get("minQty", 0))
                max_qty = float(f.get("maxQty", 0))
                step_size = float(f.get("stepSize", 0))
            elif ft in ("NOTIONAL", "MIN_NOTIONAL"):
                min_notional = float(f.get("minNotional", 5.0))
            elif ft == "PRICE_FILTER":
                tick_size = float(f.get("tickSize", 0.01))

        if step_size <= 0:
            return None

        return SymbolFilter(
            min_qty=min_qty,
            max_qty=max_qty,
            step_size=step_size,
            min_notional=min_notional,
            tick_size=tick_size,
            base_asset_precision=info.get("baseAssetPrecision", 8),
            quote_asset_precision=info.get("quoteAssetPrecision", 8),
        )

    def get_symbol_filter(self, symbol: str) -> SymbolFilter | None:
        """Get cached trading rules for a symbol."""
        return self._symbol_filters.get(symbol.upper())

    def get_cached_price(self, symbol: str) -> float | None:
        """Get the last known price for a symbol from the cache."""
        return self._latest_price_cache.get(symbol.upper())

    def round_quantity(self, symbol: str, qty: float) -> float:
        """Round a quantity to the nearest valid step size for a symbol."""
        f = self._symbol_filters.get(symbol.upper())
        if f is None or f.step_size <= 0:
            return qty
        precision = max(0, int(round(-math.log10(f.step_size))))
        rounded = math.floor(qty * 10**precision) / 10**precision
        return max(f.min_qty, min(f.max_qty, rounded))

    def round_price(self, symbol: str, price: float) -> float:
        """Round a price to the nearest valid tick size for a symbol."""
        f = self._symbol_filters.get(symbol.upper())
        if f is None or f.tick_size <= 0:
            return price
        precision = max(0, int(round(-math.log10(f.tick_size))))
        return round(price, precision)

    def format_filters_for_prompt(self) -> str:
        """Format all symbol filters as text for the LLM prompt."""
        if not self._symbol_filters:
            return "No exchange trading rules available."
        lines = []
        for sym, f in sorted(self._symbol_filters.items()):
            lines.append(
                f"  {sym}: min_qty={f.min_qty}, step={f.step_size}, "
                f"min_notional=${f.min_notional:.2f}, tick={f.tick_size}"
            )
        return "\n".join(lines)

    # ── CryptoBroker protocol methods ──────────────────────────

    def invalidate_account_cache(self) -> None:
        """Force the next get_account() to make a fresh API call."""
        self._account_cache = None

    async def get_account(self) -> CryptoAccount:
        """Get account snapshot with USDT-equivalent balances.

        Uses a short-lived cache to avoid redundant calls within the same cycle.
        Only looks up ticker prices for the configured trading pair base assets
        (e.g. BTC, ETH) to avoid hundreds of API calls on testnet accounts.
        """
        now = time.monotonic()
        if self._account_cache:
            cached_time, cached_account = self._account_cache
            if now - cached_time < self._account_cache_ttl:
                return cached_account

        info = await self.client.get_account()
        balances = info.get("balances", [])

        total = 0.0
        available = 0.0
        locked = 0.0
        usdt_free = 0.0

        to_price: list[dict[str, Any]] = []
        for b in balances:
            asset = b["asset"]
            free = float(b["free"])
            lock = float(b["locked"])

            if asset == "USDT":
                total += free + lock
                available += free
                locked += lock
                usdt_free = free
            elif free + lock > 0:
                if self._relevant_assets is None or asset in self._relevant_assets:
                    to_price.append({"asset": asset, "free": free, "locked": lock})

        async def _price_for(asset: str) -> float | None:
            try:
                ticker = await asyncio.wait_for(
                    self.client.get_symbol_ticker(symbol=f"{asset}USDT"),
                    timeout=5.0,
                )
                price = float(ticker["price"])
                self._latest_price_cache[f"{asset}USDT"] = price
                return price
            except BinanceAPIException, KeyError, TimeoutError, asyncio.TimeoutError:
                return None

        if to_price:
            prices = await asyncio.gather(
                *[_price_for(b["asset"]) for b in to_price],
                return_exceptions=True,
            )
            for b, price in zip(to_price, prices):
                if isinstance(price, (int, float)) and price is not None:
                    total += (b["free"] + b["locked"]) * price
                    available += b["free"] * price
                    locked += b["locked"] * price

        account = CryptoAccount(
            total_balance_usdt=total,
            available_balance_usdt=available,
            in_order_usdt=locked,
            usdt_free=usdt_free,
        )
        self._account_cache = (time.monotonic(), account)
        return account

    async def get_balances(self) -> list[CryptoBalance]:
        """Get all non-zero balances."""
        info = await self.client.get_account()
        balances = []
        for b in info.get("balances", []):
            free = float(b["free"])
            lock = float(b["locked"])
            if free > 0 or lock > 0:
                balances.append(CryptoBalance(asset=b["asset"], free=free, locked=lock))
        return balances

    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Get open orders, optionally filtered by symbol."""
        if symbol:
            return await self.client.get_open_orders(symbol=symbol)
        return await self.client.get_open_orders()

    async def get_klines(self, symbol: str, interval: str = "1m", limit: int = 100) -> list[Kline]:
        """Fetch historical klines (candlesticks)."""
        async with self._semaphore:
            raw = await self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
        klines = []
        for k in raw:
            klines.append(
                Kline(
                    open_time=int(k[0]),
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                    close_time=int(k[6]),
                )
            )
        return klines

    async def get_order_book(self, symbol: str, limit: int = 10) -> dict[str, Any]:
        """Get order book depth."""
        async with self._semaphore:
            book = await self.client.get_order_book(symbol=symbol, limit=limit)
        return {
            "bids": [[float(p), float(q)] for p, q in book.get("bids", [])],
            "asks": [[float(p), float(q)] for p, q in book.get("asks", [])],
        }

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: float | None = None,
    ) -> dict[str, Any]:
        """Place an order on Binance.

        Automatically rounds quantity to the symbol's step size.
        """
        quantity = self.round_quantity(symbol, quantity)

        sf = self.get_symbol_filter(symbol)
        if sf:
            qty_precision = max(0, int(round(-math.log10(sf.step_size))))
        else:
            qty_precision = 8

        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": f"{quantity:.{qty_precision}f}",
        }

        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("price is required for LIMIT orders")
            params["price"] = f"{price:.8f}"
            params["timeInForce"] = "IOC"  # Immediate-or-cancel for fast trading

        async with self._semaphore:
            result = await self.client.create_order(**params)
        self.invalidate_account_cache()
        logger.info(
            "Order placed: %s %s %s qty=%s — orderId=%s status=%s",
            side.upper(),
            symbol,
            order_type.upper(),
            quantity,
            result.get("orderId"),
            result.get("status"),
        )
        return result

    async def cancel_order(self, symbol: str, order_id: str) -> dict[str, Any]:
        """Cancel an open order."""
        result = await self.client.cancel_order(symbol=symbol, orderId=order_id)
        logger.info("Order cancelled: %s orderId=%s", symbol, order_id)
        return result

    async def get_ticker_price(self, symbol: str) -> float:
        """Get current ticker price for a symbol."""
        async with self._semaphore:
            ticker = await self.client.get_symbol_ticker(symbol=symbol)
        price = float(ticker["price"])
        self._latest_price_cache[symbol.upper()] = price
        return price
