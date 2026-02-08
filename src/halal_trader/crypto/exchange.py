"""Binance exchange client — async REST wrapper implementing CryptoBroker protocol."""

import logging
from typing import Any

from binance import AsyncClient, BinanceAPIException

from halal_trader.domain.models import CryptoAccount, CryptoBalance, Kline

logger = logging.getLogger(__name__)

# Binance testnet base URLs
_TESTNET_API_URL = "https://testnet.binance.vision/api"
_TESTNET_WS_URL = "wss://testnet.binance.vision/ws"


class BinanceClient:
    """Async Binance client implementing the CryptoBroker protocol.

    Supports both testnet and production via a simple toggle.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        testnet: bool = True,
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._testnet = testnet
        self._client: AsyncClient | None = None

    async def connect(self) -> None:
        """Create and initialise the async Binance client."""
        self._client = await AsyncClient.create(
            api_key=self._api_key,
            api_secret=self._secret_key,
            testnet=self._testnet,
        )
        mode = "TESTNET" if self._testnet else "PRODUCTION"
        logger.info("Binance client connected (%s)", mode)

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

    # ── CryptoBroker protocol methods ──────────────────────────

    async def get_account(self) -> CryptoAccount:
        """Get account snapshot with USDT-equivalent balances."""
        info = await self.client.get_account()
        balances = info.get("balances", [])

        total = 0.0
        available = 0.0
        locked = 0.0

        for b in balances:
            asset = b["asset"]
            free = float(b["free"])
            lock = float(b["locked"])

            if asset == "USDT":
                total += free + lock
                available += free
                locked += lock
            elif free + lock > 0:
                # Estimate USDT value via ticker (best-effort)
                try:
                    ticker = await self.client.get_symbol_ticker(symbol=f"{asset}USDT")
                    price = float(ticker["price"])
                    total += (free + lock) * price
                    available += free * price
                    locked += lock * price
                except BinanceAPIException, KeyError:
                    pass  # Skip non-USDT-paired assets

        return CryptoAccount(
            total_balance_usdt=total,
            available_balance_usdt=available,
            in_order_usdt=locked,
        )

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

        Args:
            symbol: Trading pair, e.g. "BTCUSDT"
            side: "BUY" or "SELL"
            quantity: Amount to trade
            order_type: "MARKET" or "LIMIT"
            price: Required for LIMIT orders
        """
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": f"{quantity:.8f}",
        }

        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("price is required for LIMIT orders")
            params["price"] = f"{price:.8f}"
            params["timeInForce"] = "IOC"  # Immediate-or-cancel for fast trading

        result = await self.client.create_order(**params)
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
        ticker = await self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker["price"])
