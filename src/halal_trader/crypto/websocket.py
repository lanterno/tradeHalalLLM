"""Binance WebSocket manager — real-time kline and trade stream subscriptions."""

import asyncio
import logging
from collections import deque
from typing import Any

from binance import AsyncClient, BinanceSocketManager

from halal_trader.domain.models import Kline

logger = logging.getLogger(__name__)

# Maximum number of candles to keep per symbol in the rolling buffer.
_MAX_BUFFER_SIZE = 200


class BinanceWSManager:
    """Manages WebSocket subscriptions for real-time 1-minute klines.

    Maintains a rolling buffer of recent candles per symbol so that the
    trading cycle can read the latest data without making REST calls.
    """

    def __init__(self, client: AsyncClient, symbols: list[str]) -> None:
        self._client = client
        self._symbols = [s.lower() for s in symbols]
        self._bsm: BinanceSocketManager | None = None

        # Rolling kline buffers: symbol -> deque of Kline
        self._kline_buffers: dict[str, deque[Kline]] = {
            s.upper(): deque(maxlen=_MAX_BUFFER_SIZE) for s in self._symbols
        }

        # Latest ticker prices from the stream
        self._latest_prices: dict[str, float] = {}

        # Background tasks for each stream
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False

    async def start(self) -> None:
        """Start WebSocket streams for all configured symbols."""
        self._bsm = BinanceSocketManager(self._client)
        self._running = True

        for symbol in self._symbols:
            task = asyncio.create_task(self._kline_stream(symbol), name=f"ws-kline-{symbol}")
            self._tasks.append(task)

        logger.info(
            "WebSocket streams started for %d symbols: %s",
            len(self._symbols),
            ", ".join(s.upper() for s in self._symbols),
        )

    async def stop(self) -> None:
        """Stop all WebSocket streams."""
        self._running = False

        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks.clear()
        logger.info("WebSocket streams stopped")

    def get_klines(self, symbol: str, limit: int = 100) -> list[Kline]:
        """Get the latest klines from the buffer for a symbol.

        Args:
            symbol: Trading pair in uppercase, e.g. "BTCUSDT"
            limit: Maximum number of candles to return

        Returns:
            List of Kline objects, oldest first.
        """
        buf = self._kline_buffers.get(symbol.upper(), deque())
        items = list(buf)
        return items[-limit:] if len(items) > limit else items

    def get_latest_price(self, symbol: str) -> float | None:
        """Get the latest price from the stream for a symbol."""
        return self._latest_prices.get(symbol.upper())

    @property
    def buffer_sizes(self) -> dict[str, int]:
        """Return current buffer sizes for monitoring."""
        return {sym: len(buf) for sym, buf in self._kline_buffers.items()}

    # ── Private stream handlers ────────────────────────────────

    async def _kline_stream(self, symbol: str) -> None:
        """Subscribe to 1-minute kline stream for a single symbol."""
        sym_upper = symbol.upper()
        reconnect_delay = 1

        while self._running:
            try:
                async with self._bsm.kline_socket(symbol, interval="1m") as stream:
                    reconnect_delay = 1  # Reset on successful connection
                    logger.debug("Kline stream connected for %s", sym_upper)

                    while self._running:
                        msg = await asyncio.wait_for(stream.recv(), timeout=60)
                        if msg is None:
                            break
                        self._process_kline_msg(sym_upper, msg)

            except asyncio.CancelledError:
                break
            except asyncio.TimeoutError:
                logger.warning("Kline stream timeout for %s, reconnecting...", sym_upper)
            except Exception as e:
                if not self._running:
                    break
                logger.warning(
                    "Kline stream error for %s: %s — reconnecting in %ds",
                    sym_upper,
                    e,
                    reconnect_delay,
                )
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)  # Exponential backoff

    def _process_kline_msg(self, symbol: str, msg: dict[str, Any]) -> None:
        """Process a raw kline WebSocket message and update the buffer."""
        if msg.get("e") != "kline":
            return

        k = msg.get("k", {})
        if not k:
            return

        kline = Kline(
            open_time=int(k["t"]),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            close_time=int(k["T"]),
        )

        # Update latest price
        self._latest_prices[symbol] = kline.close

        # If the candle is closed, append to buffer; otherwise update the last entry
        is_closed = k.get("x", False)
        buf = self._kline_buffers[symbol]

        if is_closed:
            buf.append(kline)
        elif buf and buf[-1].open_time == kline.open_time:
            # Update in-progress candle
            buf[-1] = kline
        else:
            # First update of a new candle
            buf.append(kline)
