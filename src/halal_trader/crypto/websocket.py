"""Binance WebSocket manager — real-time kline and trade stream subscriptions."""

import asyncio
import logging
import time
from collections import deque
from typing import Any

from binance import AsyncClient, BinanceSocketManager

from halal_trader.domain.models import Kline

logger = logging.getLogger(__name__)

_MAX_BUFFER_SIZE = 200


class BinanceWSManager:
    """Manages WebSocket subscriptions for real-time 1-minute klines.

    Uses a single combined/multiplex WebSocket connection for all symbols,
    falling back to per-symbol connections if the multiplex fails.
    Maintains a rolling buffer of recent candles per symbol so that the
    trading cycle can read the latest data without making REST calls.
    """

    def __init__(self, client: AsyncClient, symbols: list[str]) -> None:
        self._client = client
        self._symbols = [s.lower() for s in symbols]
        self._bsm: BinanceSocketManager | None = None

        self._kline_buffers: dict[str, deque[Kline]] = {
            s.upper(): deque(maxlen=_MAX_BUFFER_SIZE) for s in self._symbols
        }
        self._latest_prices: dict[str, float] = {}
        self._last_message_time: dict[str, float] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False

    async def start(self) -> None:
        """Start WebSocket streams for all configured symbols."""
        self._bsm = BinanceSocketManager(self._client)
        self._running = True

        if len(self._symbols) > 1:
            task = asyncio.create_task(self._combined_kline_stream(), name="ws-kline-combined")
            self._tasks.append(task)
        else:
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
        """Get the latest klines from the buffer for a symbol."""
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

    def health_status(self) -> dict[str, float]:
        """Return per-symbol staleness in seconds since last message."""
        now = time.monotonic()
        return {
            sym.upper(): now - self._last_message_time.get(sym.upper(), 0)
            if sym.upper() in self._last_message_time
            else float("inf")
            for sym in self._symbols
        }

    def check_health(self, stale_threshold: float = 120.0) -> list[str]:
        """Return list of symbols with stale data (> threshold seconds)."""
        stale = []
        status = self.health_status()
        for sym, staleness in status.items():
            if staleness > stale_threshold:
                stale.append(sym)
        if stale:
            logger.warning(
                "Stale WebSocket data for: %s",
                ", ".join(f"{s} ({status[s]:.0f}s)" for s in stale),
            )
        return stale

    # ── Private stream handlers ────────────────────────────────

    async def _combined_kline_stream(self) -> None:
        """Subscribe to a single multiplexed kline stream for all symbols."""
        streams = [f"{s}@kline_1m" for s in self._symbols]
        reconnect_delay = 1

        while self._running:
            try:
                async with self._bsm.multiplex_socket(streams) as stream:
                    reconnect_delay = 1
                    logger.debug(
                        "Combined kline stream connected for %d symbols", len(self._symbols)
                    )

                    while self._running:
                        msg = await asyncio.wait_for(stream.recv(), timeout=60)
                        if msg is None:
                            break
                        # Multiplex messages wrap the data in a "data" key
                        data = msg.get("data", msg)
                        if data.get("e") == "kline":
                            symbol = data.get("s", "").upper()
                            self._process_kline_msg(symbol, data)

            except asyncio.CancelledError:
                break
            except asyncio.TimeoutError:
                logger.warning("Combined kline stream timeout, reconnecting...")
            except Exception as e:
                if not self._running:
                    break
                logger.warning(
                    "Combined kline stream error: %s — reconnecting in %ds",
                    e,
                    reconnect_delay,
                )
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)

    async def _kline_stream(self, symbol: str) -> None:
        """Subscribe to 1-minute kline stream for a single symbol (fallback)."""
        sym_upper = symbol.upper()
        reconnect_delay = 1

        while self._running:
            try:
                async with self._bsm.kline_socket(symbol, interval="1m") as stream:
                    reconnect_delay = 1
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
                reconnect_delay = min(reconnect_delay * 2, 60)

    def _process_kline_msg(self, symbol: str, msg: dict[str, Any]) -> None:
        """Process a raw kline WebSocket message and update the buffer."""
        if msg.get("e") != "kline":
            return

        k = msg.get("k", {})
        if not k:
            return

        close_val = float(k["c"])
        if close_val <= 0:
            return

        kline = Kline(
            open_time=int(k["t"]),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=close_val,
            volume=float(k["v"]),
            close_time=int(k["T"]),
        )

        self._latest_prices[symbol] = kline.close
        self._last_message_time[symbol] = time.monotonic()

        is_closed = k.get("x", False)
        buf = self._kline_buffers.get(symbol)
        if buf is None:
            return

        if is_closed:
            buf.append(kline)
        elif buf and buf[-1].open_time == kline.open_time:
            buf[-1] = kline
        else:
            buf.append(kline)
