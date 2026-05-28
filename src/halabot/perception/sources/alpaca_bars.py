"""Alpaca bar source — emits observation.bar for the halal universe.

Polls the Alpaca MCP client for recent bars across the universe and emits one
``observation.bar`` per *new* bar (deduped by ``asset:bar_time``). On the first
poll the whole recent window is emitted (bootstrapping the buffer so momentum
works immediately); later polls emit only freshly-printed bars. Read-only.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from halabot.perception.poll import PollingSource
from halabot.platform.clock import Clock
from halabot.platform.events import Event, EventType, new_event

logger = logging.getLogger(__name__)

UniverseProvider = Callable[[], Awaitable[list[str]]]


class AlpacaBarSource(PollingSource):
    def __init__(
        self,
        mcp: Any,
        universe: UniverseProvider,
        clock: Clock,
        *,
        timeframe: str = "1Hour",
        days: int = 5,
        interval_s: float = 900.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        super().__init__("alpaca-bars", interval_s=interval_s, sleep=sleep)
        self._mcp = mcp
        self._universe = universe
        self._clock = clock
        self._tf = timeframe
        self._days = days

    async def fetch(self) -> list[Any]:
        symbols = await self._universe()
        out: list[dict[str, Any]] = []
        for sym in symbols:
            try:
                resp = await self._mcp.get_stock_bars(sym, days=self._days, timeframe=self._tf)
            except Exception as exc:  # noqa: BLE001 — one symbol's failure skips it, not the batch
                logger.warning("alpaca-bars fetch failed for %s: %r", sym, exc)
                continue
            for bar in _extract_bars(resp, sym):
                out.append({"_asset": sym, **bar})
        return out

    def to_event(self, raw: dict[str, Any]) -> Event | None:
        try:
            o = float(raw.get("o", raw.get("open", 0)))
            h = float(raw.get("h", raw.get("high", 0)))
            low = float(raw.get("l", raw.get("low", 0)))
            c = float(raw.get("c", raw.get("close", 0)))
            v = float(raw.get("v", raw.get("volume", 0)))
        except (TypeError, ValueError):
            return None
        if c <= 0:
            return None
        return new_event(
            self._clock,
            EventType.OBSERVATION_BAR,
            source="alpaca-bars",
            asset=raw["_asset"],
            payload={"o": o, "h": h, "low": low, "c": c, "v": v, "bar_ts": str(raw.get("t", ""))},
        )

    def dedup_key(self, raw: dict[str, Any]) -> str | None:
        return f"{raw['_asset']}:{raw.get('t', '')}"


def _extract_bars(resp: Any, symbol: str) -> list[dict[str, Any]]:
    """Pull one symbol's bar list out of Alpaca MCP's response.

    The live shape is ``{"bars": {"<SYMBOL>": [ {t,o,h,l,c,v,...}, ... ]}}`` —
    ``bars`` is a dict keyed by symbol, not a flat list. Also tolerates a
    ``{"result": ...}`` envelope, a flat ``{"bars": [...]}``/``{"data": [...]}``
    list, and a bare list.
    """
    if isinstance(resp, dict) and isinstance(resp.get("result"), (dict, list)):
        resp = resp["result"]
    bars: Any
    if isinstance(resp, dict):
        bars = resp.get("bars")
        if bars is None:
            bars = resp.get("data", [])
        if isinstance(bars, dict):  # the real shape: per-symbol dict of lists
            bars = bars.get(symbol, [])
    elif isinstance(resp, list):
        bars = resp
    else:
        return []
    if not isinstance(bars, list):
        return []
    return [b for b in bars if isinstance(b, dict)]
