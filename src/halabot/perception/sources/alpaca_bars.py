"""Alpaca bar source — emits observation.bar for the halal universe.

Polls the Alpaca MCP client for recent bars across the universe and emits one
``observation.bar`` per *new* bar, oldest first: a bar is new when its time is
after the asset's high-water mark. The marks start from the event log
(:class:`~halabot.perception.watermark.PgBarWatermark`), so a restart resumes
where the last process stopped instead of publishing the whole window again;
with no history (a first run) the whole recent window is emitted, which is what
fills the buffer so momentum works immediately. Read-only.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from halabot.perception.poll import PollingSource
from halabot.perception.watermark import BarWatermark
from halabot.platform.clock import Clock, parse_iso
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
        watermark: BarWatermark | None = None,
    ) -> None:
        super().__init__("alpaca-bars", interval_s=interval_s, sleep=sleep)
        self._mcp = mcp
        self._universe = universe
        self._clock = clock
        self._tf = timeframe
        self._days = days
        self._watermark = watermark
        self._marks: dict[str, datetime] = {}  # newest emitted bar time per asset

    async def _prime(self) -> None:
        first = not self._primed
        await super()._prime()
        if not first or self._watermark is None:
            return
        try:
            self._marks = {a: _aware(ts) for a, ts in (await self._watermark.load()).items()}
        except Exception as exc:  # noqa: BLE001 — without marks the in-memory dedup still runs
            logger.warning("alpaca-bars watermark load failed: %r", exc)
            return
        logger.info("alpaca-bars resuming after the logged bars of %d assets", len(self._marks))

    async def fetch(self) -> list[Any]:
        symbols = await self._universe()
        out: list[dict[str, Any]] = []
        for sym in symbols:
            try:
                resp = await self._mcp.get_stock_bars(sym, days=self._days, timeframe=self._tf)
            except Exception as exc:  # noqa: BLE001 — one symbol's failure skips it, not the batch
                logger.warning("alpaca-bars fetch failed for %s: %r", sym, exc)
                continue
            mark = self._marks.get(sym)
            fresh: list[tuple[datetime | None, dict[str, Any]]] = []
            for bar in _extract_bars(resp, sym):
                ts = _bar_time(bar)
                if mark is not None and ts is not None and ts <= mark:
                    continue  # already published (by this process or the last one)
                fresh.append((ts, bar))
            # Oldest first: the buffer only accepts a bar newer than its last.
            if all(ts is not None for ts, _ in fresh):
                fresh.sort(key=lambda pair: pair[0] or datetime.min.replace(tzinfo=UTC))
            out.extend({"_asset": sym, **bar} for _, bar in fresh)
        return out

    def emitted(self, raw: dict[str, Any]) -> None:
        ts = _bar_time(raw)
        asset = raw["_asset"]
        if ts is not None and (asset not in self._marks or ts > self._marks[asset]):
            self._marks[asset] = ts

    def to_event(self, raw: dict[str, Any]) -> Event | None:
        try:
            o = float(raw.get("o", raw.get("open", 0)))
            h = float(raw.get("h", raw.get("high", 0)))
            low = float(raw.get("l", raw.get("low", 0)))
            c = float(raw.get("c", raw.get("close", 0)))
            v = float(raw.get("v", raw.get("volume", 0)))
        except TypeError, ValueError:
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


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


def _bar_time(bar: dict[str, Any]) -> datetime | None:
    ts = parse_iso(bar.get("t"))
    return _aware(ts) if ts is not None else None


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
