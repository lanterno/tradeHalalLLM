"""Macro release-calendar source — emits ``observation.macro`` (Task B slice 1).

Activates the engine's dormant catalyst seam: upcoming scheduled macro
releases (CPI / NFP / FOMC / GDP from the FRED release calendar) flow in
as :data:`EventType.OBSERVATION_MACRO` events, one per (symbol, release)
— the router folds them into ``BeliefState.catalysts_pending`` so
``material_shift``'s imminent-catalyst branch can finally fire.

Reuse pattern (sanctioned by ``perception/sources`` docstring +
REARCHITECTURE.md L1 source table): the concrete legacy fetcher
(``halal_trader.trading.fred_catalysts.FREDReleaseCalendarSource``) is
constructed fresh in the CLI wiring and passed in through the local
duck-typed :class:`CatalystFetcher` Protocol — this module imports
nothing from the legacy package (mirrors ``zoya_compliance``).

Facts only, no interpretation (L1): the payload carries the schedule and
a static kind→impact prior; direction/consequence stays downstream.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, time, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from halabot.perception.dedup import DedupStore
from halabot.perception.poll import PollingSource
from halabot.platform.clock import Clock
from halabot.platform.events import Event, EventType, new_event

logger = logging.getLogger(__name__)

UniverseProvider = Callable[[], Awaitable[list[str]]]

# Release calendars change slowly; 30 min keeps the pending set fresh
# without hammering FRED (the legacy fetcher additionally caches 6h).
_DEFAULT_INTERVAL_S = 1800.0

# Static expected-impact prior per release kind (0..1). A prior, not a
# forecast: FOMC/CPI move everything; GDP is usually priced in. Tuned to
# clear (or not) BeliefSettings.catalyst_impact_threshold (default 0.7).
_KIND_IMPACT: dict[str, float] = {
    "fomc": 0.9,
    "cpi": 0.9,
    "nfp": 0.8,
    "gdp": 0.6,  # usually priced in — deliberately below the 0.7 threshold
}
_DEFAULT_IMPACT = 0.5

# MacroObservation.kind spelling (schemas/observations.py) is uppercase.
_KIND_NAMES: dict[str, str] = {
    "fomc": "FOMC",
    "cpi": "CPI",
    "nfp": "NFP",
    "gdp": "GDP",
}

# FRED's release calendar is DATE-only (promoted to UTC midnight upstream).
# UTC midnight = ~19:30-20:30 ET the PRIOR evening — an imminence window
# (±30 min) anchored there fires overnight and misses the actual print
# entirely (found in adversarial review, 2026-07-03). Anchor date-only
# schedules to the release's canonical ET clock time instead; zoneinfo
# handles DST. Timestamps that already carry a real time pass through.
_ET = ZoneInfo("America/New_York")
_KIND_RELEASE_TIME_ET: dict[str, time] = {
    "cpi": time(8, 30),
    "nfp": time(8, 30),
    "gdp": time(8, 30),
    "fomc": time(14, 0),
}
_DEFAULT_RELEASE_TIME_ET = time(9, 30)  # market open — neutral anchor


class CatalystFetcher(Protocol):
    """Duck-typed legacy catalyst source — ``fetch`` returns objects with
    ``symbol`` / ``kind`` / ``title`` / ``timestamp`` attributes (the
    legacy ``trading.catalysts.Catalyst`` shape), already fanned out
    per-symbol."""

    async def fetch(self, symbols: Sequence[str]) -> list[Any]: ...


class MacroCatalystSource(PollingSource):
    def __init__(
        self,
        fetcher: CatalystFetcher,
        universe: UniverseProvider,
        clock: Clock,
        *,
        interval_s: float = _DEFAULT_INTERVAL_S,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        dedup_store: DedupStore | None = None,
    ) -> None:
        super().__init__(
            "macro-catalysts", interval_s=interval_s, sleep=sleep, dedup_store=dedup_store
        )
        self._fetcher = fetcher
        self._universe = universe
        self._clock = clock

    async def fetch(self) -> list[Any]:
        symbols = await self._universe()
        if not symbols:
            return []
        return await self._fetcher.fetch(symbols)

    def to_event(self, raw: Any) -> Event | None:
        symbol = str(getattr(raw, "symbol", "") or "").upper()
        ts = getattr(raw, "timestamp", None)
        if not symbol or not isinstance(ts, datetime):
            return None
        if ts.tzinfo is None:  # defend against naive-datetime fetchers
            ts = ts.replace(tzinfo=UTC)
        kind_raw = str(getattr(raw, "kind", "") or "macro").lower()
        # Date-only schedule (UTC midnight signature) → anchor to the
        # release's canonical ET time so imminence covers the real print.
        if ts.astimezone(UTC).timetz() == time(0, 0, tzinfo=UTC):
            et_time = _KIND_RELEASE_TIME_ET.get(kind_raw, _DEFAULT_RELEASE_TIME_ET)
            ts = datetime.combine(ts.astimezone(UTC).date(), et_time, tzinfo=_ET).astimezone(UTC)
        # catalysts_pending is about UPCOMING releases; a release already
        # >1h in the past is stale calendar noise, not a pending catalyst.
        if ts < self._clock.now() - timedelta(hours=1):
            return None
        return new_event(
            self._clock,
            EventType.OBSERVATION_MACRO,
            source="macro-catalysts",
            asset=symbol,
            payload={
                "kind": _KIND_NAMES.get(kind_raw, kind_raw.upper()),
                "asset": symbol,
                "scheduled_for": ts.isoformat(),
                "expected_impact": _KIND_IMPACT.get(kind_raw, _DEFAULT_IMPACT),
                "actual": None,
                "consensus": None,
                # Not part of the MacroObservation contract, but carried so
                # the belief's Catalyst.detail stays human-readable.
                "detail": str(getattr(raw, "title", "") or "")[:200],
            },
        )

    def dedup_key(self, raw: Any) -> str | None:
        symbol = str(getattr(raw, "symbol", "") or "").upper()
        kind = str(getattr(raw, "kind", "") or "macro").lower()
        ts = getattr(raw, "timestamp", None)
        when = ts.date().isoformat() if isinstance(ts, datetime) else "?"
        return f"{symbol}:{kind}:{when}"
