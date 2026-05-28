"""Zoya compliance re-screening source — emits compliance.verdict (INV-7).

Periodically re-screens the universe through the Zoya Shariah screener and emits
one ``compliance.verdict`` per symbol. This is the engine's *own* freshness +
lapse detector, independent of the legacy daily screen: it keeps each belief's
``screened_at`` fresh (so the entry-freshness gate doesn't fail-closed) and, when
a held name's re-screen returns a real ``not_halal``/``doubtful``, the belief
worker force-exits it (Appendix D, fix R-05).

Transient-safe (INV-2): an API/transport failure maps to ``transient_error=True``
(a NO-VERDICT), which never overwrites a good prior verdict nor triggers a
lapse-exit. Verdicts are NOT deduped — re-emitting on each cadence is what keeps
``screened_at`` fresh; the long ``interval_s`` (default 6h) bounds Zoya load.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from halabot.perception.poll import PollingSource
from halabot.platform.clock import Clock
from halabot.platform.events import Event, EventType, new_event

logger = logging.getLogger(__name__)

UniverseProvider = Callable[[], Awaitable[list[str]]]
# Re-screen well inside the 24h gate TTL so verdicts never go stale mid-session.
_DEFAULT_INTERVAL_S = 6 * 3600.0


class Screener(Protocol):
    """Duck-typed Zoya screener — ``screen_stock`` returns a result dict with
    ``compliance`` (halal|not_halal|doubtful), ``detail``, optional ``error``."""

    async def screen_stock(self, symbol: str) -> dict[str, Any]: ...


class ZoyaComplianceSource(PollingSource):
    def __init__(
        self,
        screener: Screener,
        universe: UniverseProvider,
        clock: Clock,
        *,
        interval_s: float = _DEFAULT_INTERVAL_S,
        per_symbol_spacing_s: float = 0.2,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        super().__init__("zoya-compliance", interval_s=interval_s, sleep=sleep)
        self._screener = screener
        self._universe = universe
        self._clock = clock
        self._spacing = per_symbol_spacing_s

    async def fetch(self) -> list[Any]:
        symbols = await self._universe()
        out: list[dict[str, Any]] = []
        for sym in symbols:
            try:
                result = await self._screener.screen_stock(sym)
            except Exception as exc:  # noqa: BLE001 — a screen failure is a transient NO-VERDICT
                logger.warning("zoya-compliance screen failed for %s: %r", sym, exc)
                result = {
                    "compliance": "doubtful",
                    "detail": type(exc).__name__,
                    "error": True,
                }
            out.append({"_asset": sym, **result})
            if self._spacing > 0:
                await self._sleep(self._spacing)
        return out

    def to_event(self, raw: dict[str, Any]) -> Event | None:
        asset = raw.get("_asset")
        if not asset:
            return None
        status = str(raw.get("compliance") or "doubtful")
        if status not in ("halal", "not_halal", "doubtful"):
            status = "doubtful"
        return new_event(
            self._clock,
            EventType.COMPLIANCE_VERDICT,
            source="zoya-compliance",
            asset=asset,
            payload={
                "status": status,
                "detail": str(raw.get("detail") or "")[:300],
                "screening_id": None,  # legacy FK is owned by the Phase-4 execution path
                "transient_error": bool(raw.get("error", False)),
            },
        )

    # No dedup: re-emitting each cadence refreshes screened_at (entry-freshness
    # gate) and catches a status flip on a held name (lapse detection).
