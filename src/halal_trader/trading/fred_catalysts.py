"""FRED-driven macro-catalyst calendar.

The St. Louis Fed's FRED API publishes scheduled release dates for
every economic series it tracks. The four releases that move U.S.
equity index volatility most are:

* **CPI** — Consumer Price Index (release_id 10), monthly.
* **NFP** — Employment Situation (release_id 50), monthly.
* **FOMC** — Federal Open Market Committee Statement (release_id 326),
  ~8x/year.
* **GDP** — Gross Domestic Product (release_id 53), quarterly + revisions.

This module turns those into :class:`Catalyst` rows the existing
``StockCatalystFeed`` consumes — so the stock cycle's ``CatalystRiskPolicy``
shrinks position sizing in the 4h window before each one without any
extra wiring.

Design choices:

* **Single async client.** The class accepts an optional injected
  ``httpx.AsyncClient`` so tests can use a transport mock; in
  production it owns one.
* **In-memory cache** keyed on ``(release_id, look_ahead_days)`` with a
  6h TTL. FRED publishes release dates well in advance — burning a
  request every cycle is wasteful and rate-limit-adjacent.
* **Graceful failure.** A 401 (bad key), 429 (rate-limited), or any
  other error degrades to ``[]`` so the cycle still runs on whatever
  other catalyst sources are wired.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from halal_trader.trading.catalysts import Catalyst

logger = logging.getLogger(__name__)


# Release IDs — verified against FRED's /releases endpoint. These are
# the four that drive U.S. index vol on the highest-impact tier.
FRED_RELEASE_IDS: dict[str, int] = {
    "cpi": 10,
    "nfp": 50,
    "fomc": 326,  # FOMC Statement (the meeting, not the minutes)
    "gdp": 53,
}

# Each FRED release maps to a Catalyst ``kind`` so the existing
# CatalystRiskPolicy.high_impact_kinds tuple can opt-in to it.
RELEASE_KINDS: dict[str, str] = {
    "cpi": "cpi",
    "nfp": "nfp",
    "fomc": "fomc",
    "gdp": "gdp",
}


_CACHE_TTL_S = 6 * 60 * 60  # 6 hours
_API_BASE = "https://api.stlouisfed.org/fred"


@dataclass
class _CacheEntry:
    fetched_at: float
    catalysts: list[Catalyst]


@dataclass
class FREDReleaseCalendarSource:
    """A :class:`CatalystSource` backed by the FRED release calendar.

    The fetch protocol matches every other source under
    ``trading.catalysts`` — call ``await source.fetch(symbols)`` and you
    get a list of :class:`Catalyst` rows.

    Macro releases aren't tied to a single symbol; we apply each one to
    *every* requested symbol so the existing ``CatalystRiskPolicy``
    fires per-symbol sizing reductions consistently. ``symbol`` is
    therefore the ticker the operator passed in, not the issuer of the
    release.
    """

    api_key: str
    look_ahead_days: int = 30
    enabled_releases: tuple[str, ...] = ("cpi", "nfp", "fomc", "gdp")
    _client: Any | None = None
    _cache: dict[str, _CacheEntry] = field(default_factory=dict)

    async def fetch(self, symbols: Sequence[str]) -> list[Catalyst]:
        if not symbols or not self.api_key:
            return []
        upcoming = await self._upcoming_release_dates()
        out: list[Catalyst] = []
        for symbol in symbols:
            sym = symbol.upper()
            for release, ts in upcoming:
                kind = RELEASE_KINDS.get(release, "macro")
                out.append(
                    Catalyst(
                        symbol=sym,
                        kind=kind,
                        title=f"{release.upper()} release",
                        timestamp=ts,
                        source="fred",
                        extra={"release": release},
                    )
                )
        return out

    async def _upcoming_release_dates(self) -> list[tuple[str, datetime]]:
        """Return [(release_name, timestamp)] for each enabled release."""
        cache_key = ",".join(sorted(self.enabled_releases)) + f":{self.look_ahead_days}"
        cached = self._cache.get(cache_key)
        if cached and (time.monotonic() - cached.fetched_at) < _CACHE_TTL_S:
            # Re-derive (release, ts) pairs from the cached Catalysts.
            seen: list[tuple[str, datetime]] = []
            for c in cached.catalysts:
                rel = str(c.extra.get("release", "")) or c.kind
                seen.append((rel, c.timestamp))
            # Cache stores per-symbol Catalysts; collapse to unique (release, ts).
            unique: dict[tuple[str, datetime], None] = {(r, t): None for r, t in seen}
            return list(unique.keys())

        out: list[tuple[str, datetime]] = []
        for release in self.enabled_releases:
            release_id = FRED_RELEASE_IDS.get(release)
            if release_id is None:
                continue
            try:
                dates = await self._fetch_release_dates(release_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("FRED fetch failed for %s: %s", release, exc)
                continue
            for d in dates:
                out.append((release, d))

        # Cache as Catalysts so we don't redo the assembly above on the next call.
        cached_catalysts = [
            Catalyst(
                symbol="*",  # placeholder; per-symbol fan-out happens in fetch()
                kind=RELEASE_KINDS.get(rel, "macro"),
                title=f"{rel.upper()} release",
                timestamp=ts,
                source="fred",
                extra={"release": rel},
            )
            for rel, ts in out
        ]
        self._cache[cache_key] = _CacheEntry(
            fetched_at=time.monotonic(), catalysts=cached_catalysts
        )
        return out

    async def _fetch_release_dates(self, release_id: int) -> list[datetime]:
        """Pull future release dates for one release_id."""
        client = await self._get_client()
        params = {
            "release_id": release_id,
            "api_key": self.api_key,
            "file_type": "json",
            "include_release_dates_with_no_data": "true",
            # FRED returns historical dates; we only want upcoming.
            "realtime_start": datetime.now(UTC).date().isoformat(),
            "realtime_end": (
                datetime.now(UTC).date() + timedelta(days=self.look_ahead_days)
            ).isoformat(),
            "sort_order": "asc",
            "limit": 50,
        }
        resp = await client.get(f"{_API_BASE}/release/dates", params=params)
        if resp.status_code != 200:
            logger.debug(
                "FRED %s returned %d: %s",
                release_id,
                resp.status_code,
                resp.text[:200],
            )
            return []
        data = resp.json()
        rows = data.get("release_dates", []) or []
        out: list[datetime] = []
        now = datetime.now(UTC)
        for row in rows:
            d = row.get("date")
            if not isinstance(d, str):
                continue
            try:
                # FRED returns YYYY-MM-DD; promote to UTC midnight.
                ts = datetime.fromisoformat(d).replace(tzinfo=UTC)
            except ValueError:
                continue
            if ts >= now:
                out.append(ts)
        return out

    async def _get_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
