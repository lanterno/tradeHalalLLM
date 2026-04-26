"""SEC EDGAR 8-K material-event catalyst source.

Public companies file an 8-K within 4 business days of any material
event — acquisitions, executive departures, FDA decisions, profit
warnings, restructuring, dividends, accounting irregularities, etc.
The filings hit EDGAR within minutes and are free + unauthenticated;
they predate most retail news aggregators by an hour or more.

This source pulls each watched ticker's recent 8-Ks and emits one
:class:`Catalyst` per filing. The cycle's ``CatalystRiskPolicy`` then
shrinks position sizing in the 4h window after a material event lands.

Design choices:

* **Ticker → CIK map** is loaded once from
  ``https://www.sec.gov/files/company_tickers.json`` and cached for
  24h. The map is small (~10k rows) and changes rarely.
* **Per-CIK submissions endpoint**: ``data.sec.gov/submissions/CIK
  {cik:010d}.json`` returns the company's recent filings. We keep the
  last 100, filter for 8-K, and emit Catalysts for those filed in the
  last ``look_back_hours`` (default 24).
* **6h in-memory cache** per (ticker, look_back) so we don't hit
  EDGAR more than once every 6 hours per symbol.
* **User-Agent enforcement.** SEC will 403 a request without a real
  contact. Empty user_agent → fetch returns []; the cycle degrades
  cleanly.
* **Item-classified kind.** 8-Ks have item numbers (1.01, 2.02, 5.02,
  8.01, …); the ``kind`` field encodes "8-k:<item>" so the prompt can
  show the operator which kind of event it was.
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


_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

_FILINGS_CACHE_TTL_S = 6 * 60 * 60  # 6 hours
_TICKER_MAP_TTL_S = 24 * 60 * 60  # 24 hours

# A small lookup that turns 8-K item numbers into a short label so the
# prompt can flag the most-actionable event types directly. The full
# 8-K item list runs longer but these are the high-impact ones we care
# about for a day-trading horizon.
ITEM_LABELS: dict[str, str] = {
    "1.01": "material agreement entered",
    "1.02": "material agreement terminated",
    "2.01": "completion of acquisition",
    "2.02": "results of operations (earnings)",
    "2.03": "material direct financial obligation",
    "2.04": "triggering events on debt",
    "2.05": "material costs (restructuring)",
    "2.06": "material impairments",
    "3.01": "delisting / failure to comply",
    "3.02": "unregistered equity sale",
    "4.01": "auditor changed",
    "4.02": "previous financials non-reliance",
    "5.02": "executive departure / appointment",
    "5.03": "amendments to charter / bylaws",
    "5.07": "shareholder vote results",
    "7.01": "regulation FD disclosure",
    "8.01": "other events",
}


@dataclass
class _CacheEntry:
    fetched_at: float
    catalysts: list[Catalyst]


@dataclass
class EDGAREightKSource:
    """A :class:`CatalystSource` over the SEC EDGAR 8-K feed.

    Required: a non-empty ``user_agent`` (SEC contact). When unset,
    ``fetch`` quietly returns ``[]`` and the cycle continues on its
    other catalyst sources.
    """

    user_agent: str
    look_back_hours: int = 24
    _client: Any | None = None
    _ticker_map: dict[str, str] = field(default_factory=dict)  # symbol(upper) → cik(10-digit)
    _ticker_map_fetched_at: float = 0.0
    _filings_cache: dict[str, _CacheEntry] = field(default_factory=dict)

    async def fetch(self, symbols: Sequence[str]) -> list[Catalyst]:
        if not symbols or not self.user_agent:
            return []
        try:
            await self._ensure_ticker_map()
        except Exception as exc:  # noqa: BLE001
            logger.debug("EDGAR ticker-map fetch failed: %s", exc)
            return []

        out: list[Catalyst] = []
        for symbol in symbols:
            sym = symbol.upper()
            cik = self._ticker_map.get(sym)
            if cik is None:
                continue
            try:
                cats = await self._fetch_8k_for_cik(sym, cik)
            except Exception as exc:  # noqa: BLE001
                logger.debug("EDGAR 8-K fetch failed for %s: %s", sym, exc)
                continue
            out.extend(cats)
        return out

    # ── Ticker map ────────────────────────────────────────────────

    async def _ensure_ticker_map(self) -> None:
        if (
            self._ticker_map
            and (time.monotonic() - self._ticker_map_fetched_at) < _TICKER_MAP_TTL_S
        ):
            return
        client = await self._get_client()
        resp = await client.get(_TICKERS_URL)
        if resp.status_code != 200:
            raise RuntimeError(f"ticker map returned {resp.status_code}")
        # SEC ships the table as ``{"0": {"cik_str": ..., "ticker": ..., "title": ...}, ...}``
        data = resp.json()
        out: dict[str, str] = {}
        for row in data.values():
            ticker = str(row.get("ticker", "")).upper()
            cik = row.get("cik_str")
            if not ticker or cik is None:
                continue
            out[ticker] = f"{int(cik):010d}"
        self._ticker_map = out
        self._ticker_map_fetched_at = time.monotonic()

    # ── Per-CIK filings ───────────────────────────────────────────

    async def _fetch_8k_for_cik(self, symbol: str, cik: str) -> list[Catalyst]:
        cache_key = f"{symbol}:{self.look_back_hours}"
        cached = self._filings_cache.get(cache_key)
        if cached and (time.monotonic() - cached.fetched_at) < _FILINGS_CACHE_TTL_S:
            return list(cached.catalysts)

        client = await self._get_client()
        resp = await client.get(_SUBMISSIONS_URL.format(cik=cik))
        if resp.status_code != 200:
            logger.debug("EDGAR submissions %s returned %d", cik, resp.status_code)
            return []
        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", []) or []
        dates = recent.get("filingDate", []) or []
        primaries = recent.get("primaryDocument", []) or []
        accessions = recent.get("accessionNumber", []) or []
        items_lists = recent.get("items", []) or []

        cutoff = datetime.now(UTC) - timedelta(hours=self.look_back_hours)
        out: list[Catalyst] = []
        for i, form in enumerate(forms):
            if form != "8-K":
                continue
            ts = _parse_filing_date(dates[i] if i < len(dates) else "")
            if ts is None or ts < cutoff:
                continue
            items_str = items_lists[i] if i < len(items_lists) else ""
            items = [it.strip() for it in str(items_str).split(",") if it.strip()]
            label = _summarize_items(items)
            kind = f"8-k:{items[0]}" if items else "8-k"
            accession = accessions[i] if i < len(accessions) else ""
            primary = primaries[i] if i < len(primaries) else ""
            if accession and primary:
                acc = accession.replace("-", "")
                url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{primary}"
            else:
                url = ""
            out.append(
                Catalyst(
                    symbol=symbol,
                    kind=kind,
                    title=f"8-K filed: {label}" if label else "8-K filed",
                    timestamp=ts,
                    source="edgar",
                    url=url,
                    extra={"items": items, "accession": accession},
                )
            )
        self._filings_cache[cache_key] = _CacheEntry(
            fetched_at=time.monotonic(), catalysts=list(out)
        )
        return out

    # ── HTTP client ───────────────────────────────────────────────

    async def _get_client(self) -> Any:
        if self._client is None:
            import httpx

            # SEC requires a User-Agent with contact info; failing that
            # they 403 every request. Set Accept-Encoding too so they
            # don't gzip-bomb us with a misconfigured client.
            self._client = httpx.AsyncClient(
                timeout=10.0,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "application/json, application/atom+xml",
                    "Accept-Encoding": "gzip, deflate",
                },
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._client = None


# ── Helpers ───────────────────────────────────────────────────────


def _parse_filing_date(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).replace(tzinfo=UTC)
    except ValueError:
        return None


def _summarize_items(items: list[str]) -> str:
    """Turn a list of 8-K item numbers into a short human label."""
    if not items:
        return ""
    parts: list[str] = []
    for it in items[:3]:  # keep prompt-cheap: at most 3 items per filing
        # EDGAR sometimes prefixes items as "Item 2.02"; strip that.
        cleaned = it.replace("Item", "").strip()
        label = ITEM_LABELS.get(cleaned, cleaned)
        parts.append(label)
    return "; ".join(parts)
