"""SEC EDGAR client for screening: tickers, industry codes, XBRL frames.

All free, all public. EDGAR asks for a descriptive User-Agent with contact
details (EDGAR_USER_AGENT) and at most 10 requests/second.

The XBRL *frames* API returns one concept (e.g. us-gaap LongTermDebt) for
every filer in one calendar period in a single request, so screening a
1,500-name universe takes a few dozen frame requests plus one submissions
request per company for its SIC code.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_MIN_INTERVAL_S = 0.12  # under EDGAR's 10 requests/second
_FRAME_MEMO = 256
_FOREIGN_ANNUAL = {"20-F", "40-F"}
_DOMESTIC_ANNUAL = {"10-K"}  # parsed frames kept per client (~6k filers each)


@dataclass(frozen=True, slots=True)
class Fact:
    """One reported value: ``val`` as of ``end`` (instant) or for the period ending ``end``."""

    val: float
    end: date
    accn: str


@dataclass(frozen=True, slots=True)
class Company:
    cik: int
    ticker: str
    title: str


class SecClient:
    def __init__(
        self,
        user_agent: str,
        *,
        client: httpx.AsyncClient | None = None,
        min_interval_s: float = _MIN_INTERVAL_S,
    ) -> None:
        if not user_agent or "@" not in user_agent:
            raise ValueError("EDGAR requires a User-Agent with a contact e-mail (EDGAR_USER_AGENT)")
        self._client = client or httpx.AsyncClient(timeout=60.0)
        self._owns_client = client is None
        self._headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self._min_interval = min_interval_s
        self._last = 0.0
        # Per-client memos: a screening history re-reads the same quarter's
        # frames (5 per screen, 4 shared with the next) and the same SIC codes.
        self._frames: OrderedDict[tuple[str, str, str, str], dict[int, Fact]] = OrderedDict()
        self._sics: dict[int, tuple[int | None, str]] = {}
        self._foreign: dict[int, bool] = {}

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _fetch(self, url: str) -> httpx.Response | None:
        loop = asyncio.get_running_loop()
        wait = self._last + self._min_interval - loop.time()
        if wait > 0:
            await asyncio.sleep(wait)
        self._last = loop.time()
        response = await self._client.get(url, headers=self._headers)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response

    async def _get(self, url: str) -> Any:
        response = await self._fetch(url)
        return response.json() if response is not None else None

    async def text(self, url: str) -> str | None:
        response = await self._fetch(url)
        return response.text if response is not None else None

    async def submissions(self, cik: int) -> dict[str, Any] | None:
        payload = await self._get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        return payload if isinstance(payload, dict) else None

    async def companies(self) -> dict[str, Company]:
        """Ticker -> company, for every SEC registrant with a listed ticker."""
        payload = await self._get("https://www.sec.gov/files/company_tickers.json")
        out: dict[str, Company] = {}
        for row in payload.values():
            ticker = str(row["ticker"]).upper()
            out[ticker] = Company(int(row["cik_str"]), ticker, str(row["title"]))
        return out

    async def filers(self, period: str) -> dict[int, str]:
        """CIK -> current entity name of every filer reporting total assets at ``period``."""
        payload = await self._get(
            f"https://data.sec.gov/api/xbrl/frames/us-gaap/Assets/USD/{period}.json"
        )
        return {int(r["cik"]): str(r["entityName"]) for r in (payload or {}).get("data", [])}

    async def historical_names(self) -> list[tuple[str, int]]:
        """(name, CIK) for every name any EDGAR entity has filed under, former names included."""
        body = await self.text("https://www.sec.gov/Archives/edgar/cik-lookup-data.txt")
        out: list[tuple[str, int]] = []
        for line in (body or "").splitlines():
            name, _, rest = line.rstrip(":").rpartition(":")
            if name and rest.isdigit():
                out.append((name, int(rest)))
        return out

    async def sic(self, cik: int) -> tuple[int | None, str]:
        """(SIC code, description) from the company's submissions record."""
        if cik in self._sics:
            return self._sics[cik]
        payload = await self.submissions(cik)
        if not payload:
            return None, ""
        code = payload.get("sic")
        desc = str(payload.get("sicDescription") or "")
        self._sics[cik] = ((int(str(code)) if code not in (None, "") else None), desc)
        forms = set(((payload.get("filings") or {}).get("recent") or {}).get("form") or [])
        self._foreign[cik] = bool(forms & _FOREIGN_ANNUAL) and not forms & _DOMESTIC_ANNUAL
        return self._sics[cik]

    async def foreign_filer(self, cik: int) -> bool:
        """True for a foreign private issuer: annual reports on 20-F/40-F, never 10-K.

        Its XBRL share count is ordinary shares, while the US listing is
        usually an ADR worth several of them, so price x shares overstates
        its market cap by the ADR ratio.
        """
        if cik not in self._foreign:
            await self.sic(cik)
        return self._foreign.get(cik, False)

    async def frame(self, taxonomy: str, concept: str, unit: str, period: str) -> dict[int, Fact]:
        """One concept for every filer in one period: CIK -> Fact.

        ``period`` is SEC frame notation: "CY2025" (annual duration),
        "CY2025Q4" (quarterly duration) or "CY2025Q4I" (instant).
        """
        key = (taxonomy, concept, unit, period)
        if key in self._frames:
            self._frames.move_to_end(key)
            return self._frames[key]
        payload = await self._get(
            f"https://data.sec.gov/api/xbrl/frames/{taxonomy}/{concept}/{unit}/{period}.json"
        )
        facts = {
            int(row["cik"]): Fact(float(row["val"]), date.fromisoformat(row["end"]), row["accn"])
            for row in (payload or {}).get("data", [])
        }
        self._frames[key] = facts
        if len(self._frames) > _FRAME_MEMO:
            self._frames.popitem(last=False)
        return facts
