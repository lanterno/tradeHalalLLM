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
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_MIN_INTERVAL_S = 0.12  # under EDGAR's 10 requests/second


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

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get(self, url: str) -> Any:
        loop = asyncio.get_running_loop()
        wait = self._last + self._min_interval - loop.time()
        if wait > 0:
            await asyncio.sleep(wait)
        self._last = loop.time()
        response = await self._client.get(url, headers=self._headers)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def companies(self) -> dict[str, Company]:
        """Ticker -> company, for every SEC registrant with a listed ticker."""
        payload = await self._get("https://www.sec.gov/files/company_tickers.json")
        out: dict[str, Company] = {}
        for row in payload.values():
            ticker = str(row["ticker"]).upper()
            out[ticker] = Company(int(row["cik_str"]), ticker, str(row["title"]))
        return out

    async def sic(self, cik: int) -> tuple[int | None, str]:
        """(SIC code, description) from the company's submissions record."""
        payload = await self._get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        if not payload:
            return None, ""
        code = payload.get("sic")
        return (int(code) if code not in (None, "") else None), str(
            payload.get("sicDescription") or ""
        )

    async def frame(self, taxonomy: str, concept: str, unit: str, period: str) -> dict[int, Fact]:
        """One concept for every filer in one period: CIK -> Fact.

        ``period`` is SEC frame notation: "CY2025" (annual duration),
        "CY2025Q4" (quarterly duration) or "CY2025Q4I" (instant).
        """
        payload = await self._get(
            f"https://data.sec.gov/api/xbrl/frames/{taxonomy}/{concept}/{unit}/{period}.json"
        )
        if not payload:
            return {}
        return {
            int(row["cik"]): Fact(float(row["val"]), date.fromisoformat(row["end"]), row["accn"])
            for row in payload.get("data", [])
        }
