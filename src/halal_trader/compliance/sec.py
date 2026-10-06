"""SEC EDGAR client for screening: tickers, industry codes, XBRL frames.

All free, all public. EDGAR asks for a descriptive User-Agent with contact
details (EDGAR_USER_AGENT) and at most 10 requests/second.

The XBRL *frames* API returns one concept (e.g. us-gaap LongTermDebt) for
every filer in one calendar period in a single request, so screening a
1,500-name universe takes a few hundred frame requests plus one submissions
request per company for its SIC code.

Every request is retried (``_fetch``): a 429, a 5xx, a timeout or a dropped
connection waits and tries again, up to ``retries`` more times, honouring
Retry-After and never closer together than EDGAR's spacing. Before that,
one flaky response aborted the night's screen. What still fails after the
retries raises ``SecUnavailable``; the screen decides whether that sinks
the run (a frame, the ticker map) or only the one company (its SIC code).
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_MIN_INTERVAL_S = 0.12  # under EDGAR's 10 requests/second
_FRAME_MEMO = 384  # parsed frames kept per client; one screen reads about 300
_FOREIGN_ANNUAL = {"20-F", "40-F"}
_DOMESTIC_ANNUAL = {"10-K"}
_RETRIES = 4  # after the first try: five in all
_BACKOFF_S = 2.0  # 2, 4, 8, 16 s between tries, unless Retry-After asks for more
_MAX_WAIT_S = 120.0  # the longest single wait, whatever Retry-After says
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class SecUnavailable(RuntimeError):
    """EDGAR did not answer usefully after every retry."""


def retry_after(response: httpx.Response) -> float | None:
    """Seconds a Retry-After header asks for (delta-seconds or an HTTP date), if any."""
    value = (response.headers.get("Retry-After") or "").strip()
    if not value:
        return None
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except TypeError, ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max((when - datetime.now(UTC)).total_seconds(), 0.0)


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
        retries: int = _RETRIES,
        backoff_s: float = _BACKOFF_S,
    ) -> None:
        if not user_agent or "@" not in user_agent:
            raise ValueError("EDGAR requires a User-Agent with a contact e-mail (EDGAR_USER_AGENT)")
        self._client = client or httpx.AsyncClient(timeout=60.0)
        self._owns_client = client is None
        self._headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self._min_interval = min_interval_s
        self._retries = retries
        self._backoff = backoff_s
        self._last = 0.0
        # Per-client memos: a screening history re-reads the same quarter's
        # frames (5 per screen, 4 shared with the next) and the same SIC codes.
        self._frames: OrderedDict[tuple[str, str, str, str], dict[int, Fact]] = OrderedDict()
        self._sics: dict[int, tuple[int | None, str]] = {}
        self._foreign: dict[int, bool] = {}
        self._filed: dict[str, date] = {}  # accession number -> filing date

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _fetch(self, url: str) -> httpx.Response | None:
        """GET ``url``: None on a 404, the response on success, SecUnavailable when
        a retryable failure outlasts the retries. Other HTTP errors raise at once."""
        loop = asyncio.get_running_loop()
        for attempt in range(self._retries + 1):
            wait = self._last + self._min_interval - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = loop.time()
            final = attempt == self._retries
            try:
                response = await self._client.get(url, headers=self._headers)
            except httpx.TimeoutException, httpx.TransportError:
                if final:
                    raise SecUnavailable(f"{url}: no answer after {attempt + 1} tries") from None
                delay = self._backoff * 2**attempt
                logger.warning("EDGAR %s: no answer; retrying in %.0f s", url, delay)
            else:
                if response.status_code == 404:
                    return None
                if response.status_code not in _RETRY_STATUS:
                    response.raise_for_status()
                    return response
                if final:
                    raise SecUnavailable(
                        f"{url}: HTTP {response.status_code} after {attempt + 1} tries"
                    )
                delay = max(retry_after(response) or 0.0, self._backoff * 2**attempt)
                logger.warning(
                    "EDGAR %s: HTTP %d; retrying in %.0f s", url, response.status_code, delay
                )
            await asyncio.sleep(min(delay, _MAX_WAIT_S))
        raise AssertionError("unreachable")  # pragma: no cover

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
        """Ticker -> company, for every SEC registrant with a listed ticker.

        A missing or empty file raises rather than returning nothing: an
        empty map would screen every name "not an SEC registrant" and empty
        the halal universe, where a failed screen leaves the last one in force.
        """
        payload = await self._get("https://www.sec.gov/files/company_tickers.json")
        if not isinstance(payload, dict) or not payload:
            raise SecUnavailable("company_tickers.json: missing or empty")
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
        recent = (payload.get("filings") or {}).get("recent") or {}
        forms = set(recent.get("form") or [])
        self._foreign[cik] = bool(forms & _FOREIGN_ANNUAL) and not forms & _DOMESTIC_ANNUAL
        for accn, filed in zip(
            recent.get("accessionNumber") or [], recent.get("filingDate") or [], strict=False
        ):
            try:
                self._filed[str(accn)] = date.fromisoformat(str(filed))
            except ValueError:
                continue
        return self._sics[cik]

    def filed(self, accn: str) -> date | None:
        """The filing date of an accession number, if a submissions record read so far lists it.

        Comes with the SIC code (``sic``), so it costs no request. Covers the
        filer's recent filings: about a year or more for a large company.
        """
        return self._filed.get(accn)

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
        "CY2025Q4" (quarterly duration) or "CY2025Q4I" (instant). A concept
        nobody filed for the period is a 404 and reads as no facts; a frame
        EDGAR fails to serve raises (it is shared by every company screened).
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
