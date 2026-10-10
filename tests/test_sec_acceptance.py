"""A filing's acceptance time comes from its EDGAR index header, in New York time."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import pytest

from halal_trader.compliance.sec import (
    SecClient,
    SecUnavailable,
    accession_dashed,
    header_url,
    parse_acceptance,
)

UA = "test test@example.invalid"
AAPL_URL = (
    "https://www.sec.gov/Archives/edgar/data/320193/000032019323000104/"
    "0000320193-23-000104-index-headers.html"
)


def header_page(stamp: str, *, escaped: bool = False) -> str:
    """An index-header page as EDGAR serves it: the SGML header inside a comment."""
    lt, gt = ("&lt;", "&gt;") if escaped else ("<", ">")
    return (
        "<HTML><HEAD><TITLE>SEC EDGAR Submission 0000320193-23-000104</TITLE>\n<!--\n"
        f"{lt}SEC-HEADER{gt}0000320193-23-000104.hdr.sgml : 20231102\n"
        f"{lt}ACCEPTANCE-DATETIME{gt}{stamp}\n"
        f"{lt}ACCESSION-NUMBER{gt}0000320193-23-000104\n{lt}TYPE{gt}8-K\n-->\n</HEAD></HTML>"
    )


def _client(handler: Callable[[httpx.Request], httpx.Response], **kw: float | int) -> SecClient:
    return SecClient(
        UA,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        min_interval_s=0.0,
        **kw,  # type: ignore[arg-type]
    )


def test_the_header_time_is_new_york_time_in_summer_and_in_winter() -> None:
    # AAPL's Q4 FY2023 results: the JSON says 2023-11-03T00:30:32Z, four hours late.
    assert parse_acceptance(header_page("20231102163032")) == datetime(
        2023, 11, 2, 20, 30, 32, tzinfo=UTC
    )
    assert parse_acceptance(header_page("20240115080000")) == datetime(
        2024, 1, 15, 13, 0, 0, tzinfo=UTC
    )
    assert parse_acceptance(header_page("20231102163032", escaped=True)) == datetime(
        2023, 11, 2, 20, 30, 32, tzinfo=UTC
    )


def test_a_header_without_an_acceptance_time_reads_as_none() -> None:
    assert parse_acceptance("<SEC-HEADER>x\n<TYPE>8-K\n") is None
    assert parse_acceptance(header_page("20231332163032")) is None  # month 13


def test_accession_numbers_with_or_without_dashes() -> None:
    assert accession_dashed("000032019323000104") == "0000320193-23-000104"
    assert accession_dashed("0000320193-23-000104") == "0000320193-23-000104"
    with pytest.raises(ValueError):
        accession_dashed("a1")
    assert header_url(320193, "000032019323000104") == AAPL_URL


async def test_acceptance_reads_the_header_under_the_company_cik() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if str(request.url) == AAPL_URL:
            return httpx.Response(200, text=header_page("20231102163032"))
        return httpx.Response(404, text="Not Found")

    sec = _client(handler)
    assert await sec.acceptance(320193, "0000320193-23-000104") == datetime(
        2023, 11, 2, 20, 30, 32, tzinfo=UTC
    )
    # A filing agent's CIK has no folder for it: none, not an error.
    assert await sec.acceptance(1193125, "0000320193-23-000104") is None
    # Not an accession number: no header, and no request.
    assert await sec.acceptance(320193, "a1") is None
    assert len(seen) == 2
    assert seen[0] == AAPL_URL and "/1193125/" in seen[1]


async def test_acceptance_raises_when_edgar_stays_down() -> None:
    sec = _client(lambda r: httpx.Response(503), retries=0)
    with pytest.raises(SecUnavailable):
        await sec.acceptance(320193, "0000320193-23-000104")


async def test_concurrent_requests_still_wait_their_turn() -> None:
    """The pacer is shared: requests in flight together still start an interval apart."""
    starts: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        starts.append(asyncio.get_running_loop().time())
        return httpx.Response(200, text=header_page("20231102163032"))

    sec = SecClient(
        UA,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        min_interval_s=0.05,
    )
    await asyncio.gather(*(sec.acceptance(320193, "0000320193-23-000104") for _ in range(4)))

    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    assert len(starts) == 4
    assert min(gaps) >= 0.045, gaps
