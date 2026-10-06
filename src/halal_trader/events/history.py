"""The event store's history (docs/EVENT_DRIVEN_ROADMAP.md, Phase A).

Four resumable backfills, each recording what it finished in
``backfill_progress`` so an interrupted run picks up where it stopped:

* **news** -- every Alpaca (Benzinga) article from 2016, one day at a time,
  newest first (the post-cutoff years are the ones LLM tests can use, so
  they land first), kept for the symbols the screen has ever covered;
* **filings** -- every 8-K, 10-Q and 10-K of the companies the screen has
  ever covered, timestamped by SEC acceptance to the second; an 8-K's
  item codes say what it is (2.02 = results of operations);
* **insiders** -- open-market purchases and sales (Form 4 codes P and S)
  from SEC's quarterly insider-transaction data sets;
* **eps** -- quarterly and annual EPS facts from XBRL with the date each
  was filed, the input to standardized unexpected earnings.

Point in time: news and filings carry their source's timestamp. The
insider data sets carry a filing *date* only, so a Form 4 is stamped
17:00 New York on that date: usable from the next session, never earlier.
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from collections.abc import Iterable
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.sec import SecClient
from halal_trader.events.store import EventRecord, EventRecorder

logger = logging.getLogger(__name__)

NEWS_FROM = date(2016, 1, 1)
FILINGS_FROM = date(2015, 6, 1)
FILING_FORMS = {"8-K", "8-K/A", "10-Q", "10-K"}
_ET = ZoneInfo("America/New_York")
_INSIDER_URL = (
    "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"
)
_EPS_CONCEPTS = ("EarningsPerShareDiluted", "EarningsPerShareBasic")


# ── progress ──────────────────────────────────────────────────


async def _done(engine: AsyncEngine, task: str) -> set[str]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT unit FROM backfill_progress WHERE task = :t"), {"t": task}
        )
        return {r.unit for r in rows}


async def _mark(engine: AsyncEngine, task: str, unit: str, items: int) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO backfill_progress (task, unit, items, done_at) "
                "VALUES (:t, :u, :n, now()) ON CONFLICT (task, unit) "
                "DO UPDATE SET items = EXCLUDED.items, done_at = EXCLUDED.done_at"
            ),
            {"t": task, "u": unit, "n": items},
        )


# ── coverage ──────────────────────────────────────────────────


async def covered_symbols(engine: AsyncEngine) -> set[str]:
    """Every symbol the screen has ever covered (the point-in-time universes)."""
    async with engine.connect() as conn:
        rows = await conn.execute(text("SELECT DISTINCT symbol FROM halal_screen_results"))
        return {r.symbol for r in rows}


async def covered_companies(engine: AsyncEngine) -> dict[int, str]:
    """CIK -> the symbol the screen covered it under most often (one per company)."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT DISTINCT ON (cik) cik, symbol FROM ("
                "  SELECT cik, symbol, count(*) AS n FROM halal_screen_current "
                "  WHERE cik IS NOT NULL GROUP BY cik, symbol"
                ") c ORDER BY cik, n DESC, symbol"
            )
        )
        return {int(r.cik): r.symbol for r in rows}


# ── news ──────────────────────────────────────────────────────


def _days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def news_records(articles: Iterable[Any], symbols: set[str]) -> list[EventRecord]:
    """One event per (article, covered symbol it names)."""
    out = []
    for a in articles:
        for sym in a.symbols:
            if sym not in symbols:
                continue
            out.append(
                EventRecord(
                    source="alpaca",
                    source_id=f"alpaca:{a.id}",
                    kind="news",
                    symbol=sym,
                    published_at=a.created_at,
                    seen_at=a.created_at,
                    payload={
                        "headline": a.headline,
                        "summary": a.summary[:500],
                        "url": a.url,
                        "publisher": a.source,
                        "symbols": list(a.symbols),
                        "backfill": True,
                    },
                )
            )
    return out


async def backfill_news(
    engine: AsyncEngine,
    market: Any,
    *,
    symbols: set[str],
    start: date = NEWS_FROM,
    end: date | None = None,
) -> int:
    """Store every covered article from ``start`` to ``end`` (default: yesterday, UTC)."""
    end = end or datetime.now(UTC).date() - timedelta(days=1)
    done = await _done(engine, "news")
    recorder = EventRecorder(engine, raise_errors=True)
    stored = 0
    for day in reversed(_days(start, end)):
        if day.isoformat() in done:
            continue
        t0 = datetime.combine(day, time(0), UTC)
        articles = await market.news(
            None, start=t0, end=t0 + timedelta(days=1) - timedelta(seconds=1), max_pages=500
        )
        written = await recorder.record(news_records(articles, symbols))
        await _mark(engine, "news", day.isoformat(), written)
        stored += written
        logger.info("news %s: %d articles, %d events", day, len(articles), written)
    return stored


# ── filings ───────────────────────────────────────────────────


def filing_records(payload: dict[str, Any], symbol: str) -> list[EventRecord]:
    """Events for the 8-K/10-Q/10-K rows of one submissions page (``filings.recent`` shape)."""
    out = []
    forms = payload.get("form") or []
    for i, form in enumerate(forms):
        if form not in FILING_FORMS:
            continue
        filed = date.fromisoformat(payload["filingDate"][i])
        if filed < FILINGS_FROM:
            continue
        accepted = str(payload["acceptanceDateTime"][i] or "")
        published = (
            datetime.fromisoformat(accepted.replace("Z", "+00:00"))
            if accepted
            else datetime.combine(filed, time(17), _ET).astimezone(UTC)
        )
        items = [x for x in str((payload.get("items") or [""] * len(forms))[i]).split(",") if x]
        out.append(
            EventRecord(
                source="sec",
                source_id=str(payload["accessionNumber"][i]),
                kind=form.lower(),
                symbol=symbol,
                published_at=published,
                seen_at=published,
                payload={
                    "form": form,
                    "items": items,
                    "report_date": (payload.get("reportDate") or [""] * len(forms))[i] or None,
                    "primary_doc": (payload.get("primaryDocument") or [""] * len(forms))[i],
                    "backfill": True,
                },
            )
        )
    return out


async def backfill_filings(engine: AsyncEngine, sec: SecClient, companies: dict[int, str]) -> int:
    done = await _done(engine, "filings")
    recorder = EventRecorder(engine, raise_errors=True)
    stored = 0
    for cik, symbol in sorted(companies.items()):
        if str(cik) in done:
            continue
        subs = await sec.submissions(cik)
        records: list[EventRecord] = []
        if subs:
            records += filing_records(subs["filings"]["recent"], symbol)
            for page in subs["filings"].get("files") or []:
                if str(page.get("filingTo", "9999")) < FILINGS_FROM.isoformat():
                    continue
                older = await sec._get(f"https://data.sec.gov/submissions/{page['name']}")
                if older:
                    records += filing_records(older, symbol)
        written = await recorder.record(records)
        await _mark(engine, "filings", str(cik), written)
        stored += written
    logger.info("filings: %d events", stored)
    return stored


# ── insiders ──────────────────────────────────────────────────


def quarters(start: date, end: date) -> list[str]:
    out = []
    for year in range(start.year, end.year + 1):
        for q in range(1, 5):
            first = date(year, 3 * q - 2, 1)
            if start <= first <= end:
                out.append(f"{year}q{q}")
    return out


def _tsv(zf: zipfile.ZipFile, name: str) -> csv.DictReader[str]:
    return csv.DictReader(
        io.TextIOWrapper(zf.open(name), encoding="utf-8", errors="replace"), delimiter="\t"
    )


def _f(value: str) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def insider_records(blob: bytes, companies: dict[int, str]) -> list[EventRecord]:
    """Open-market purchases and sales (codes P, S) of covered companies from one data set."""
    zf = zipfile.ZipFile(io.BytesIO(blob))
    subs: dict[str, dict[str, str]] = {}
    for row in _tsv(zf, "SUBMISSION.tsv"):
        try:
            cik = int(row["ISSUERCIK"])
        except ValueError:
            continue
        if cik in companies and row["DOCUMENT_TYPE"] in ("4", "4/A"):
            subs[row["ACCESSION_NUMBER"]] = row
    owners: dict[str, list[dict[str, str]]] = {}
    for row in _tsv(zf, "REPORTINGOWNER.tsv"):
        if row["ACCESSION_NUMBER"] in subs:
            owners.setdefault(row["ACCESSION_NUMBER"], []).append(row)
    out = []
    for row in _tsv(zf, "NONDERIV_TRANS.tsv"):
        acc = row["ACCESSION_NUMBER"]
        code = row["TRANS_CODE"]
        if acc not in subs or code not in ("P", "S"):
            continue
        sub = subs[acc]
        filed = datetime.strptime(sub["FILING_DATE"], "%d-%b-%Y").date()
        published = datetime.combine(filed, time(17), _ET).astimezone(UTC)
        shares, price = _f(row["TRANS_SHARES"]), _f(row["TRANS_PRICEPERSHARE"])
        people = owners.get(acc, [])
        out.append(
            EventRecord(
                source="sec_form4",
                source_id=f"{acc}:{row['NONDERIV_TRANS_SK']}",
                kind="insider_buy" if code == "P" else "insider_sell",
                symbol=companies[int(sub["ISSUERCIK"])],
                published_at=published,
                seen_at=published,
                payload={
                    "accession": acc,
                    "trans_date": row["TRANS_DATE"],
                    "shares": shares,
                    "price": price,
                    "value": shares * price if shares and price else None,
                    "owned_after": _f(row["SHRS_OWND_FOLWNG_TRANS"]),
                    "direct": row["DIRECT_INDIRECT_OWNERSHIP"] == "D",
                    "owners": [o["RPTOWNERCIK"] for o in people],
                    "relationship": sorted({o["RPTOWNER_RELATIONSHIP"] for o in people}),
                    "titles": sorted({o["RPTOWNER_TITLE"] for o in people if o["RPTOWNER_TITLE"]}),
                    "plan_10b5_1": sub.get("AFF10B5ONE") in ("1", "true", "True"),
                    "backfill": True,
                },
            )
        )
    return out


async def backfill_insiders(
    engine: AsyncEngine,
    sec: SecClient,
    companies: dict[int, str],
    *,
    start: date = date(2016, 1, 1),
    end: date | None = None,
) -> int:
    end = end or datetime.now(UTC).date()
    done = await _done(engine, "insiders")
    recorder = EventRecorder(engine, raise_errors=True)
    stored = 0
    for q in quarters(start, end):
        if q in done:
            continue
        response = await sec._fetch(_INSIDER_URL.format(q=q))
        if response is None:  # not published yet
            continue
        written = await recorder.record(insider_records(response.content, companies))
        await _mark(engine, "insiders", q, written)
        stored += written
        logger.info("insiders %s: %d events", q, written)
    return stored


# ── EPS facts ─────────────────────────────────────────────────


def eps_rows(cik: int, concept: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for fact in (payload.get("units") or {}).get("USD/shares", []):
        if "start" not in fact or "filed" not in fact:
            continue
        rows.append(
            {
                "cik": cik,
                "concept": concept,
                "start": date.fromisoformat(fact["start"]),
                "end": date.fromisoformat(fact["end"]),
                "val": float(fact["val"]),
                "form": str(fact.get("form") or ""),
                "fp": fact.get("fp"),
                "fy": fact.get("fy"),
                "filed": date.fromisoformat(fact["filed"]),
                "accn": str(fact["accn"]),
            }
        )
    return rows


async def backfill_eps(engine: AsyncEngine, sec: SecClient, companies: dict[int, str]) -> int:
    done = await _done(engine, "eps")
    stored = 0
    for cik in sorted(companies):
        if str(cik) in done:
            continue
        rows: list[dict[str, Any]] = []
        for concept in _EPS_CONCEPTS:
            payload = await sec._get(
                f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{concept}.json"
            )
            if payload:
                rows += eps_rows(cik, concept, payload)
        if rows:
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        'INSERT INTO eps_facts (cik, concept, start, "end", val, form, fp, fy, '
                        "filed, accn) VALUES (:cik, :concept, :start, :end, :val, :form, :fp, "
                        ":fy, :filed, :accn) ON CONFLICT DO NOTHING"
                    ),
                    rows,
                )
        await _mark(engine, "eps", str(cik), len(rows))
        stored += len(rows)
    logger.info("eps facts: %d rows", stored)
    return stored
