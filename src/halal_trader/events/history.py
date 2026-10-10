"""The event store's history (the event-driven research plan, Phase A).

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

A filing's timestamp from the submissions JSON is hours late for about a
third of filings (``compliance/sec.py``). :func:`correct_filing_times`
rewrites the stored ones from each filing's EDGAR header, resumably
(task ``filing-times``), and the evening refresh stamps new filings from
their header before storing them.
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import zipfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Final

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from halal_trader.compliance.sec import SecClient, accession_dashed, filed_at
from halal_trader.core.num import to_float
from halal_trader.events.store import EventRecord, EventRecorder
from halal_trader.market_hours import MARKET_TZ

logger = logging.getLogger(__name__)

NEWS_FROM = date(2016, 1, 1)
FILINGS_FROM = date(2015, 6, 1)
FILING_FORMS = {"8-K", "8-K/A", "10-Q", "10-K"}
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
    await mark_units(engine, task, {unit: items})


async def mark_units(engine: AsyncEngine, task: str, items: dict[str, int]) -> None:
    """Record each unit (with its item count) as done, in one transaction."""
    if not items:
        return
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO backfill_progress (task, unit, items, done_at) "
                "VALUES (:t, :u, :n, now()) ON CONFLICT (task, unit) "
                "DO UPDATE SET items = EXCLUDED.items, done_at = EXCLUDED.done_at"
            ),
            [{"t": task, "u": u, "n": n} for u, n in items.items()],
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
        published = filed_at(filed, str(payload["acceptanceDateTime"][i] or ""))
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


# ── filing times ──────────────────────────────────────────────

# One unit per accession read from its header; items = the stored time minus
# the header's, in seconds (0: it was right). An accession EDGAR has no
# header for goes to the second task, so a rerun skips it too.
TIMES_TASK: Final = "filing-times"
TIMES_MISSING_TASK: Final = "filing-times-missing"
TIME_FORMS: Final = ("8-K", "8-K/A")
# The order 8-Ks are corrected in, by item code: structural and unclear
# ones first, then results of operations, then agreements, deals and
# management changes; every other 8-K after these, and other forms last.
ITEM_PRIORITY: Final = (
    frozenset({"4.02", "3.01", "2.04", "1.03", "2.06", "3.02", "4.01", "1.02"}),
    frozenset({"2.02"}),
    frozenset({"5.02", "1.01", "2.01", "2.03", "2.05"}),
)
TIMES_CONCURRENCY: Final = 3  # header requests in flight (the pacer keeps them < 10/s)
_TIMES_BATCH = 200  # filings written (and marked done) per transaction
_TIMES_LOG_EVERY = 1000


def filing_priority(kind: str, items: Iterable[str]) -> int:
    """A filing's place in the correction order, 0 first (``ITEM_PRIORITY``)."""
    if kind not in ("8-k", "8-k/a"):
        return len(ITEM_PRIORITY) + 1
    codes = set(items)
    return next((i for i, tier in enumerate(ITEM_PRIORITY) if codes & tier), len(ITEM_PRIORITY))


def time_delta(stored: datetime, accepted: datetime) -> int:
    """Seconds the stored time is after the header's acceptance (positive: late)."""
    return round((stored - accepted).total_seconds())


def header_stamped(record: EventRecord, accepted: datetime | None) -> EventRecord:
    """A filing record stamped at its header's acceptance time; without one, kept
    at the submissions JSON's time with ``time_source: json`` saying so."""
    if accepted is None:
        return replace(record, payload={**record.payload, "time_source": "json"})
    return replace(
        record,
        published_at=accepted,
        seen_at=accepted,
        payload={**record.payload, "time_source": "header"},
    )


async def unstored_filings(
    engine: AsyncEngine, records: Sequence[EventRecord]
) -> list[EventRecord]:
    """The filing records the store has no row for yet (same accession and symbol)."""
    if not records:
        return []
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT source_id, symbol FROM events "
                "WHERE source = 'sec' AND source_id = ANY(:ids)"
            ),
            {"ids": sorted({r.source_id for r in records})},
        )
        have = {(r.source_id, r.symbol) for r in rows}
    return [r for r in records if (r.source_id, r.symbol) not in have]


async def company_ciks(engine: AsyncEngine) -> dict[str, list[int]]:
    """Symbol -> the companies' CIKs it was screened under (most screened first),
    then any CIK ``ticker_ciks`` matched it to."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT symbol, cik, count(*) AS n FROM halal_screen_current "
                "WHERE cik IS NOT NULL GROUP BY symbol, cik "
                "UNION ALL SELECT symbol, cik, 0 AS n FROM ticker_ciks "
                "WHERE status = 'mapped' AND cik IS NOT NULL "
                "ORDER BY symbol, n DESC, cik"
            )
        )
        out: dict[str, list[int]] = {}
        for r in rows:
            ciks = out.setdefault(r.symbol, [])
            if int(r.cik) not in ciks:
                ciks.append(int(r.cik))
    return out


@dataclass(frozen=True, slots=True)
class _Filing:
    accession: str
    symbols: tuple[str, ...]
    published_at: datetime  # as stored (the earliest, if its rows disagree)
    priority: int
    unstamped: int  # its rows in the window not stamped from the header


async def _stored_filings(
    engine: AsyncEngine, start: date, end: date, forms: Sequence[str]
) -> list[_Filing]:
    """The stored filings of ``forms`` dated ``start``..``end`` (New York days), one
    per accession, in correction order: by priority, newest first within one."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT source_id, symbol, kind, published_at, payload->'items' AS items, "
                "payload->>'time_source' AS time_source "
                "FROM events WHERE source = 'sec' AND kind = ANY(:kinds) "
                "AND published_at >= :t0 AND published_at < :t1"
            ),
            {
                "kinds": sorted({f.lower() for f in forms}),
                "t0": datetime.combine(start, time(0), MARKET_TZ),
                "t1": datetime.combine(end + timedelta(days=1), time(0), MARKET_TZ),
            },
        )
        grouped: dict[str, list[Any]] = {}
        for r in rows:
            grouped.setdefault(r.source_id, []).append(r)
    filings = [
        _Filing(
            accession=acc,
            symbols=tuple(sorted({r.symbol for r in rs})),
            published_at=min(r.published_at for r in rs),
            priority=min(filing_priority(r.kind, [str(x) for x in (r.items or [])]) for r in rs),
            unstamped=sum(r.time_source != "header" for r in rs),
        )
        for acc, rs in grouped.items()
    ]
    filings.sort(key=lambda f: (f.priority, -f.published_at.timestamp(), f.accession))
    return filings


def _candidate_ciks(filing: _Filing, ciks: dict[str, list[int]]) -> list[int]:
    """Where a filing's header may be: its company's CIKs, then the accession's own
    filer (the company itself when it filed it, else a filing agent: no header)."""
    out = [c for s in filing.symbols for c in ciks.get(s, [])]
    try:
        out.append(int(accession_dashed(filing.accession)[:10]))
    except ValueError:
        pass  # not an accession number: no filer to try
    return list(dict.fromkeys(out))


@dataclass
class FilingTimes:
    """What one :func:`correct_filing_times` pass did."""

    checked: int = 0  # filings whose header time was read
    corrected: int = 0  # of those, filings stored at another time
    rows: int = 0  # event rows retimed (an accession can be stored under two symbols)
    missing: int = 0  # filings with no header under any candidate CIK
    done_before: int = 0  # filings an earlier pass finished (no request)
    # Rows of those that another path stored since at the JSON's time (a second
    # symbol), given the time their filing's corrected rows carry.
    copied: int = 0
    deltas: Counter[int] = field(default_factory=Counter)  # stored minus header, s -> filings


async def header_times(engine: AsyncEngine, accessions: Iterable[str]) -> dict[str, datetime]:
    """Accession -> its header's acceptance time, for the filings with a stored row
    stamped from it (``time_source: header``)."""
    ids = sorted(set(accessions))
    if not ids:
        return {}
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT source_id, min(published_at) AS at FROM events "
                "WHERE source = 'sec' AND source_id = ANY(:ids) "
                "AND payload->>'time_source' = 'header' GROUP BY source_id"
            ),
            {"ids": ids},
        )
        return {r.source_id: r.at for r in rows}


async def _retime(conn: AsyncConnection, times: dict[str, datetime]) -> None:
    """Stamp every row of each accession, under every symbol, at its header time."""
    if times:
        await conn.execute(
            text(
                "UPDATE events SET published_at = :t, seen_at = :t, "
                "payload = COALESCE(payload, CAST('{}' AS JSONB)) "
                "|| jsonb_build_object('time_source', 'header') "
                "WHERE source = 'sec' AND source_id = :acc AND (published_at <> :t "
                "OR seen_at <> :t OR payload->>'time_source' IS DISTINCT FROM 'header')"
            ),
            [{"acc": acc, "t": t} for acc, t in times.items()],
        )


async def write_times(
    engine: AsyncEngine,
    fixes: dict[str, tuple[datetime, int]],
    missing: Sequence[str] = (),
) -> None:
    """Stamp every row of each accession in ``fixes`` (header time, stored minus
    header in seconds) at that time, under every symbol it is stored under, and
    mark the units done, with ``missing`` in ``filing-times-missing``; in one
    transaction: a unit is done only with its rows written."""
    async with engine.begin() as conn:
        await _retime(conn, {acc: t for acc, (t, _) in fixes.items()})
        units = [{"t": TIMES_TASK, "u": acc, "n": d} for acc, (_, d) in fixes.items()]
        units += [{"t": TIMES_MISSING_TASK, "u": acc, "n": 0} for acc in missing]
        if units:
            await conn.execute(
                text(
                    "INSERT INTO backfill_progress (task, unit, items, done_at) "
                    "VALUES (:t, :u, :n, now()) ON CONFLICT (task, unit) "
                    "DO UPDATE SET items = EXCLUDED.items, done_at = EXCLUDED.done_at"
                ),
                units,
            )


async def _header_time(
    sec: SecClient, filing: _Filing, ciks: dict[str, list[int]]
) -> datetime | None:
    for cik in _candidate_ciks(filing, ciks):
        accepted = await sec.acceptance(cik, filing.accession)
        if accepted is not None:
            return accepted
    return None


async def correct_filing_times(
    engine: AsyncEngine,
    sec: SecClient,
    *,
    start: date,
    end: date,
    forms: Sequence[str] = TIME_FORMS,
    limit: int | None = None,
    concurrency: int = TIMES_CONCURRENCY,
) -> FilingTimes:
    """Restamp stored filings at the acceptance time of their EDGAR header.

    Covers the filings of ``forms`` dated ``start``..``end`` that no earlier
    pass finished, in ``ITEM_PRIORITY`` order, at most ``limit`` of them.
    Each filing's rows (``source='sec'``, ``source_id`` = the accession) get
    the header's time as ``published_at`` and ``seen_at`` and
    ``time_source: header`` in their payload; its unit in ``filing-times``
    records how far off the stored time was. A rerun skips finished units
    and rewrites nothing that is already right, except a row another path
    stored since under a second symbol at the JSON's time: it takes the time
    its filing's corrected rows carry, with no request and outside ``limit``.

    ``concurrency`` requests are in flight at once: a header takes about
    0.4 s to come back, and the client's pacer still spaces the requests
    under EDGAR's limit. A failure (EDGAR's outage: SecUnavailable) stops
    the pass after writing every header read before it.
    """
    out = FilingTimes()
    read_before = await _done(engine, TIMES_TASK)
    no_header = await _done(engine, TIMES_MISSING_TASK)
    filings = await _stored_filings(engine, start, end, forms)
    known = await header_times(
        engine, [f.accession for f in filings if f.accession in read_before and f.unstamped]
    )
    todo: list[_Filing] = []
    copies: dict[str, datetime] = {}
    for filing in filings:
        if filing.accession in no_header or (
            filing.accession in read_before and not filing.unstamped
        ):
            out.done_before += 1
        elif filing.accession in known:  # read before; a row stored late since
            out.done_before += 1
            out.copied += filing.unstamped
            copies[filing.accession] = known[filing.accession]
        else:  # never read, or read with no stamped row left to copy from
            todo.append(filing)
    if copies:
        async with engine.begin() as conn:
            await _retime(conn, copies)
    if limit is not None:
        todo = todo[: max(limit, 0)]
    if not todo:
        return out
    ciks = await company_ciks(engine)
    queue = iter(todo)  # shared: each filing goes to one worker, in order
    fixes: dict[str, tuple[datetime, int]] = {}
    missing: list[str] = []
    read = 0

    async def flush() -> None:
        nonlocal fixes, missing
        batch, gone = fixes, missing
        fixes, missing = {}, []
        await write_times(engine, batch, gone)

    async def worker() -> None:
        nonlocal read
        for filing in queue:
            accepted = await _header_time(sec, filing, ciks)
            if accepted is None:
                out.missing += 1
                missing.append(filing.accession)
            else:
                delta = time_delta(filing.published_at, accepted)
                out.checked += 1
                out.deltas[delta] += 1
                if delta:
                    out.corrected += 1
                    out.rows += len(filing.symbols)
                fixes[filing.accession] = (accepted, delta)
            read += 1
            if read % _TIMES_LOG_EVERY == 0:
                logger.info(
                    "filing times: %d/%d read, %d corrected, %d without a header",
                    read,
                    len(todo),
                    out.corrected,
                    out.missing,
                )
            if len(fixes) + len(missing) >= _TIMES_BATCH:
                await flush()

    workers = [asyncio.create_task(worker()) for _ in range(max(concurrency, 1))]
    try:
        await asyncio.gather(*workers)
    finally:
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        await flush()
    return out


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
        published = filed_at(filed)
        shares, price = to_float(row["TRANS_SHARES"]), to_float(row["TRANS_PRICEPERSHARE"])
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
                    "owned_after": to_float(row["SHRS_OWND_FOLWNG_TRANS"]),
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
