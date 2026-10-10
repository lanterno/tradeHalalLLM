"""Re-screen the stored rows the index veto's old-ticker match changes.

``index_veto.held`` counts a renamed company as held when a view lists one
of its old tickers on a day that ticker still named it (SPUS's FB for Meta
in 2020). Screens stored before that rule read those holdings as
exclusions. ``affected`` finds the rows that rule changes: on every stored
screen date where a view holds an old ticker, it replays the veto and the
board rule over the date's stored rows twice, with old tickers and
without, from each row's stored market cap, SIC code and result before the
veto (``pre_veto``) and the SEC company names a screen reads
(``runner.company_map``). A row is affected when the two replays give it a
different outcome and its stored outcome is not yet the new one. An
outcome is the verdict and its reasons, the veto's naming the index that
excluded the company but not the size floor it quotes (a floor that moved
without moving a verdict is no reason to re-screen). That covers

* a company held under an old ticker: it passes, or the other index's
  exclusion replaces the one that held it (HLAL held CPAY as FLT; SPUS,
  not holding it, still excludes it);
* any name a moved size range now crosses: a company held under an old
  ticker joins its index's range and shifts the floor.

``rescreen_renamed`` re-screens exactly those rows with
``runner.run_screen``, the date's other stored rows sizing the veto
(``index_veto.Peer``) as the full run did. It is resumable: a row
re-screened to its new outcome is no longer affected.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Final, cast

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.aaoifi import ScreenResult, Verdict
from halal_trader.compliance.index_veto import (
    IndexView,
    Peer,
    apply_veto,
    held_as,
    require_board,
    views_at,
)
from halal_trader.compliance.runner import company_map, company_of, run_screen
from halal_trader.compliance.sec import Company, SecClient
from halal_trader.core.num import to_float
from halal_trader.events.renames import ticker_history

logger = logging.getLogger(__name__)

# The reasons index_veto.apply_veto and require_board give a pass they fail,
# and the part of the veto's that quotes the size floor.
VETO: Final = re.compile(r"excluded by (\w+)'s Shariah index ")
BOARD: Final = "business activity unverified: "
FLOOR: Final = re.compile(r" although within its size range \(.*\)$")

Outcome = tuple[str, tuple[str, ...]]  # verdict, reasons without the size floor


@dataclass(frozen=True, slots=True)
class Stored:
    """One stored screen row, as the replay reads it."""

    symbol: str
    verdict: Verdict
    reasons: list[str]
    market_cap: float | None
    sic: int | None


@dataclass(frozen=True, slots=True)
class Affected:
    as_of: date
    symbol: str
    stored: str  # the outcome on file (``describe``)
    replayed: str  # the outcome the replay gives with old tickers
    why: str


def outcome(verdict: str, reasons: Sequence[str]) -> Outcome:
    return verdict, tuple(FLOOR.sub("", r) for r in reasons)


def describe(o: Outcome) -> str:
    """``not_halal (SPUS)`` for an index's exclusion, ``doubtful (board)``, else the verdict."""
    verdict, reasons = o
    only = reasons[0] if len(reasons) == 1 else ""
    if m := VETO.match(only):
        return f"{verdict} ({m.group(1)})"
    if only.startswith(BOARD):
        return f"{verdict} (board)"
    return verdict


def pre_veto(row: Stored) -> ScreenResult:
    """The row as the screen had it before the veto and the board rule: a pass
    for a row that failed on one of them alone (a pass carries no reasons)."""
    only = row.reasons[0] if len(row.reasons) == 1 else ""
    passed = (
        row.verdict == "halal"
        or (row.verdict == "not_halal" and VETO.match(only) is not None)
        or (row.verdict == "doubtful" and only.startswith(BOARD))
    )
    metrics = {"market_cap": row.market_cap}
    if passed:
        return ScreenResult(row.symbol, "halal", [], metrics)
    return ScreenResult(row.symbol, row.verdict, list(row.reasons), metrics)


def replay(
    rows: Sequence[Stored], titles: Mapping[str, str], views: Sequence[IndexView]
) -> dict[str, Outcome]:
    """symbol -> outcome after the veto and the board rule, as run_screen applies them."""
    results = apply_veto([pre_veto(r) for r in rows], titles, views)
    results = require_board(results, {r.symbol: r.sic for r in rows}, titles, views)
    return {r.symbol: outcome(r.verdict, r.reasons) for r in results}


def renamed_holdings(view: IndexView) -> dict[str, str]:
    """Today's symbol -> the old ticker the view holds its company under."""
    return {s: old for s in ticker_history().olds if (old := held_as(view, s)) is not None}


async def stored_rows(engine: AsyncEngine, as_of: date) -> list[Stored]:
    """The date's current-method rows (``halal_screen_current``)."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT symbol, verdict, reasons, metrics->>'market_cap' AS cap, "
                "metrics->>'sic' AS sic FROM halal_screen_current WHERE as_of = :d "
                "ORDER BY symbol"
            ),
            {"d": as_of},
        )
        out = []
        for r in rows:
            sic = to_float(r.sic)
            out.append(
                Stored(
                    r.symbol,
                    cast(Verdict, r.verdict),
                    [str(x) for x in r.reasons or []],
                    to_float(r.cap),
                    int(sic) if sic is not None else None,
                )
            )
        return out


async def _dates(engine: AsyncEngine) -> list[date]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT DISTINCT as_of FROM halal_screen_results ORDER BY as_of")
        )
        return [r.as_of for r in rows]


def _titles(companies: Mapping[str, Company], rows: Sequence[Stored]) -> dict[str, str]:
    return {r.symbol: c.title for r in rows if (c := company_of(companies, r.symbol)) is not None}


def changes(
    as_of: date, rows: Sequence[Stored], titles: Mapping[str, str], views: Sequence[IndexView]
) -> list[Affected]:
    """The rows of one screen date whose outcome the old-ticker match changes."""
    renamed = {v.etf: renamed_holdings(v) for v in views}
    if not any(renamed.values()):
        return []
    new = replay(rows, titles, views)
    old = replay(rows, titles, [replace(v, period_end=None) for v in views])
    out = []
    for r in rows:
        on_file = outcome(r.verdict, r.reasons)
        if new[r.symbol] == old[r.symbol] or new[r.symbol] == on_file:
            continue
        holders = [
            f"{v.etf} holds it as {renamed[v.etf][r.symbol]} (holdings of {v.period_end})"
            for v in views
            if r.symbol in renamed[v.etf]
        ]
        why = "; ".join(holders) or "a renamed company moved an index's size range"
        out.append(Affected(as_of, r.symbol, describe(on_file), describe(new[r.symbol]), why))
    return out


async def _affected(engine: AsyncEngine, companies: Mapping[str, Company]) -> list[Affected]:
    out: list[Affected] = []
    for as_of in await _dates(engine):
        views = await views_at(engine, as_of)
        if not any(renamed_holdings(v) for v in views):
            continue
        rows = await stored_rows(engine, as_of)
        out.extend(changes(as_of, rows, _titles(companies, rows), views))
    return out


async def affected(sec: SecClient, engine: AsyncEngine) -> list[Affected]:
    """The stored rows whose outcome the old-ticker match changes, by date and symbol."""
    return await _affected(engine, await company_map(sec, engine))


async def rescreen_renamed(
    sec: SecClient, engine: AsyncEngine, *, dry_run: bool = False
) -> list[tuple[Affected, str | None]]:
    """Re-screen every affected row; returns each with its new outcome (None on a dry run)."""
    companies = await company_map(sec, engine)
    todo = await _affected(engine, companies)
    if dry_run:
        return [(a, None) for a in todo]
    by_date: dict[date, list[Affected]] = defaultdict(list)
    for a in todo:
        by_date[a.as_of].append(a)
    out: list[tuple[Affected, str | None]] = []
    for as_of, rows in sorted(by_date.items()):
        symbols = {a.symbol for a in rows}
        stored = await stored_rows(engine, as_of)
        titles = _titles(companies, stored)
        peers = {
            r.symbol: Peer(r.market_cap, titles.get(r.symbol, ""))
            for r in stored
            if r.symbol not in symbols
        }
        results = await run_screen(sec, engine, sorted(symbols), as_of, peers=peers)
        now = {r.symbol: describe(outcome(r.verdict, r.reasons)) for r in results}
        out.extend((a, now.get(a.symbol)) for a in rows)
        logger.info(
            "rescreen renamed %s: %s",
            as_of,
            ", ".join(f"{s} {now.get(s)}" for s in sorted(symbols)),
        )
    return out
