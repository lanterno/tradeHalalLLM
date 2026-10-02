"""Standardized unexpected earnings from XBRL (E1b, and the harness's known effect).

SUE for quarter q = (EPS_q - EPS_{q-4}) / std(EPS_{k} - EPS_{k-4}) over the
up-to-8 preceding quarters: the seasonal random walk of Foster, Olsen and
Shevlin (1984), the measure post-earnings drift was found with.

Point in time:

* each quarter's EPS is the **first value filed** for it (later 10-Q/10-K
  comparatives restate; the original is what the market saw);
* fiscal Q4 is rarely tagged on its own: it is the annual value minus the
  three quarters inside that fiscal year, known when the 10-K is filed;
* the event time is the earnings release, the **8-K with item 2.02**
  accepted between the quarter's end and the report's filing (the press
  release states the same EPS); without one, the report's filing date at
  17:00 New York.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

_ET = ZoneInfo("America/New_York")
_QUARTER_DAYS = (80, 100)
_YEAR_DAYS = (350, 380)
_MIN_HISTORY = 4  # differences needed for the standard deviation
_MAX_HISTORY = 8


@dataclass(frozen=True, slots=True)
class Quarter:
    end: date
    eps: float
    filed: date


@dataclass(frozen=True, slots=True)
class SueObs:
    cik: int
    symbol: str
    quarter_end: date
    announced_at: datetime
    sue: float
    eps: float


def quarterly_series(facts: list[tuple[str, date, date, float, date]]) -> list[Quarter]:
    """(concept, start, end, val, filed) facts of one company -> first-filed quarterly EPS.

    Diluted EPS is preferred; basic fills quarters diluted lacks.
    """
    first: dict[tuple[str, date, date], tuple[date, float]] = {}
    for concept, start, end, val, filed in facts:
        key = (concept, start, end)
        if key not in first or filed < first[key][0]:
            first[key] = (filed, val)
    out: dict[date, Quarter] = {}
    for concept in ("EarningsPerShareBasic", "EarningsPerShareDiluted"):  # diluted wins
        quarters = {
            (s, e): fv
            for (c, s, e), fv in first.items()
            if c == concept and _QUARTER_DAYS[0] <= (e - s).days <= _QUARTER_DAYS[1]
        }
        series = {e: Quarter(e, val, filed) for (_, e), (filed, val) in quarters.items()}
        tagged_ends = {e for (_, e) in quarters}
        for (c, s, e), (filed, val) in first.items():
            if c != concept or not _YEAR_DAYS[0] <= (e - s).days <= _YEAR_DAYS[1]:
                continue
            if e in tagged_ends:
                continue
            inside = [(qs, qe) for (qs, qe) in quarters if qs >= s - timedelta(days=7) and qe < e]
            if len(inside) == 3:
                q4 = val - sum(quarters[k][1] for k in inside)
                series[e] = Quarter(e, q4, filed)
        out.update(series)
    return sorted(out.values(), key=lambda q: q.end)


def sue_values(series: list[Quarter]) -> list[tuple[Quarter, float]]:
    """(quarter, SUE) wherever four quarters back and enough history exist."""
    out = []
    ends = [q.end for q in series]
    diffs: dict[date, float] = {}
    for i, q in enumerate(series):
        # The same quarter a year earlier: 4 positions back, and ~1 year apart.
        if i >= 4 and 350 <= (q.end - ends[i - 4]).days <= 380:
            diffs[q.end] = q.eps - series[i - 4].eps
    for i, q in enumerate(series):
        if q.end not in diffs:
            continue
        history = [diffs[p.end] for p in series[max(0, i - _MAX_HISTORY) : i] if p.end in diffs]
        if len(history) < _MIN_HISTORY:
            continue
        sd = float(np.std(history, ddof=1))
        if sd <= 1e-9:
            continue
        out.append((q, diffs[q.end] / sd))
    return out


async def sue_observations(engine: AsyncEngine, companies: dict[int, str]) -> list[SueObs]:
    """Every SUE the stored EPS facts allow, dated by its earnings release."""
    facts: dict[int, list[tuple[str, date, date, float, date]]] = defaultdict(list)
    releases: dict[str, list[datetime]] = defaultdict(list)
    async with engine.connect() as conn:
        for r in await conn.execute(
            text('SELECT cik, concept, start, "end", val, filed FROM eps_facts')
        ):
            facts[int(r.cik)].append((r.concept, r.start, r.end, float(r.val), r.filed))
        for r in await conn.execute(
            text(
                "SELECT symbol, published_at FROM events "
                "WHERE kind = '8-k' AND payload->'items' ? '2.02'"
            )
        ):
            releases[r.symbol].append(r.published_at)
    out = []
    for cik, rows in facts.items():
        symbol = companies.get(cik)
        if symbol is None:
            continue
        times = sorted(releases.get(symbol, []))
        for q, sue in sue_values(quarterly_series(rows)):
            lo = datetime.combine(q.end, time(0), UTC)
            hi = datetime.combine(q.filed + timedelta(days=1), time(23, 59), UTC)
            i = bisect_left(times, lo)
            announced = (
                times[i]
                if i < len(times) and times[i] <= hi
                else datetime.combine(q.filed, time(17), _ET).astimezone(UTC)
            )
            out.append(SueObs(cik, symbol, q.end, announced, sue, q.eps))
    return out
