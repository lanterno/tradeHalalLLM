"""What prices did after each event, and whether the scores knew.

Labels are close-to-close returns from the **first close after
publication** (``d0``): a headline at 10:00 ET is labelled from that
day's close, one at 17:00 from the next session's. That leaves out the
move between publication and the close, which the reactor can partly
trade, so the labels measure the drift that is left -- the part a
days-to-weeks strategy lives on (operator decision, 2026-10-02).

``abn_ret`` subtracts SPUS over the same sessions. Labels are written
once a horizon has fully elapsed and never rewritten.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core.signal_eval import information_coefficient
from halal_trader.market_hours import MARKET_CLOSE, MARKET_TZ

logger = logging.getLogger(__name__)

HORIZONS = (1, 5, 20)
BENCHMARK = "SPUS"


def first_close_session(published: datetime, sessions: list[date]) -> int | None:
    """Index in ``sessions`` of the first close at or after ``published``."""
    local = published.astimezone(MARKET_TZ)
    i = bisect_right(sessions, local.date()) - 1
    if i >= 0 and sessions[i] == local.date() and local.time() < MARKET_CLOSE:
        return i
    j = bisect_right(sessions, local.date())
    return j if j < len(sessions) else None


async def _closes(engine: AsyncEngine, symbols: list[str]) -> dict[str, dict[date, float]]:
    out: dict[str, dict[date, float]] = defaultdict(dict)
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT symbol, day, close FROM daily_bars "
                "WHERE adjustment = 'all' AND symbol = ANY(:s)"
            ),
            {"s": symbols},
        )
        for r in rows:
            out[r.symbol][r.day] = float(r.close)
    return out


async def label_events(engine: AsyncEngine) -> int:
    """Label every event whose horizons have elapsed; returns labels written."""
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT e.id, e.symbol, e.published_at FROM events e "
                    "WHERE (SELECT count(*) FROM event_labels l WHERE l.event_id = e.id) < :n"
                ),
                {"n": len(HORIZONS)},
            )
        ).all()
        done = defaultdict(set)
        for r in await conn.execute(text("SELECT event_id, horizon FROM event_labels")):
            done[r.event_id].add(r.horizon)
    if not rows:
        return 0
    closes = await _closes(engine, sorted({r.symbol for r in rows} | {BENCHMARK}))
    bench = closes.get(BENCHMARK, {})
    sessions = sorted(bench)
    labels = []
    for r in rows:
        d0 = first_close_session(r.published_at, sessions)
        series = closes.get(r.symbol)
        if d0 is None or not series:
            continue
        for h in HORIZONS:
            if h in done[r.id] or d0 + h >= len(sessions):
                continue
            start, end = sessions[d0], sessions[d0 + h]
            if start not in series or end not in series:
                continue
            ret = series[end] / series[start] - 1.0
            bench_ret = bench[end] / bench[start] - 1.0
            labels.append({"e": r.id, "h": h, "r": ret, "a": ret - bench_ret})
    if labels:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO event_labels (event_id, horizon, ret, abn_ret, labeled_at) "
                    "VALUES (:e, :h, :r, :a, :at) ON CONFLICT DO NOTHING"
                ),
                [{**lab, "at": datetime.now(UTC)} for lab in labels],
            )
    logger.info("event labels: %d written", len(labels))
    return len(labels)


@dataclass(frozen=True, slots=True)
class ScorerReport:
    scorer: str
    horizon: int
    events: int  # one per (symbol, publication day): repackaged headlines count once
    ic: float
    above: int  # events scored at or above the threshold
    above_abn: float | None  # their mean abnormal return
    below_abn: float | None


async def report(engine: AsyncEngine, *, threshold: float = 0.85) -> list[ScorerReport]:
    """IC of each scorer against abnormal returns, one row per scorer and horizon.

    A catalyst arrives as 10-30 repackaged headlines; counting each would
    multiply the evidence. Events are collapsed to one per (scorer, symbol,
    publication day in New York) with the highest score, which is what the
    reactor acts on.
    """
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT s.scorer, e.symbol, e.published_at, l.horizon, s.score, l.abn_ret "
                    "FROM event_scores s JOIN events e ON e.id = s.event_id "
                    "JOIN event_labels l ON l.event_id = e.id"
                )
            )
        ).all()
    best: dict[tuple[str, str, date, int], tuple[float, float]] = {}
    for r in rows:
        key = (r.scorer, r.symbol, r.published_at.astimezone(MARKET_TZ).date(), r.horizon)
        if key not in best or r.score > best[key][0]:
            best[key] = (float(r.score), float(r.abn_ret))
    grouped: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for (scorer, _, _, horizon), pair in best.items():
        grouped[(scorer, horizon)].append(pair)
    out = []
    for (scorer, horizon), pairs in sorted(grouped.items()):
        scores = np.array([p[0] for p in pairs])
        abn = np.array([p[1] for p in pairs])
        hi = abn[scores >= threshold]
        lo = abn[scores < threshold]
        out.append(
            ScorerReport(
                scorer=scorer,
                horizon=horizon,
                events=len(pairs),
                ic=information_coefficient(scores, abn),
                above=len(hi),
                above_abn=float(hi.mean()) if len(hi) else None,
                below_abn=float(lo.mean()) if len(lo) else None,
            )
        )
    return out
