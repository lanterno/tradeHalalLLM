"""The pre-registered Phase C signals, as event-portfolio candidates.

Each follows its entry in the plan's progress log ("Pre-registered S2
Phase C trials"); thresholds are trailing, so no candidate is chosen with
a percentile that includes the future.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.events.portfolio import Candidate
from halal_trader.events.study import Bars, entry_point

_ET = ZoneInfo("America/New_York")
TRAILING = timedelta(days=365)
PERCENTILE = 90.0
_MIN_WINDOW = 200  # trailing values needed before a percentile means anything


def above_trailing_percentile(
    scored: Sequence[tuple[datetime, str, float]], *, percentile: float = PERCENTILE
) -> list[Candidate]:
    """Keep (time, symbol, value) rows at or above the percentile of the values
    seen in the preceding year (strictly before each row)."""
    rows = sorted(scored)
    times = [r[0] for r in rows]
    out = []
    for i, (t, symbol, value) in enumerate(rows):
        lo = bisect_left(times, t - TRAILING)
        hi = bisect_left(times, t)
        window = [r[2] for r in rows[lo:hi]]
        if len(window) < _MIN_WINDOW:
            continue
        if value >= float(np.percentile(window, percentile)):
            out.append(Candidate(symbol, t, value))
    return out


# ── E1b: SUE ───────────────────────────────────────────────────


async def e1b(engine: AsyncEngine) -> list[Candidate]:
    from halal_trader.events.history import covered_companies
    from halal_trader.events.sue import sue_observations

    obs = await sue_observations(engine, await covered_companies(engine))
    return above_trailing_percentile([(o.announced_at, o.symbol, o.sue) for o in obs])


# ── E1c: announcement return ───────────────────────────────────


async def _releases(engine: AsyncEngine) -> list[tuple[str, datetime]]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT symbol, published_at FROM events "
                "WHERE kind = '8-k' AND payload->'items' ? '2.02'"
            )
        )
        return [(r.symbol, r.published_at) for r in rows]


def announcement_returns(
    releases: Sequence[tuple[str, datetime]], bars: Bars
) -> list[tuple[datetime, str, float]]:
    """(time known, symbol, abnormal return) over the release's two-session window.

    The reaction session is the one whose prices first reflect the release
    (the clock rule); the window runs from the close before it to the close
    of the session after it, against SPY. It is known at that last close,
    so its time is 16:01 New York that day: tradable from the next open.
    """
    sessions = bars.sessions
    out = []
    for symbol, published in releases:
        point = entry_point(published, sessions)
        if point is None:
            continue
        i = point[0]
        if i < 1 or i + 1 >= len(sessions):
            continue
        before, after = sessions[i - 1], sessions[i + 1]
        c = bars.close.get(symbol, {})
        b = bars.close.get("SPY", {})
        if not (c.get(before) and c.get(after) and b.get(before) and b.get(after)):
            continue
        abn = (c[after] / c[before] - 1) - (b[after] / b[before] - 1)
        known = datetime.combine(after, time(16, 1), _ET)
        out.append((known, symbol, abn))
    return out


async def e1c(engine: AsyncEngine, bars: Bars) -> list[Candidate]:
    return above_trailing_percentile(announcement_returns(await _releases(engine), bars))


# ── E3: insider purchase clusters ─────────────────────────────

CLUSTER_WINDOW = timedelta(days=30)
CLUSTER_MIN_INSIDERS = 2
CLUSTER_MIN_VALUE = 100_000.0


def clusters(buys: Sequence[tuple[datetime, str, str, float]]) -> list[Candidate]:
    """(filed at, symbol, insider, value) open-market buys -> one candidate per cluster.

    A cluster completes at the filing that brings a symbol's trailing 30 days
    to >= 2 distinct insiders and >= $100k combined; the symbol cannot signal
    again until 30 days after that.
    """
    by_symbol: dict[str, list[tuple[datetime, str, float]]] = defaultdict(list)
    for t, symbol, insider, value in buys:
        by_symbol[symbol].append((t, insider, value))
    out = []
    for symbol, rows in by_symbol.items():
        rows.sort()
        quiet_until: datetime | None = None
        for i, (t, _, _) in enumerate(rows):
            if quiet_until is not None and t < quiet_until:
                continue
            window = [r for r in rows[: i + 1] if r[0] > t - CLUSTER_WINDOW]
            insiders = {r[1] for r in window}
            total = sum(r[2] for r in window)
            if len(insiders) >= CLUSTER_MIN_INSIDERS and total >= CLUSTER_MIN_VALUE:
                out.append(Candidate(symbol, t, total))
                quiet_until = t + CLUSTER_WINDOW
    return out


async def e3(engine: AsyncEngine) -> list[Candidate]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT symbol, published_at, payload FROM events WHERE kind = 'insider_buy'")
        )
        buys = []
        for r in rows:
            p = r.payload or {}
            roles = " ".join(p.get("relationship") or [])
            if p.get("plan_10b5_1") or not ("Director" in roles or "Officer" in roles):
                continue
            value = p.get("value") or 0.0
            for owner in p.get("owners") or ["?"]:
                buys.append((r.published_at, r.symbol, owner, float(value)))
                break  # one insider per transaction row
    return clusters(buys)
