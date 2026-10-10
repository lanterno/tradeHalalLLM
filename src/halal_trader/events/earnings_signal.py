"""Earnings releases as catalysts: the reactor's first rebuilt test (2026-10-09).

The reactor's test as built (events/intraday.py) failed: buying 60 s after any
positively scored in-session headline lost 0.30% the same day, net of cost.
Two flaws stood out: the scored headlines include fluff ("how much $1000
invested ten years ago would be worth"), and the richest catalysts, earnings,
mostly land outside the session, where the reactor never acts. So the first
rebuilt test reads only **earnings releases**, from the facts extracted from
their headlines (events/earnings_parse.py), and enters at their first
tradable price (events/study.py: the open after a pre-market or after-hours
release, the close for one in the session), net of cost, against SPY.

One observation per release: a company's earliest result headline on a New
York day. Its guidance counts if published within a day of it. A metric the
parser flags as not comparable is unknown: no surprise, no beat or miss.

Pre-registered (written before any result was seen):

* **signals:** ``sales`` -- the revenue surprise against consensus (revenue
  surprises are the better-documented drift); ``beat-raise`` -- +1 for each of
  a sales beat, an EPS beat and raised guidance, -1 for each miss or cut
  (-3..+3);
* **rule:** trained on 2016-2021: a signal is kept only if its top decile's
  5-day net abnormal return is positive with t >= 2; then 2022-2024 must agree
  in sign with t >= 2; 2025-2026 stays untouched for the final test.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.market_hours import MARKET_TZ

GUIDANCE_WINDOW = timedelta(days=1)
SALES_CLIP = 0.5  # a surprise beyond +/-50% is a parse or a tiny base, not information


@dataclass(frozen=True, slots=True)
class Release:
    symbol: str
    published_at: datetime
    sales_surprise: float | None
    eps_surprise: float | None
    sales_verdict: str | None
    eps_verdict: str | None
    guidance: str | None  # raises | lowers | maintains | ... (the extractor's action)

    @property
    def beat_raise(self) -> int:
        score = 0
        score += {"beat": 1, "beats": 1, "miss": -1, "misses": -1}.get(self.sales_verdict or "", 0)
        score += {"beat": 1, "beats": 1, "miss": -1, "misses": -1}.get(self.eps_verdict or "", 0)
        score += {"raises": 1, "lowers": -1, "cuts": -1}.get(self.guidance or "", 0)
        return score


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except TypeError, ValueError:
        return None


def _comparable(fields: dict[str, Any]) -> dict[str, Any]:
    """A result's fields with each metric the parser flags as not comparable
    ("May Not Compare", or not in the estimate's units) unknown: no surprise
    and no verdict, since Benzinga's word is computed from the same figures."""
    for metric in ("eps", "sales"):
        if fields.get(f"{metric}_not_comparable"):
            fields |= {f"{metric}_surprise": None, f"{metric}_verdict": None}
    return fields


async def releases(engine: AsyncEngine) -> list[Release]:
    """Every earnings release with an extracted result, earliest headline first."""
    from halal_trader.events.earnings_parse import EXTRACTOR

    results: dict[tuple[str, date], tuple[datetime, dict[str, Any]]] = {}
    guidance: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT e.symbol, e.published_at, f.kind, f.fields FROM event_facts f "
                "JOIN events e ON e.id = f.event_id WHERE f.extractor = :x "
                "AND f.kind IN ('result', 'guidance') AND e.symbol IS NOT NULL "
                "ORDER BY e.published_at"
            ),
            {"x": EXTRACTOR},
        )
        for r in rows:
            fields = _comparable(dict(r.fields))
            if r.kind == "result":
                key = (r.symbol, r.published_at.astimezone(MARKET_TZ).date())
                results.setdefault(key, (r.published_at, fields))  # the earliest headline
            elif fields.get("action"):
                guidance[r.symbol].append((r.published_at, str(fields["action"])))
    out = []
    for (symbol, _), (at, f) in sorted(results.items(), key=lambda kv: kv[1][0]):
        action = next(
            (a for t, a in guidance.get(symbol, []) if abs(t - at) <= GUIDANCE_WINDOW), None
        )
        sales = _num(f.get("sales_surprise"))
        out.append(
            Release(
                symbol,
                at,
                max(-SALES_CLIP, min(SALES_CLIP, sales)) if sales is not None else None,
                _num(f.get("eps_surprise")),
                f.get("sales_verdict"),
                f.get("eps_verdict"),
                action,
            )
        )
    return out


def signal_of(release: Release, name: str) -> float | None:
    if name == "sales":
        return release.sales_surprise
    if name == "beat-raise":
        return float(release.beat_raise)
    raise ValueError(f"unknown earnings signal {name!r}")
