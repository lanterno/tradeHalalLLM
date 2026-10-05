"""Is the core ready for real money? Checked every evening (the G1 gate for the core).

The core claims no edge, so its gate tests mechanics, not alpha (decided
under the operator's delegation, 2026-10-04): the real paper account must
have followed the rule, cleanly, for long enough.

* **duration:** at least ``MIN_DAYS`` trading days of the core account's
  equity, including at least one executed monthly rebalance;
* **tracking:** the account's daily returns follow the ``core`` forward
  book's (the rule run on closing prices): annualised tracking error under
  ``MAX_TRACKING_ERROR`` and a cumulative gap under ``MAX_GAP``;
* **clean:** no refused order and no halted run in the last ``MIN_DAYS``.

When all hold, the evening run alerts once: the operator's step is to put
the live account's keys in place and set CORE_PAPER=false.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

MIN_DAYS = 20
MAX_TRACKING_ERROR = 0.03
MAX_GAP = 0.01


@dataclass
class Readiness:
    days: int
    monthly_runs: int
    tracking_error: float | None
    gap: float | None
    refused: int
    halted: int
    failures: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.failures


async def check(engine: AsyncEngine, *, today: date, book: str = "core") -> Readiness:
    async with engine.connect() as conn:
        equity = (
            await conn.execute(
                text(
                    "SELECT day, equity FROM broker_equity WHERE account = 'core' AND equity > 0 "
                    "ORDER BY day"
                )
            )
        ).all()
        navs = {
            r.day: float(r.nav)
            for r in await conn.execute(
                text("SELECT day, nav FROM forward_book_days WHERE book = :b"), {"b": book}
            )
        }
        since = today - timedelta(days=int(MIN_DAYS * 1.5))
        monthly = (
            await conn.execute(text("SELECT count(*) FROM core_runs WHERE monthly AND executed"))
        ).scalar() or 0
        refused = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM core_orders WHERE status <> 'submitted' "
                    "AND submitted_at >= :s"
                ),
                {"s": since},
            )
        ).scalar() or 0
        halted = (
            await conn.execute(
                text("SELECT count(*) FROM core_runs WHERE halted IS NOT NULL AND run_on >= :s"),
                {"s": since},
            )
        ).scalar() or 0

    # Days both the account and the book have, from the account's first.
    days = [r.day for r in equity if r.day in navs]
    acct = {r.day: float(r.equity) for r in equity}
    diffs = []
    for prev, day in zip(days, days[1:], strict=False):
        diffs.append((acct[day] / acct[prev] - 1) - (navs[day] / navs[prev] - 1))
    tracking = (
        math.sqrt(sum((d - sum(diffs) / len(diffs)) ** 2 for d in diffs) / (len(diffs) - 1))
        * math.sqrt(252)
        if len(diffs) > 1
        else None
    )
    gap = (
        (acct[days[-1]] / acct[days[0]]) / (navs[days[-1]] / navs[days[0]]) - 1
        if len(days) > 1
        else None
    )
    r = Readiness(len(days), int(monthly), tracking, gap, int(refused), int(halted))
    if r.days < MIN_DAYS:
        r.failures.append(f"{r.days} of {MIN_DAYS} trading days")
    if r.monthly_runs < 1:
        r.failures.append("no executed monthly rebalance yet")
    if tracking is None or tracking > MAX_TRACKING_ERROR:
        r.failures.append(
            "tracking error not yet measurable"
            if tracking is None
            else f"tracking error {tracking:.1%} > {MAX_TRACKING_ERROR:.0%}"
        )
    if gap is None or abs(gap) > MAX_GAP:
        r.failures.append(
            "cumulative gap not yet measurable" if gap is None else f"gap {gap:+.2%} vs the book"
        )
    if refused:
        r.failures.append(f"{refused} refused order(s) recently")
    if halted:
        r.failures.append(f"{halted} halted run(s) recently")
    return r
