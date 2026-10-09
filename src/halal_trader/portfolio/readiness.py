"""Is the core ready for real money? Checked every evening (the G1 gate for the core).

The core claims no edge, so its gate tests mechanics, not alpha (decided
under the operator's delegation, 2026-10-04): the real paper account must
have followed the rule, cleanly, for long enough.

* **duration:** at least ``MIN_DAYS`` trading days of the account's equity
  alongside the ``core`` forward book's NAV;
* **a real rebalance with sells:** at least one monthly rebalance that
  traded and included sells, within the last ``MONTHLY_LOOKBACK`` trading
  days. The first rebalance of a fresh account only buys, and selling is
  half of what a rebalance has to get right (waiting for proceeds, sizing
  buys to them), so it has to have been seen;
* **tracking:** the account's daily returns follow the book's (the rule run
  on closing prices): annualised tracking error under ``MAX_TRACKING_ERROR``
  and a cumulative gap under ``MAX_GAP``. Both start from the account's
  *pre-trade* equity on its first run (the cash it held before the first
  orders), so the first day's trading cost is measured, not absorbed into
  the base;
* **every day accounted for:** a run that traded (or had nothing to do) on
  each of the last ``MIN_DAYS`` trading days (early closes included: the
  job has a 12:40 run for them);
* **clean:** in that window, no refused order, no halted run, and no order
  left unfilled or partially filled.

When all hold, the evening run alerts once. Going live is then the
operator's step (live keys, CORE_PAPER=false, the dated token);
:func:`live_gate` is what core/safeguards.py checks at that point, against
the paper record as it stood on its last day.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.market_hours import is_trading_day, trading_days_back
from halal_trader.portfolio.core_account import CORE_LIVE, CORE_PAPER

MIN_DAYS = 20
MONTHLY_LOOKBACK = 2 * MIN_DAYS  # trading days: always holds the latest monthly run
MAX_TRACKING_ERROR = 0.03
MAX_GAP = 0.01
# How long after the paper record ends the live switch may still rely on it.
MAX_PAPER_LAG = timedelta(days=14)


@dataclass
class Readiness:
    days: int
    monthly_runs: int  # monthly rebalances that traded and sold, in the lookback
    tracking_error: float | None
    gap: float | None
    refused: int
    halted: int
    unfilled: int = 0
    partial: int = 0
    missing_runs: list[date] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.failures


async def check(
    engine: AsyncEngine, *, today: date, book: str = "core", account: str = CORE_PAPER
) -> Readiness:
    """The gate for ``account`` as of ``today`` (only data up to ``today`` counts)."""
    from halal_trader.execution.ledger import equity_history
    from halal_trader.portfolio.execution_quality import report
    from halal_trader.research.forward_book import nav_series

    equity = await equity_history(engine, account, through=today)
    navs = dict(await nav_series(engine, book, through=today))
    async with engine.connect() as conn:
        runs = (
            await conn.execute(
                text(
                    "SELECT run_on, monthly, executed, halted, equity FROM core_runs "
                    "WHERE account = :a AND run_on <= :t ORDER BY run_on, recorded_at"
                ),
                {"a": account, "t": today},
            )
        ).all()
        sell_days = {
            r.d
            for r in await conn.execute(
                text(
                    "SELECT DISTINCT (submitted_at AT TIME ZONE 'America/New_York')::date AS d "
                    "FROM core_orders WHERE account = :a AND side = 'sell' "
                    "AND status = 'submitted'"
                ),
                {"a": account},
            )
        }

    # The window: the last MIN_DAYS trading days, through today once today's
    # run is in (before it, through the previous session, so a morning check
    # does not count today's run as missing).
    ran_on = {r.run_on for r in runs if r.executed}
    end = today if today in ran_on or not is_trading_day(today) else today - timedelta(days=1)
    window = trading_days_back(end, MIN_DAYS)
    since = window[0]
    lookback = trading_days_back(end, MONTHLY_LOOKBACK)[0]

    # Pre-trade equity: the first run that traded, before its orders, stands in
    # for the account on the book's last day before it.
    acct = dict(equity)
    first_run = next((r for r in runs if r.executed and r.halted is None and r.equity), None)
    if first_run is not None:
        before = [d for d in navs if d < first_run.run_on]
        if before:
            base_day = max(before)
            acct.setdefault(base_day, float(first_run.equity))
    days = sorted(d for d in acct if d in navs)
    diffs = [
        (acct[day] / acct[prev] - 1) - (navs[day] / navs[prev] - 1)
        for prev, day in zip(days, days[1:], strict=False)
    ]
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

    monthly = sum(
        1
        for r in runs
        if r.monthly
        and r.executed
        and r.halted is None
        and r.run_on >= lookback
        and r.run_on in sell_days
    )
    halted = sum(1 for r in runs if r.halted is not None and r.run_on >= since)
    # A session before the account's first ledger day could not have had a
    # run. Leaving those out never loosens the gate: MIN_DAYS counted days
    # need MIN_DAYS sessions of ledger, so by the time the duration check
    # passes the whole window lies after the account's start.
    opened = equity[0][0] if equity else None
    missing = [d for d in window if d not in ran_on and (opened is None or d >= opened)]
    async with engine.connect() as conn:
        refused = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM core_orders WHERE account = :a AND status <> 'submitted' "
                    "AND (submitted_at AT TIME ZONE 'America/New_York')::date BETWEEN :s AND :t"
                ),
                {"a": account, "s": since, "t": today},
            )
        ).scalar() or 0
    # A market order that never filled, or filled in part, is a position the
    # book holds and the account does not: as disqualifying as a refusal.
    fills = await report(engine, since, today, account=account)
    unfilled, partial = fills.count("unfilled"), fills.count("partial")

    r = Readiness(
        days=len(days),
        monthly_runs=monthly,
        tracking_error=tracking,
        gap=gap,
        refused=int(refused),
        halted=halted,
        unfilled=unfilled,
        partial=partial,
        missing_runs=missing,
    )
    if r.days < MIN_DAYS:
        r.failures.append(f"{r.days} of {MIN_DAYS} trading days")
    if monthly < 1:
        r.failures.append("no monthly rebalance with sells yet")
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
    if missing:
        r.failures.append(
            f"no run on {len(missing)} trading day(s): "
            + ", ".join(d.isoformat() for d in missing[:5])
        )
    if refused:
        r.failures.append(f"{refused} refused order(s) recently")
    if halted:
        r.failures.append(f"{halted} halted run(s) recently")
    if unfilled:
        r.failures.append(f"{unfilled} unfilled order(s) recently")
    if partial:
        r.failures.append(f"{partial} partially filled order(s) recently")
    return r


async def live_gate(engine: AsyncEngine, *, today: date) -> list[str]:
    """Why the paper record does not let the core go live (empty: it does).

    The paper account stops trading when the keys switch to live, so its gate
    is taken as of its last run: it must have passed then, and, before the
    first live run, that day must be recent.
    """
    async with engine.connect() as conn:
        last_paper = (
            await conn.execute(
                text("SELECT max(run_on) FROM core_runs WHERE account = :a AND executed"),
                {"a": CORE_PAPER},
            )
        ).scalar()
        live_runs = (
            await conn.execute(
                text("SELECT count(*) FROM core_runs WHERE account = :a AND executed"),
                {"a": CORE_LIVE},
            )
        ).scalar() or 0
    if last_paper is None:
        return ["no paper rehearsal on record: the core has never run on paper"]
    problems = []
    if not live_runs and today - last_paper > MAX_PAPER_LAG:
        problems.append(
            f"the paper record ended {last_paper}: rehearse again on paper before going live"
        )
    gate = await check(engine, today=last_paper, account=CORE_PAPER)
    problems.extend(f"paper gate as of {last_paper}: {f}" for f in gate.failures)
    return problems
