"""AAOIFI compliance summary — the Halal page's and the weekly digest's source.

Per broker account (the core first; the day-trader beside it):

* trades today / this month / this quarter (New York calendar days);
* every **buy** this quarter judged against the in-house screen
  (``halal_screen_results``): the newest verdict for that symbol on or
  before the trade's date. A buy of a symbol that screen did not hold halal
  (not halal, doubtful, or never screened) is a violation;
* purification accrued and disbursed this quarter, and what is still owed
  whenever it accrued.

The day-trader's trades are the ``trades`` table; the core's orders are
``core_orders``. A sale is never a violation: selling a holding the screen
fails is the remedy, not the breach.

Read-only: pure SQL aggregations, no broker or screener calls. It judges by
the screen's record and does not re-implement the AAOIFI ratio rules
(``compliance/`` does). The day-trader's own order gate used an older
screener at the time, so a violation there says the in-house screen
disagrees with what it bought.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import and_, func, text
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import RoundTripPurificationRow
from halal_trader.portfolio.core_account import CORE_ACCOUNTS, CORE_LIVE, CORE_PAPER, DAY_TRADER

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

# The live core is recorded apart from the paper one (portfolio/core_account.py).
ACCOUNTS = (
    (CORE_LIVE, "Core portfolio (live)"),
    (CORE_PAPER, "Core portfolio"),
    (DAY_TRADER, "Day-trader"),
)
VERDICTS = ("halal", "doubtful", "not_halal", "unscreened")
# Orders that never reached the market: not trades.
_NOT_TRADED_DAY_TRADER = ("rejected", "canceled", "cancelled", "expired")
_NOT_TRADED_CORE = ("refused",)
_LISTED = 50  # non-halal buys itemised per account


@dataclass(frozen=True)
class AccountCompliance:
    """One account's quarter: what it traded and whether its buys were halal."""

    account: str
    label: str
    trades_today: int
    trades_this_month: int
    trades_this_quarter: int
    buys_this_quarter: int
    # Buys this quarter by the screen's verdict at the time ("unscreened":
    # the screen had no verdict for the symbol on or before that day).
    buy_verdicts: dict[str, int] = field(default_factory=dict)
    # The offending buys, newest first: {symbol, day, verdict, screen_as_of}.
    non_halal_buys: list[dict[str, Any]] = field(default_factory=list)

    @property
    def non_halal_buys_quarter(self) -> int:
        return sum(n for v, n in self.buy_verdicts.items() if v != "halal")

    @property
    def status(self) -> str:
        return "violation" if self.non_halal_buys_quarter else "compliant"


@dataclass(frozen=True)
class AAOIFISummary:
    """Snapshot of halal-compliance state at a point in time.

    Periods are New York calendar days: today, this month (from the 1st)
    and this quarter (from the 1st of its first month). The page renders
    the quarter.
    """

    quarter_start: date
    month_start: date
    today_start: date
    accounts: tuple[AccountCompliance, ...] = ()

    # Purification: dividend (accruals ledger, plus the legacy dividend table)
    # and capital-gains sides combined. Accrued and disbursed are this
    # quarter's; unpaid is everything accrued and not yet given away, whenever
    # it accrued, or an unpaid obligation would vanish when the quarter turns.
    purification_accrued_usd: float = 0.0
    purification_disbursed_usd: float = 0.0
    purification_unpaid_usd: float | None = None
    # The part owed by accounts trading real money. Paper accounts accrue
    # purification as a rehearsal: shown, but not a reason for "attention".
    purification_unpaid_live_usd: float | None = None
    purification_unpaid_by_account: dict[str, float] = field(default_factory=dict)

    @property
    def trades_today(self) -> int:
        return sum(a.trades_today for a in self.accounts)

    @property
    def trades_this_month(self) -> int:
        return sum(a.trades_this_month for a in self.accounts)

    @property
    def trades_this_quarter(self) -> int:
        return sum(a.trades_this_quarter for a in self.accounts)

    @property
    def non_halal_fills_quarter(self) -> int:
        """Buys this quarter of a symbol the screen did not hold halal, any account."""
        return sum(a.non_halal_buys_quarter for a in self.accounts)

    @property
    def purification_outstanding_usd(self) -> float:
        if self.purification_unpaid_usd is not None:
            return max(0.0, self.purification_unpaid_usd)
        return max(0.0, self.purification_accrued_usd - self.purification_disbursed_usd)

    @property
    def is_compliant(self) -> bool:
        """True iff no account bought a non-halal symbol this quarter."""
        return self.non_halal_fills_quarter == 0

    @property
    def status(self) -> str:
        """Operator-readable status: 'compliant' | 'attention' | 'violation'.

        * 'violation' — some account bought a symbol the screen did not hold
          halal this quarter. Renders red.
        * 'attention' — a real-money account owes purification. Amber.
        * 'compliant' — green.
        """
        if self.non_halal_fills_quarter > 0:
            return "violation"
        owed = (
            self.purification_unpaid_live_usd
            if self.purification_unpaid_live_usd is not None
            else self.purification_outstanding_usd
        )
        if owed > 0.01:
            return "attention"
        return "compliant"


def periods(today: date) -> tuple[date, date, date]:
    """(quarter start, month start, today) for a New York calendar day."""
    quarter_first_month = ((today.month - 1) // 3) * 3 + 1
    return today.replace(month=quarter_first_month, day=1), today.replace(day=1), today


def _as_day(dt: datetime) -> date:
    from halal_trader.market_hours import MARKET_TZ

    return dt.astimezone(MARKET_TZ).date()


async def compute_aaoifi_summary(
    engine: AsyncEngine,
    *,
    now: datetime | None = None,
) -> AAOIFISummary:
    """Aggregate compliance state for the Halal page and the weekly digest.

    ``now`` is for tests; production callers leave it None.
    """
    if now is None:
        now = datetime.now(UTC)
    quarter, month, today = periods(_as_day(now))

    from halal_trader.config import get_settings
    from halal_trader.portfolio.core_account import core_account

    # The configured core and the day-trader always show; the other core
    # environment only once it has traded this quarter (the paper core's
    # history the quarter it went live, say).
    shown = {core_account(get_settings().core.paper), "paper"}
    accounts = tuple(
        a
        for a in [
            await _account(engine, account, label, quarter, month, today)
            for account, label in ACCOUNTS
        ]
        if a.account in shown or a.trades_this_quarter
    )
    async with AsyncSession(engine) as session:
        q_start = datetime.combine(quarter, datetime.min.time(), UTC)
        accrued = await _sum_purification_accrued(session, q_start)
        disbursed = await _sum_purification_disbursed(session, q_start)
        ever = datetime(2000, 1, 1, tzinfo=UTC)
        by_account = await _unpaid_by_account(session, ever)
        live = _live_accounts()

    return AAOIFISummary(
        quarter_start=quarter,
        month_start=month,
        today_start=today,
        accounts=accounts,
        purification_accrued_usd=accrued,
        purification_disbursed_usd=disbursed,
        purification_unpaid_usd=sum(by_account.values()),
        purification_unpaid_live_usd=sum(v for a, v in by_account.items() if a in live),
        purification_unpaid_by_account=by_account,
    )


def _account_trades_sql(account: str) -> tuple[str, dict[str, Any]]:
    """The account's trades this quarter as (symbol, side, day), and the SQL's
    parameters: the day-trader's ``trades`` rows, the core's ``core_orders``,
    each minus what never traded."""
    if account == DAY_TRADER:
        return (
            "SELECT symbol, side, (timestamp AT TIME ZONE 'America/New_York')::date AS day "
            "FROM trades WHERE status <> ALL(:not_traded) AND timestamp >= :since",
            {"not_traded": list(_NOT_TRADED_DAY_TRADER)},
        )
    return (
        "SELECT symbol, side, (submitted_at AT TIME ZONE 'America/New_York')::date AS day "
        "FROM core_orders WHERE status <> ALL(:not_traded) AND account = :account "
        "AND submitted_at >= :since",
        {"not_traded": list(_NOT_TRADED_CORE), "account": account},
    )


async def _account(
    engine: AsyncEngine, account: str, label: str, quarter: date, month: date, today: date
) -> AccountCompliance:
    """One account's trade counts and its buys' verdicts this quarter.

    The verdict for a buy is the newest ``halal_screen_current`` row for its
    symbol with ``as_of`` on or before the trade's day; ties on ``as_of``
    (several screening methods on one day) go to the newest ``screened_at``.
    """
    # A coarse bound for the index; the exact test is on the New York day,
    # whose midnight falls after UTC's.
    since = datetime.combine(quarter, datetime.min.time(), UTC)
    trades_sql, trades_params = _account_trades_sql(account)
    async with engine.connect() as conn:
        counts = (
            await conn.execute(
                text(
                    "SELECT count(*) FILTER (WHERE day >= :q) AS quarter, "
                    "count(*) FILTER (WHERE day >= :m) AS month, "
                    f"count(*) FILTER (WHERE day = :t) AS today FROM ({trades_sql}) x"
                ),
                {**trades_params, "since": since, "q": quarter, "m": month, "t": today},
            )
        ).one()
        buys = (
            await conn.execute(
                text(
                    f"SELECT x.symbol, x.day, s.verdict, s.as_of FROM ({trades_sql}) x "
                    "LEFT JOIN LATERAL (SELECT verdict, as_of FROM halal_screen_current r "
                    "WHERE r.symbol = x.symbol AND r.as_of <= x.day "
                    "ORDER BY r.as_of DESC, r.screened_at DESC LIMIT 1) s ON true "
                    "WHERE x.side = 'buy' AND x.day >= :q ORDER BY x.day DESC, x.symbol"
                ),
                {**trades_params, "since": since, "q": quarter},
            )
        ).all()
    verdicts = dict.fromkeys(VERDICTS, 0)
    offending: list[dict[str, Any]] = []
    for b in buys:
        v = b.verdict or "unscreened"
        verdicts[v] = verdicts.get(v, 0) + 1
        if v != "halal" and len(offending) < _LISTED:
            offending.append(
                {
                    "symbol": b.symbol,
                    "day": b.day.isoformat(),
                    "verdict": v,
                    "screen_as_of": b.as_of.isoformat() if b.as_of else None,
                }
            )
    return AccountCompliance(
        account=account,
        label=label,
        trades_today=int(counts.today or 0),
        trades_this_month=int(counts.month or 0),
        trades_this_quarter=int(counts.quarter or 0),
        buys_this_quarter=len(buys),
        buy_verdicts=verdicts,
        non_halal_buys=offending,
    )


async def _sum_purification_accrued(session: AsyncSession, since: datetime) -> float:
    """Sum dividend-side + capital-gains-side purification due since
    ``since``."""
    ledger = await _accruals(session, since, paid_only=False)
    cap = (
        await session.exec(
            select(
                func.coalesce(func.sum(RoundTripPurificationRow.purification_due_usd), 0.0)
            ).where(RoundTripPurificationRow.timestamp >= since)
        )
    ).one()
    return ledger + float(cap or 0.0)


async def _unpaid_by_account(session: AsyncSession, since: datetime) -> dict[str, float]:
    """Purification accrued and not yet paid, per broker account. The round-trip
    ledger predates the core and carries no account: it was the day-trader's."""
    out: dict[str, float] = {}
    for account, _ in ACCOUNTS:
        owed = await _accruals(session, since, paid_only=False, accounts=[account]) - (
            await _accruals(session, since, paid_only=True, accounts=[account])
        )
        if abs(owed) > 1e-9:
            out[account] = owed
    legacy = (await _sum_legacy(session, since, paid_only=False)) - (
        await _sum_legacy(session, since, paid_only=True)
    )
    if abs(legacy) > 1e-9:
        out[DAY_TRADER] = out.get(DAY_TRADER, 0.0) + legacy
    return out


async def _sum_legacy(session: AsyncSession, since: datetime, *, paid_only: bool) -> float:
    """The round-trip (capital gains) ledger's purification since ``since``."""
    cap_q = select(
        func.coalesce(func.sum(RoundTripPurificationRow.purification_due_usd), 0.0)
    ).where(RoundTripPurificationRow.timestamp >= since)
    if paid_only:
        cap_q = cap_q.where(col(RoundTripPurificationRow.disbursed).is_(True))
    cap = (await session.exec(cap_q)).one()
    return float(cap or 0.0)


def _live_accounts() -> list[str]:
    """The broker accounts trading real money (none while both are paper)."""
    from halal_trader.config import get_settings

    settings = get_settings()
    live = [] if settings.alpaca.paper_trade else [DAY_TRADER]
    if not settings.core.paper:
        live.append(CORE_LIVE)
    return live


async def _accruals(
    session: AsyncSession,
    since: datetime,
    *,
    paid_only: bool,
    accounts: list[str] | None = None,
) -> float:
    """The broker accounts' dividend purification (compliance/purification.py), by
    payment date. Forward books are notional and stay out of the summary."""
    from halal_trader.db.models import PurificationAccrual

    names = [DAY_TRADER, *CORE_ACCOUNTS] if accounts is None else accounts
    if not names:
        return 0.0
    when = func.coalesce(PurificationAccrual.payable_date, PurificationAccrual.ex_date)
    conditions = [col(PurificationAccrual.account).in_(names), when >= since.date()]
    if paid_only:
        conditions.append(col(PurificationAccrual.paid_at).is_not(None))
    total = (
        await session.exec(
            select(func.coalesce(func.sum(PurificationAccrual.amount), 0.0)).where(*conditions)
        )
    ).one()
    return float(total or 0.0)


async def _sum_purification_disbursed(session: AsyncSession, since: datetime) -> float:
    """Sum disbursed purification across both ledgers (paid_at /
    disbursed_at NOT NULL)."""
    ledger = await _accruals(session, since, paid_only=True)
    cap = (
        await session.exec(
            select(
                func.coalesce(func.sum(RoundTripPurificationRow.purification_due_usd), 0.0)
            ).where(
                and_(
                    col(RoundTripPurificationRow.timestamp) >= since,
                    col(RoundTripPurificationRow.disbursed).is_(True),
                )
            )
        )
    ).one()
    return ledger + float(cap or 0.0)
