"""AAOIFI compliance summary — the dashboard tile data source.

Round-4 wave 2.E: the operator's dashboard gets a "live AAOIFI
compliance" tile. This module computes the rollup that powers it:

* trades-this-quarter (and this-month / today)
* screenings-by-decision (halal / doubtful / not_halal counts)
* compliance violation count (any trade that filled with
  ``decision != 'halal'``)
* purification accrued (dividend + capital-gains sides combined)
* purification disbursed (paid out to charity)
* outstanding purification due

The summary is **read-only** — it reads from existing tables
(`halal_screenings`, `purification_entries`,
`round_trip_purification`, `trades`, `crypto_trades`) and returns
a typed dataclass the dashboard route serialises to JSON.

Design choice: pure SQL aggregations, no Python-side per-row math.
Keeps the tile fast even when the audit log has 100k+ rows.

Halal-jurisprudence note: AAOIFI 2.4 (debt screening) and 5.4
(financial-services screening) are referenced via the
:mod:`halal.aaoifi_seed` documentation — this summary doesn't
re-implement those rules; it counts decisions made by upstream
screeners (Zoya, CoinGecko, manual overrides) that already encode
the AAOIFI thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import and_, func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import (
    CryptoTrade,
    HalalScreening,
    PurificationEntry,
    RoundTripPurificationRow,
    Trade,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AAOIFISummary:
    """Snapshot of halal-compliance state at a point in time.

    All counts are inclusive of the current period; the period
    boundaries are 'today' (UTC midnight), 'this month' (1st of
    current month UTC), and 'this quarter' (1st of current quarter
    UTC). Operator's dashboard renders the quarter view by default.
    """

    quarter_start: datetime
    month_start: datetime
    today_start: datetime

    # Trade counts (any trade that filled, regardless of compliance)
    trades_today: int
    trades_this_month: int
    trades_this_quarter: int

    # Screening decisions across the quarter (most useful audit slice)
    halal_screenings_quarter: int
    doubtful_screenings_quarter: int
    not_halal_screenings_quarter: int

    # The number that matters most: trades that filled with a
    # NON-halal screening attached. Should always be 0 in normal ops;
    # any non-zero value is a red-alert tile state.
    non_halal_fills_quarter: int

    # Purification: combined across dividend (PurificationEntry) and
    # capital-gains (RoundTripPurificationRow) sides.
    purification_accrued_usd: float
    purification_disbursed_usd: float

    @property
    def purification_outstanding_usd(self) -> float:
        return max(0.0, self.purification_accrued_usd - self.purification_disbursed_usd)

    @property
    def is_compliant(self) -> bool:
        """True iff no non-halal trades filled this quarter."""
        return self.non_halal_fills_quarter == 0

    @property
    def status(self) -> str:
        """Operator-readable status: 'compliant' | 'attention' | 'violation'.

        * 'violation' — at least one non-halal trade filled this
          quarter. Tile renders red.
        * 'attention' — outstanding purification is non-zero (operator
          owes a disbursement). Tile renders amber.
        * 'compliant' — green.
        """
        if self.non_halal_fills_quarter > 0:
            return "violation"
        if self.purification_outstanding_usd > 0.01:
            return "attention"
        return "compliant"


def _quarter_start_utc(now: datetime) -> datetime:
    """Return the start of the calendar quarter containing ``now``."""
    quarter_first_month = ((now.month - 1) // 3) * 3 + 1
    return datetime(now.year, quarter_first_month, 1, tzinfo=UTC)


def _month_start_utc(now: datetime) -> datetime:
    return datetime(now.year, now.month, 1, tzinfo=UTC)


def _today_start_utc(now: datetime) -> datetime:
    return datetime(now.year, now.month, now.day, tzinfo=UTC)


async def compute_aaoifi_summary(
    engine: "AsyncEngine",
    *,
    now: datetime | None = None,
) -> AAOIFISummary:
    """Aggregate compliance state into a single dashboard-tile object.

    The ``now`` arg is for testing — production callers leave it
    None and get the current UTC time. All boundary datetimes are
    UTC-tz-aware so the SQL comparisons against tz-aware columns
    work correctly.
    """
    if now is None:
        now = datetime.now(UTC)
    quarter = _quarter_start_utc(now)
    month = _month_start_utc(now)
    today = _today_start_utc(now)

    async with AsyncSession(engine) as session:
        # ── Trade counts (combined stocks + crypto) ───────────
        trades_today = await _count_trades_since(session, today)
        trades_month = await _count_trades_since(session, month)
        trades_quarter = await _count_trades_since(session, quarter)

        # ── Screening counts per decision (quarter) ───────────
        halal_q = await _count_screenings(session, quarter, "halal")
        doubtful_q = await _count_screenings(session, quarter, "doubtful")
        not_halal_q = await _count_screenings(session, quarter, "not_halal")

        # ── Non-halal fills (the red-alert metric) ────────────
        non_halal_fills = await _count_non_halal_fills(session, quarter)

        # ── Purification (accrued + disbursed) ────────────────
        accrued = await _sum_purification_accrued(session, quarter)
        disbursed = await _sum_purification_disbursed(session, quarter)

    return AAOIFISummary(
        quarter_start=quarter,
        month_start=month,
        today_start=today,
        trades_today=trades_today,
        trades_this_month=trades_month,
        trades_this_quarter=trades_quarter,
        halal_screenings_quarter=halal_q,
        doubtful_screenings_quarter=doubtful_q,
        not_halal_screenings_quarter=not_halal_q,
        non_halal_fills_quarter=non_halal_fills,
        purification_accrued_usd=accrued,
        purification_disbursed_usd=disbursed,
    )


async def _count_trades_since(session: AsyncSession, since: datetime) -> int:
    """Total trades (stocks + crypto) on or after ``since``."""
    stocks = (
        await session.exec(select(func.count()).select_from(Trade).where(Trade.timestamp >= since))
    ).one()
    crypto = (
        await session.exec(
            select(func.count()).select_from(CryptoTrade).where(CryptoTrade.timestamp >= since)
        )
    ).one()
    return int(stocks or 0) + int(crypto or 0)


async def _count_screenings(session: AsyncSession, since: datetime, decision: str) -> int:
    result = await session.exec(
        select(func.count())
        .select_from(HalalScreening)
        .where(
            and_(
                HalalScreening.timestamp >= since,
                HalalScreening.decision == decision,
            )
        )
    )
    return int(result.one() or 0)


async def _count_non_halal_fills(session: AsyncSession, since: datetime) -> int:
    """Any *filled* trade (stocks or crypto) joined to a screening with
    a non-``halal`` decision. The screening rows are the single source
    of truth — a trade with no `halal_screening_id` is "unattested" and
    counted here defensively (better surfaced than hidden)."""
    stmt_stock = (
        select(func.count())
        .select_from(Trade)
        .join(HalalScreening, Trade.halal_screening_id == HalalScreening.id, isouter=True)
        .where(
            and_(
                Trade.timestamp >= since,
                Trade.status.in_(["filled", "submitted"]),
                ~(HalalScreening.decision == "halal"),
            )
        )
    )
    stmt_crypto = (
        select(func.count())
        .select_from(CryptoTrade)
        .join(HalalScreening, CryptoTrade.halal_screening_id == HalalScreening.id, isouter=True)
        .where(
            and_(
                CryptoTrade.timestamp >= since,
                CryptoTrade.status.in_(["filled", "submitted"]),
                ~(HalalScreening.decision == "halal"),
            )
        )
    )
    s = (await session.exec(stmt_stock)).one() or 0
    c = (await session.exec(stmt_crypto)).one() or 0
    return int(s) + int(c)


async def _sum_purification_accrued(session: AsyncSession, since: datetime) -> float:
    """Sum dividend-side + capital-gains-side purification due since
    ``since``."""
    div = (
        await session.exec(
            select(func.coalesce(func.sum(PurificationEntry.purification_usd), 0.0)).where(
                PurificationEntry.timestamp >= since
            )
        )
    ).one()
    cap = (
        await session.exec(
            select(
                func.coalesce(func.sum(RoundTripPurificationRow.purification_due_usd), 0.0)
            ).where(RoundTripPurificationRow.timestamp >= since)
        )
    ).one()
    return float(div or 0.0) + float(cap or 0.0)


async def _sum_purification_disbursed(session: AsyncSession, since: datetime) -> float:
    """Sum disbursed purification across both ledgers (paid_at /
    disbursed_at NOT NULL)."""
    div = (
        await session.exec(
            select(func.coalesce(func.sum(PurificationEntry.purification_usd), 0.0)).where(
                and_(
                    PurificationEntry.timestamp >= since,
                    PurificationEntry.paid_at.is_not(None),
                )
            )
        )
    ).one()
    cap = (
        await session.exec(
            select(
                func.coalesce(func.sum(RoundTripPurificationRow.purification_due_usd), 0.0)
            ).where(
                and_(
                    RoundTripPurificationRow.timestamp >= since,
                    RoundTripPurificationRow.disbursed.is_(True),
                )
            )
        )
    ).one()
    return float(div or 0.0) + float(cap or 0.0)
