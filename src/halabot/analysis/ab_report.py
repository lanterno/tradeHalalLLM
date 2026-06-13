"""Shadow-vs-live A/B report (REARCHITECTURE Part IV, Phase-3 gate).

Compares, over a window, the shadow engine's ``policy.trade_proposed`` events
(``hb_event_log``) against the legacy stock cycle's actual filled trades
(``trades``) — both in the shared Postgres. The headline metric is **churn**:
does the conviction engine propose materially fewer trades than the live cycle
executed? (P&L comparison needs hypothetical-fill tracking — a later step; this
first cut establishes the trade-count / symbol-spread comparison.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.analysis.significance import (
    PromotionVerdict,
    promotion_gate,
    variance,
)
from halabot.platform.db import event_log as _event_log
from halabot.platform.db import outcome as _outcome
from halabot.platform.events import EventType


def churn_reduction(shadow_total: int, live_total: int) -> float | None:
    """Fraction fewer trades the shadow proposed vs the live cycle (None if the
    live cycle made no trades). Single source of truth for the churn metric."""
    if live_total <= 0:
        return None
    return 1.0 - (shadow_total / live_total)


@dataclass
class ABReport:
    since: datetime
    until: datetime
    shadow_total: int
    live_total: int
    shadow_by_symbol: dict[str, int] = field(default_factory=dict)
    live_by_symbol: dict[str, int] = field(default_factory=dict)
    # Shadow hypothetical P&L (from closed hb_outcome rows in the window).
    shadow_closed: int = 0
    shadow_avg_return_pct: float | None = None
    shadow_win_rate: float | None = None
    shadow_weighted_return: float = 0.0  # Σ return_pct × closed_weight (book-level proxy)
    shadow_return_std: float | None = None
    # Live realized per-trade returns (regret_records.pnl_pct) + the promotion gate.
    live_closed: int = 0
    live_avg_return_pct: float | None = None
    promotion: PromotionVerdict | None = None

    @property
    def churn_reduction_pct(self) -> float | None:
        """Fraction fewer trades the shadow proposed vs the live cycle, or None
        if the live cycle made no trades in the window."""
        return churn_reduction(self.shadow_total, self.live_total)

    @property
    def symbols_only_live(self) -> set[str]:
        """Symbols the live cycle traded but the shadow never proposed (the
        churn the conviction engine avoided)."""
        return set(self.live_by_symbol) - set(self.shadow_by_symbol)


async def ab_report(engine: AsyncEngine, *, since: datetime, until: datetime) -> ABReport:
    shadow_by_symbol: dict[str, int] = {}
    live_by_symbol: dict[str, int] = {}

    async with engine.connect() as conn:
        # Shadow proposals from the durable event log.
        t = _event_log
        shadow_rows = await conn.execute(
            sa.select(t.c.asset, sa.func.count())
            .where(
                t.c.type == str(EventType.POLICY_TRADE_PROPOSED),
                t.c.ts >= since,
                t.c.ts <= until,
            )
            .group_by(t.c.asset)
        )
        for asset, n in shadow_rows:
            shadow_by_symbol[asset or "?"] = int(n)

        # Live filled trades from the legacy table (raw SQL — avoids importing
        # the legacy SQLModel into this package's metadata). Both sides count
        # by design — the shadow book also counts its sell legs. NOTE: as of
        # 2026-06-11 the live monitor records its SL/TP/trailing exits as
        # 'filled' SELL rows (previously only LLM sells produced one), so
        # live_total includes monitor-exit legs from that date; churn_reduction
        # values are not comparable across windows spanning that change.
        live_rows = await conn.execute(
            sa.text(
                "SELECT symbol, count(*) FROM trades "
                "WHERE timestamp >= :since AND timestamp <= :until "
                "AND status = 'filled' AND side IN ('buy', 'sell') "
                "GROUP BY symbol"
            ),
            {"since": since, "until": until},
        )
        for symbol, n in live_rows:
            live_by_symbol[symbol or "?"] = int(n)

        # Shadow hypothetical per-trade returns from closed outcomes in the window.
        o = _outcome
        shadow_returns = [
            float(r[0])
            for r in await conn.execute(
                sa.select(o.c.return_pct).where(o.c.exit_ts >= since, o.c.exit_ts <= until)
            )
        ]
        weighted_ret = float(
            (
                await conn.execute(
                    sa.select(
                        sa.func.coalesce(sa.func.sum(o.c.return_pct * o.c.closed_weight), 0.0)
                    ).where(o.c.exit_ts >= since, o.c.exit_ts <= until)
                )
            ).scalar_one()
            or 0.0
        )
        labels = [
            int(r[0])
            for r in await conn.execute(
                sa.select(o.c.label).where(o.c.exit_ts >= since, o.c.exit_ts <= until)
            )
        ]

        # Live realized per-trade returns from the legacy regret_records (raw SQL,
        # avoids importing the legacy SQLModel). Empty/missing → no live P&L data
        # (the gate stays "insufficient samples" until it accrues — data-gated).
        live_returns: list[float] = []
        try:
            live_rows2 = await conn.execute(
                sa.text(
                    "SELECT pnl_pct FROM regret_records "
                    "WHERE closed_at >= :since AND closed_at <= :until"
                ),
                {"since": since, "until": until},
            )
            live_returns = [float(r[0]) for r in live_rows2 if r[0] is not None]
        except Exception:  # noqa: BLE001 — table may not exist in a fresh/test DB
            live_returns = []

    closed = len(shadow_returns)
    avg_ret = sum(shadow_returns) / closed if closed else None
    win_rate = (sum(labels) / len(labels)) if labels else None
    ret_std = variance(shadow_returns) ** 0.5 if closed >= 2 else None

    shadow_total = sum(shadow_by_symbol.values())
    live_total = sum(live_by_symbol.values())
    churn = churn_reduction(shadow_total, live_total)
    live_avg = sum(live_returns) / len(live_returns) if live_returns else None
    promotion = promotion_gate(shadow_returns, live_returns, churn_reduction=churn)

    return ABReport(
        since=since,
        until=until,
        shadow_total=shadow_total,
        live_total=live_total,
        shadow_by_symbol=shadow_by_symbol,
        live_by_symbol=live_by_symbol,
        shadow_closed=closed,
        shadow_avg_return_pct=avg_ret,
        shadow_win_rate=win_rate,
        shadow_weighted_return=weighted_ret,
        shadow_return_std=ret_std,
        live_closed=len(live_returns),
        live_avg_return_pct=live_avg,
        promotion=promotion,
    )
