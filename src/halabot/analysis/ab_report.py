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

from halabot.platform.db import event_log as _event_log
from halabot.platform.events import EventType


@dataclass
class ABReport:
    since: datetime
    until: datetime
    shadow_total: int
    live_total: int
    shadow_by_symbol: dict[str, int] = field(default_factory=dict)
    live_by_symbol: dict[str, int] = field(default_factory=dict)

    @property
    def churn_reduction_pct(self) -> float | None:
        """Fraction fewer trades the shadow proposed vs the live cycle, or None
        if the live cycle made no trades in the window."""
        if self.live_total <= 0:
            return None
        return 1.0 - (self.shadow_total / self.live_total)

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
        # the legacy SQLModel into this package's metadata).
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

    return ABReport(
        since=since,
        until=until,
        shadow_total=sum(shadow_by_symbol.values()),
        live_total=sum(live_by_symbol.values()),
        shadow_by_symbol=shadow_by_symbol,
        live_by_symbol=live_by_symbol,
    )
