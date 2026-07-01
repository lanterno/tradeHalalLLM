"""Daily LLM spend cap — circuit breaker that engages the kill-switch.

A single ``LLMBudget`` instance is wired into the bot at startup and
consulted on every successful LLM call. When the UTC-day cumulative
spend crosses the cap, it engages the global kill-switch with a
descriptive reason and emits an operator alert.

The cap is intentionally a hard ceiling, not a soft warning. The bot
already keeps working capital under tight per-trade risk limits; an
unbounded LLM bill is the one expense that can dwarf the trading P&L
without any matching safeguard. So when in doubt: stop, page the
operator, let them clear it manually.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core import events
from halal_trader.core.halt import is_halted, set_halt

logger = logging.getLogger(__name__)


@dataclass
class _DailyState:
    date: str
    spent_usd: Decimal


class LLMBudget:
    """Track per-day LLM spend; trip the kill-switch when the cap is breached.

    ``cap_usd <= 0`` disables enforcement entirely (the bot still tracks
    spend but never trips). This is the default so a misconfigured
    operator running a $0-cost test config doesn't get an unexpected
    halt the first time they try a cloud fallback.
    """

    def __init__(self, engine: AsyncEngine, *, cap_usd: float) -> None:
        self._engine = engine
        self._cap = Decimal(str(cap_usd))
        self._state = _DailyState(date=_today(), spent_usd=Decimal("0"))
        self._lock = asyncio.Lock()
        self._tripped = False

    @property
    def cap_usd(self) -> Decimal:
        return self._cap

    @property
    def spent_today_usd(self) -> Decimal:
        if self._state.date != _today():
            return Decimal("0")
        return self._state.spent_usd

    async def record(self, cost_usd: Decimal) -> None:
        """Add ``cost_usd`` to today's running total; trip the breaker if over cap.

        Safe to call from concurrent cycles — guarded by an asyncio lock
        so the rollover/trip transition is atomic.
        """
        if cost_usd <= 0:
            return
        async with self._lock:
            today = _today()
            if self._state.date != today:
                self._state = _DailyState(date=today, spent_usd=Decimal("0"))
                self._tripped = False
            self._state.spent_usd += cost_usd

            if self._cap <= 0 or self._tripped:
                return
            if self._state.spent_usd < self._cap:
                return

            # Cap exceeded — engage the halt. Don't re-engage if the
            # operator has explicitly halted for another reason; just log.
            spent = self._state.spent_usd
            cap = self._cap
            try:
                if not await is_halted(self._engine):
                    reason = f"LLM daily spend ${spent:.2f} exceeded cap ${cap:.2f} on {today}"
                    await set_halt(self._engine, reason=reason, set_by="llm-budget")
                    logger.error(
                        "LLM budget cap tripped — kill-switch engaged",
                        extra={
                            "event": events.LLM_CHAIN_BACKOFF,
                            "spent_usd": float(spent),
                            "cap_usd": float(cap),
                            "date": today,
                        },
                    )
            finally:
                self._tripped = True


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")
