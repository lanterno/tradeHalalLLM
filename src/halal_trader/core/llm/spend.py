"""The LLM spend caps: a daily and a monthly total per pool, across processes.

Every GLM call reports its cost here (core/llm/glm.py) and asks first
whether it may run. The total lives in the ``llm_spend`` table, one row per
(UTC day, consumer), so it survives restarts and adds up the stock bot and
the shadow engine, which share one OpenRouter key and one bill.

Two modes (``LLM_BUDGET_ENFORCE``):

* observe (default): never blocks; alerts once a day at 80% of the cap and
  once when the cap is crossed, so the cap can be sized from real data;
* enforce: additionally refuses every further LLM call for the rest of the
  UTC day once the cap is reached. That is the actual spend stop -- the
  kill-switch alone would not be (the classifier and the shadow keep
  calling). Exits keep working: the position monitor does not use the LLM.

On top of the daily cap, each consumer belongs to a **monthly pool** with
its own cap (operator decision 2026-10-02: $25 live for the bot and the
shadow engine, $15 research, $10 of the $50 key limit as headroom). The
pool alerts and, in enforce mode, refuses exactly like the daily cap, so
research can never spend the live bot's budget, nor the reverse.

Nothing here raises into a caller except :class:`BudgetExhausted`, which is
the point of enforce mode. Metering failures are logged and ignored.

Replaces core/llm/budget.py's LLMBudget, which was never constructed.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

Alert = Callable[[str, str], Awaitable[None]]

_WARN_FRACTION = Decimal("0.8")

# consumer -> monthly pool. A consumer not listed is its own pool with no cap.
POOLS: dict[str, str] = {"stock": "live", "shadow": "live", "research": "research"}


def pool_members(pool: str) -> frozenset[str]:
    return frozenset(c for c, p in POOLS.items() if p == pool)


class BudgetExhausted(RuntimeError):
    """Raised instead of making an LLM call once an enforced cap is reached."""


class SpendMeter:
    def __init__(
        self,
        engine: AsyncEngine,
        *,
        consumer: str,
        cap_usd: float,
        enforce: bool = False,
        alert: Alert | None = None,
        monthly_cap_usd: float = 0.0,
    ) -> None:
        self._engine = engine
        self._consumer = consumer
        self._cap = Decimal(str(cap_usd))
        self._pool = POOLS.get(consumer, consumer)
        self._pool_members = pool_members(self._pool) or frozenset({consumer})
        self._monthly_cap = Decimal(str(monthly_cap_usd))
        self._enforce = enforce
        self._alert = alert
        self._alerted: set[tuple[date, str]] = set()
        self._exhausted_on: date | None = None

    @property
    def cap_usd(self) -> Decimal:
        return self._cap

    async def spent_today(self) -> Decimal:
        """Today's spend (UTC) of every consumer in this meter's pool.

        Pool-scoped like the monthly budget: research scoring must never
        use up the live bot's daily cap (on 2026-10-02 a research run took
        the day's total past the live $3 cap; in enforce mode that would
        have silenced the bot for the rest of the day).
        """
        async with self._engine.connect() as conn:
            total = (
                await conn.execute(
                    text(
                        "SELECT coalesce(sum(spent_usd), 0) FROM llm_spend "
                        "WHERE day = :day AND consumer = ANY(:members)"
                    ),
                    {"day": _today(), "members": sorted(self._pool_members)},
                )
            ).scalar()
        return Decimal(str(total))

    async def spent_this_month(self) -> Decimal:
        """Month-to-date spend (UTC) of every consumer in this meter's pool."""
        today = _today()
        async with self._engine.connect() as conn:
            total = (
                await conn.execute(
                    text(
                        "SELECT coalesce(sum(spent_usd), 0) FROM llm_spend "
                        "WHERE day >= :first AND consumer = ANY(:members)"
                    ),
                    {"first": today.replace(day=1), "members": sorted(self._pool_members)},
                )
            ).scalar()
        return Decimal(str(total))

    async def check(self) -> None:
        """Raise BudgetExhausted if enforce mode is on and today's or the pool's cap is spent."""
        if not self._enforce or (self._cap <= 0 and self._monthly_cap <= 0):
            return
        today = _today()
        if self._exhausted_on == today:
            raise BudgetExhausted(f"LLM budget reached for {today} ({self._consumer})")
        try:
            daily = await self.spent_today() if self._cap > 0 else None
            monthly = await self.spent_this_month() if self._monthly_cap > 0 else None
        except Exception as exc:  # noqa: BLE001 -- metering must not block on a DB hiccup
            logger.warning("LLM spend check failed: %r", exc)
            return
        if daily is not None and daily >= self._cap:
            self._exhausted_on = today
            raise BudgetExhausted(f"LLM daily cap ${self._cap} reached for {today}")
        if monthly is not None and monthly >= self._monthly_cap:
            self._exhausted_on = today
            raise BudgetExhausted(
                f"LLM monthly {self._pool} budget ${self._monthly_cap} reached for {today:%Y-%m}"
            )

    async def record(self, cost_usd: Decimal) -> None:
        """Add one call's cost to today's total; alert at 80% and at the cap."""
        if cost_usd <= 0:
            return
        today = _today()
        try:
            async with self._engine.begin() as conn:
                await conn.execute(
                    text(
                        """
                        INSERT INTO llm_spend (day, consumer, calls, spent_usd)
                        VALUES (:day, :consumer, 1, :cost)
                        ON CONFLICT (day, consumer) DO UPDATE
                            SET calls = llm_spend.calls + 1,
                                spent_usd = llm_spend.spent_usd + EXCLUDED.spent_usd
                        """
                    ),
                    {"day": today, "consumer": self._consumer, "cost": cost_usd},
                )
            spent = await self.spent_today()
            monthly = await self.spent_this_month() if self._monthly_cap > 0 else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM spend not recorded: %r", exc)
            return
        if monthly is not None:
            await self._check_monthly(today, monthly)
        if self._cap <= 0:
            return
        if spent >= self._cap:
            if self._enforce:
                self._exhausted_on = today
            await self._alert_once(
                today,
                "llm.budget_exhausted",
                f"LLM spend ${spent:.2f} reached the ${self._cap:.2f} daily cap on {today}; "
                + (
                    "further LLM calls are refused until tomorrow (UTC)."
                    if self._enforce
                    else "observe mode: calls continue (set LLM_BUDGET_ENFORCE=true to stop them)."
                ),
            )
        elif spent >= self._cap * _WARN_FRACTION:
            await self._alert_once(
                today,
                "llm.budget_warning",
                f"LLM spend ${spent:.2f} is past 80% of the ${self._cap:.2f} daily cap on {today}.",
            )

    async def _check_monthly(self, today: date, spent: Decimal) -> None:
        month = today.replace(day=1)
        if spent >= self._monthly_cap:
            if self._enforce:
                self._exhausted_on = today
            await self._alert_once(
                month,
                f"llm.budget_exhausted.{self._pool}",
                f"LLM {self._pool} spend ${spent:.2f} reached its ${self._monthly_cap:.2f} "
                f"monthly budget ({today:%Y-%m}); "
                + (
                    "further calls are refused until tomorrow (UTC), and every day after "
                    "until the month turns."
                    if self._enforce
                    else "observe mode: calls continue."
                ),
            )
        elif spent >= self._monthly_cap * _WARN_FRACTION:
            await self._alert_once(
                month,
                f"llm.budget_warning.{self._pool}",
                f"LLM {self._pool} spend ${spent:.2f} is past 80% of its "
                f"${self._monthly_cap:.2f} monthly budget ({today:%Y-%m}).",
            )

    async def _alert_once(self, day: date, kind: str, message: str) -> None:
        if (day, kind) in self._alerted:
            return
        self._alerted.add((day, kind))
        logger.warning(message, extra={"event": kind})
        if self._alert is not None:
            try:
                await self._alert(kind, message)
            except Exception as exc:  # noqa: BLE001
                logger.debug("spend alert failed: %r", exc)


def research_meter(engine: AsyncEngine, settings: Any) -> SpendMeter:
    """The research pool's meter: enforced, capped by the monthly research budget."""
    return SpendMeter(
        engine,
        consumer="research",
        cap_usd=0.0,
        enforce=True,
        monthly_cap_usd=monthly_cap_for(
            "research",
            live_usd=settings.llm.monthly_live_usd,
            research_usd=settings.llm.monthly_research_usd,
        ),
    )


def monthly_cap_for(consumer: str, *, live_usd: float, research_usd: float) -> float:
    """The monthly budget of ``consumer``'s pool (0 = none)."""
    return {"live": live_usd, "research": research_usd}.get(POOLS.get(consumer, ""), 0.0)


_meter: SpendMeter | None = None
# A meter for one stretch of work inside a process that has its own (the
# bot's evening research run scores headlines under "research", while the
# bot's calls stay under "stock"). Context-local: tasks started inside
# inherit it, nothing outside sees it.
_override: ContextVar[SpendMeter | None] = ContextVar("llm_spend_override", default=None)


def install(meter: SpendMeter | None) -> None:
    """Make ``meter`` this process's meter (called once at startup)."""
    global _meter
    _meter = meter


@contextmanager
def metered(meter: SpendMeter) -> Iterator[None]:
    """Charge every LLM call made inside this block (and its tasks) to ``meter``."""
    token = _override.set(meter)
    try:
        yield
    finally:
        _override.reset(token)


def _current() -> SpendMeter | None:
    return _override.get() or _meter


async def before_call() -> None:
    meter = _current()
    if meter is not None:
        await meter.check()


async def after_call(cost_usd: Decimal) -> None:
    meter = _current()
    if meter is not None:
        await meter.record(cost_usd)


def _today() -> date:
    return datetime.now(UTC).date()
