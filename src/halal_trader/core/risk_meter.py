"""The operator's risk meter for the day trader: the single-row ``risk_meter`` table.

One dial, five levels (operator decision of 2026-10-27, HAL-13). Each level
sets the day trader's limits together, so no combination is inconsistent;
level 3 is the configured settings. A cut takes effect at once and a raise
from the next trading day, so risk is never raised mid-session. The meter
covers the day trader only: the core is the baseline it has to beat.

Phase 1 only stores and shows the level, and each trade records it. Until
the cycle applies the level (HAL-14, after the 11-02 review), only level 3
can be set; levels 4 and 5 stay locked until the picks pass the unlock
test (HAL-16).
"""

from __future__ import annotations

import getpass
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlmodel import select

from halal_trader.db.models import RiskMeter
from halal_trader.market_hours import next_trading_day, today_eastern


@dataclass(frozen=True)
class RiskLimits:
    """What one level sets. Fractions of equity throughout."""

    level: int
    name: str
    max_position_pct: float
    typical_position_pct: tuple[float, float]
    max_positions: int
    daily_loss_limit: float
    max_drawdown_pct: float
    max_portfolio_heat_pct: float
    # How far below the fill a buy's stop may sit; None leaves the stop,
    # and whether to set one, to the model.
    max_stop_distance_pct: float | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


LEVELS: dict[int, RiskLimits] = {
    1: RiskLimits(1, "Cautious", 0.05, (0.02, 0.04), 2, 0.005, 0.03, 0.015, 0.02),
    2: RiskLimits(2, "Careful", 0.10, (0.04, 0.08), 3, 0.01, 0.05, 0.03, 0.03),
    3: RiskLimits(3, "Standard", 0.20, (0.08, 0.15), 5, 0.02, 0.08, 0.05, None),
    4: RiskLimits(4, "Bold", 0.25, (0.10, 0.18), 6, 0.025, 0.10, 0.06, None),
    5: RiskLimits(5, "Max", 0.30, (0.12, 0.22), 8, 0.03, 0.12, 0.08, None),
}
STANDARD = 3
# The levels the operator can choose now. The cycle doesn't apply a level
# yet (HAL-14), so the meter holds the configured settings.
SETTABLE = frozenset({STANDARD})


class MeterRefused(ValueError):
    """The level can't be set; the message says why."""


def refusal(level: int) -> str | None:
    """Why ``level`` can't be set now, or None when it can."""
    if level not in LEVELS:
        return f"There is no level {level}: the levels are 1 to 5."
    if level in SETTABLE:
        return None
    if level < STANDARD:
        return "Levels 1 and 2 open once the bot applies the meter, after the 11-02 review."
    return "Levels 4 and 5 stay locked until the day trader's picks pass the unlock test."


@dataclass(frozen=True)
class MeterStatus:
    level: int  # in effect today
    chosen: int  # last set; above ``level`` while a raise waits
    effective_from: date | None  # when ``chosen`` takes effect, while it waits
    reason: str | None
    set_by: str | None
    set_at: datetime | None

    @property
    def limits(self) -> RiskLimits:
        return LEVELS[self.level]

    def to_json(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "name": self.limits.name,
            "chosen": self.chosen,
            "effective_from": self.effective_from.isoformat() if self.effective_from else None,
            "reason": self.reason,
            "set_by": self.set_by,
            "set_at": self.set_at.isoformat() if self.set_at else None,
            "limits": self.limits.to_json(),
        }


def levels_json() -> list[dict[str, Any]]:
    """Every level with what it sets and whether it can be chosen now."""
    return [{**lim.to_json(), "refusal": refusal(n)} for n, lim in LEVELS.items()]


async def _row(session: AsyncSession) -> RiskMeter | None:
    result = await session.execute(select(RiskMeter).where(RiskMeter.id == 1))
    return result.scalars().first()


def _status(row: RiskMeter | None, today: date) -> MeterStatus:
    if row is None:
        return MeterStatus(STANDARD, STANDARD, None, None, None, None)
    waiting = row.effective_from is not None and today < row.effective_from
    return MeterStatus(
        level=row.previous_level if waiting else row.level,
        chosen=row.level,
        effective_from=row.effective_from if waiting else None,
        reason=row.reason,
        set_by=row.set_by,
        set_at=row.set_at,
    )


async def get_status(engine: AsyncEngine, *, today: date | None = None) -> MeterStatus:
    """The meter as of ``today`` (Eastern); no row means level 3."""
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = await _row(session)
    return _status(row, today or today_eastern())


async def set_level(
    engine: AsyncEngine,
    level: int,
    *,
    reason: str,
    set_by: str | None = None,
    today: date | None = None,
) -> MeterStatus:
    """Choose ``level``: a cut applies at once, a raise from the next trading day.

    Raises :class:`MeterRefused` for a level that can't be chosen now.
    """
    why = refusal(level)
    if why is not None:
        raise MeterRefused(why)
    if set_by is None:
        try:
            set_by = getpass.getuser()
        except OSError, KeyError:
            set_by = "unknown"
    today = today or today_eastern()

    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = await _row(session)
        current = _status(row, today).level
        values = {
            "level": level,
            "previous_level": current,
            "effective_from": next_trading_day(today) if level > current else None,
            "reason": reason,
            "set_by": set_by,
            "set_at": datetime.now(UTC),
        }
        if row is None:
            row = RiskMeter(id=1, **values)
        else:
            for name, value in values.items():
                setattr(row, name, value)
        session.add(row)
        await session.commit()
    return _status(row, today)


def describe(status: MeterStatus) -> str:
    """One line for the operator: the level, and a raise still waiting."""
    lim = status.limits
    line = f"Risk meter at level {lim.level} ({lim.name})"
    if status.effective_from is not None:
        chosen = LEVELS[status.chosen]
        line += f"; level {chosen.level} ({chosen.name}) from {status.effective_from.isoformat()}"
    if status.set_by:
        line += f", set by {status.set_by}: {status.reason}"
    return line


async def announce(settings: Any, status: MeterStatus) -> bool:
    """Tell the operator on Telegram that the level changed."""
    from halal_trader.notifications.telegram import TelegramNotifier

    notifier = TelegramNotifier(
        bot_token=settings.telegram.bot_token, chat_id=settings.telegram.chat_id
    )
    try:
        return await notifier.send(describe(status), parse_mode="")
    finally:
        await notifier.close()
