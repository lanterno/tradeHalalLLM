"""Position and P&L tracking for stock trading."""

import logging
from typing import Any

from halal_trader.core.portfolio import BasePortfolioTracker
from halal_trader.domain.models import Position
from halal_trader.domain.ports import Broker, TradeRepository

logger = logging.getLogger(__name__)


class PortfolioTracker(BasePortfolioTracker):
    """Tracks stock portfolio state and daily P&L via broker + local DB."""

    def __init__(
        self,
        broker: Broker,
        repo: TradeRepository,
        *,
        daily_loss_limit: float,
    ) -> None:
        super().__init__(repo, daily_loss_limit=daily_loss_limit)
        self._broker = broker

    # ── Hook implementations ───────────────────────────────────

    async def _get_equity(self, **kwargs: Any) -> float:
        account = await self._broker.get_account_info()
        return account.effective_equity or self._DEFAULT_EQUITY

    async def _get_today_trades(self) -> list[dict[str, Any]]:
        return await self._repo.get_today_trades()

    async def _persist_day_start(self, equity: float) -> None:
        await self._repo.start_day(equity)

    async def _persist_day_end(
        self, equity: float, pnl: float, count: int
    ) -> None:
        await self._repo.end_day(
            ending_equity=equity,
            realized_pnl=pnl,
            trades_count=count,
        )

    # ── Stock-specific methods ─────────────────────────────────

    async def get_positions_summary(self) -> list[Position]:
        """Get a summary of all current positions."""
        return await self._broker.get_all_positions()
