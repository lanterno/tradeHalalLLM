"""Position and P&L tracking."""

import logging
from typing import Any

from halal_trader.domain.models import Position
from halal_trader.domain.ports import Broker, TradeRepository

logger = logging.getLogger(__name__)

# Fallback equity when no real data is available (paper-trading safety net).
_DEFAULT_EQUITY = 100_000.0


class PortfolioTracker:
    """Tracks portfolio state and daily P&L via broker + local DB."""

    def __init__(
        self,
        broker: Broker,
        repo: TradeRepository,
        *,
        daily_loss_limit: float,
    ) -> None:
        self._broker = broker
        self._repo = repo
        self._daily_loss_limit = daily_loss_limit
        self._starting_equity: float | None = None

    async def record_day_start(self) -> float:
        """Record the starting equity for today. Returns starting equity."""
        account = await self._broker.get_account_info()
        equity = account.effective_equity or _DEFAULT_EQUITY
        self._starting_equity = equity
        await self._repo.start_day(equity)
        logger.info("Day started with equity: $%.2f", equity)
        return equity

    async def record_day_end(self) -> dict[str, Any]:
        """Record end-of-day stats. Returns summary dict."""
        account = await self._broker.get_account_info()
        equity = account.effective_equity or _DEFAULT_EQUITY
        trades = await self._repo.get_today_trades()
        trades_count = len(trades)

        realized_pnl = equity - (self._starting_equity or equity)
        await self._repo.end_day(
            ending_equity=equity,
            realized_pnl=realized_pnl,
            trades_count=trades_count,
        )

        starting = self._starting_equity or equity
        return_pct = (equity - starting) / starting if starting else 0

        summary = {
            "starting_equity": starting,
            "ending_equity": equity,
            "realized_pnl": realized_pnl,
            "return_pct": return_pct,
            "trades_count": trades_count,
        }
        logger.info(
            "Day ended: $%.2f -> $%.2f (P&L: $%+.2f, %+.2f%%, %d trades)",
            starting,
            equity,
            realized_pnl,
            return_pct * 100,
            trades_count,
        )
        return summary

    async def get_current_pnl(self) -> float:
        """Get the current unrealized + realized P&L for today."""
        account = await self._broker.get_account_info()
        equity = account.effective_equity or _DEFAULT_EQUITY
        starting = self._starting_equity or equity
        return equity - starting

    async def get_positions_summary(self) -> list[Position]:
        """Get a summary of all current positions."""
        return await self._broker.get_all_positions()

    async def should_halt_trading(self) -> bool:
        """Check if daily loss limit has been breached."""
        pnl = await self.get_current_pnl()
        starting = self._starting_equity or _DEFAULT_EQUITY
        loss_pct = abs(pnl) / starting if pnl < 0 else 0

        if loss_pct >= self._daily_loss_limit:
            logger.warning(
                "Daily loss limit breached: %.2f%% (limit: %.2f%%)",
                loss_pct * 100,
                self._daily_loss_limit * 100,
            )
            return True
        return False
