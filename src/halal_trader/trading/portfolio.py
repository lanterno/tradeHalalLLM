"""Position and P&L tracking."""

from __future__ import annotations

import logging
from typing import Any

from halal_trader.db.repository import Repository
from halal_trader.mcp.client import AlpacaMCPClient

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """Tracks portfolio state and daily P&L via Alpaca MCP + local DB."""

    def __init__(self, mcp: AlpacaMCPClient, repo: Repository) -> None:
        self._mcp = mcp
        self._repo = repo
        self._starting_equity: float | None = None

    async def record_day_start(self) -> float:
        """Record the starting equity for today. Returns starting equity."""
        account = await self._mcp.get_account_info()
        equity = self._extract_equity(account)
        self._starting_equity = equity
        await self._repo.start_day(equity)
        logger.info("Day started with equity: $%.2f", equity)
        return equity

    async def record_day_end(self) -> dict[str, Any]:
        """Record end-of-day stats. Returns summary dict."""
        account = await self._mcp.get_account_info()
        equity = self._extract_equity(account)
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
        account = await self._mcp.get_account_info()
        equity = self._extract_equity(account)
        starting = self._starting_equity or equity
        return equity - starting

    async def get_positions_summary(self) -> list[dict[str, Any]]:
        """Get a summary of all current positions."""
        raw = await self._mcp.get_all_positions()
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str):
            return [{"raw": raw}]
        return []

    async def should_halt_trading(self) -> bool:
        """Check if daily loss limit has been breached."""
        from halal_trader.config import get_settings

        settings = get_settings()
        pnl = await self.get_current_pnl()
        starting = self._starting_equity or 100000
        loss_pct = abs(pnl) / starting if pnl < 0 else 0

        if loss_pct >= settings.daily_loss_limit:
            logger.warning(
                "Daily loss limit breached: %.2f%% (limit: %.2f%%)",
                loss_pct * 100,
                settings.daily_loss_limit * 100,
            )
            return True
        return False

    def _extract_equity(self, account: Any) -> float:
        """Extract equity value from account info response."""
        if isinstance(account, dict):
            return float(account.get("equity", 0) or account.get("portfolio_value", 0) or 100000)
        return 100000.0
