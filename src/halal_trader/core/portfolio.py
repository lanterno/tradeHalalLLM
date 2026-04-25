"""Base portfolio tracker with template-method P&L tracking."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from halal_trader.db.repository import Repository

logger = logging.getLogger(__name__)


class BasePortfolioTracker(ABC):
    """Base class for portfolio state and daily P&L tracking.

    Subclasses provide broker-specific hooks for fetching equity, retrieving
    today's trades, and persisting day-start / day-end records.
    """

    _DEFAULT_EQUITY: float = 100_000.0
    _label: str = ""

    def __init__(
        self,
        repo: Repository,
        *,
        daily_loss_limit: float,
    ) -> None:
        self._repo = repo
        self._daily_loss_limit = daily_loss_limit
        self._starting_equity: float | None = None

    # ── Abstract hooks ─────────────────────────────────────────

    @abstractmethod
    async def _get_equity(self, **kwargs: Any) -> float: ...

    @abstractmethod
    async def _get_today_trades(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def _persist_day_start(self, equity: float) -> None: ...

    @abstractmethod
    async def _persist_day_end(self, equity: float, pnl: float, count: int) -> None: ...

    # ── Template methods ───────────────────────────────────────

    async def record_day_start(self) -> float:
        """Record the starting equity for today. Returns starting equity."""
        equity = await self._get_equity()
        self._starting_equity = equity
        await self._persist_day_start(equity)
        logger.info("%sDay started with equity: $%.2f", self._label, equity)
        return equity

    async def record_day_end(self) -> dict[str, Any]:
        """Record end-of-day stats. Returns summary dict."""
        equity = await self._get_equity()
        trades = await self._get_today_trades()
        trades_count = len(trades)

        realized_pnl = equity - (self._starting_equity or equity)
        await self._persist_day_end(equity, realized_pnl, trades_count)

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
            "%sDay ended: $%.2f -> $%.2f (P&L: $%+.2f, %+.2f%%, %d trades)",
            self._label,
            starting,
            equity,
            realized_pnl,
            return_pct * 100,
            trades_count,
        )
        return summary

    async def get_current_pnl(self, **kwargs: Any) -> float:
        """Get the current unrealized + realized P&L for today."""
        equity = await self._get_equity(**kwargs)
        starting = self._starting_equity or equity
        return equity - starting

    async def should_halt_trading(self, **kwargs: Any) -> bool:
        """Check if daily loss limit has been breached."""
        pnl = await self.get_current_pnl(**kwargs)
        starting = self._starting_equity or self._DEFAULT_EQUITY
        loss_pct = abs(pnl) / starting if pnl < 0 else 0

        if loss_pct >= self._daily_loss_limit:
            logger.warning(
                "%sDaily loss limit breached: %.2f%% (limit: %.2f%%)",
                self._label,
                loss_pct * 100,
                self._daily_loss_limit * 100,
            )
            return True
        return False
