"""Crypto portfolio and P&L tracking."""

import logging
from typing import Any

from halal_trader.crypto.exchange import BinanceClient
from halal_trader.domain.models import CryptoBalance
from halal_trader.domain.ports import TradeRepository

logger = logging.getLogger(__name__)

_DEFAULT_EQUITY = 10_000.0


class CryptoPortfolioTracker:
    """Tracks crypto portfolio state and daily P&L."""

    def __init__(
        self,
        broker: BinanceClient,
        repo: TradeRepository,
        *,
        daily_loss_limit: float,
    ) -> None:
        self._broker = broker
        self._repo = repo
        self._daily_loss_limit = daily_loss_limit
        self._starting_equity: float | None = None

    async def record_day_start(self) -> float:
        """Record the starting USDT-equivalent equity for today."""
        account = await self._broker.get_account()
        equity = account.total_balance_usdt or _DEFAULT_EQUITY
        self._starting_equity = equity
        await self._repo.start_crypto_day(equity)
        logger.info("Crypto day started with equity: $%.2f USDT", equity)
        return equity

    async def record_day_end(self) -> dict[str, Any]:
        """Record end-of-day stats. Returns summary dict."""
        account = await self._broker.get_account()
        equity = account.total_balance_usdt or _DEFAULT_EQUITY
        trades = await self._repo.get_today_crypto_trades()
        trades_count = len(trades)

        realized_pnl = equity - (self._starting_equity or equity)
        await self._repo.end_crypto_day(
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
            "Crypto day ended: $%.2f -> $%.2f (P&L: $%+.2f, %+.2f%%, %d trades)",
            starting,
            equity,
            realized_pnl,
            return_pct * 100,
            trades_count,
        )
        return summary

    async def get_current_pnl(self) -> float:
        """Get current P&L for today."""
        account = await self._broker.get_account()
        equity = account.total_balance_usdt or _DEFAULT_EQUITY
        starting = self._starting_equity or equity
        return equity - starting

    async def get_balances_summary(self) -> list[CryptoBalance]:
        """Get all current balances."""
        return await self._broker.get_balances()

    async def should_halt_trading(self) -> bool:
        """Check if daily loss limit has been breached."""
        pnl = await self.get_current_pnl()
        starting = self._starting_equity or _DEFAULT_EQUITY
        loss_pct = abs(pnl) / starting if pnl < 0 else 0

        if loss_pct >= self._daily_loss_limit:
            logger.warning(
                "Crypto daily loss limit breached: %.2f%% (limit: %.2f%%)",
                loss_pct * 100,
                self._daily_loss_limit * 100,
            )
            return True
        return False

    def format_positions_for_prompt(self, balances: list[CryptoBalance]) -> str:
        """Format current balances into text for the LLM prompt."""
        non_usdt = [b for b in balances if b.asset != "USDT" and b.free > 0]
        if not non_usdt:
            return "No open positions."
        lines = []
        for b in non_usdt:
            lines.append(f"  {b.asset}: {b.free:.8f} (locked: {b.locked:.8f})")
        return "\n".join(lines)
