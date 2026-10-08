"""Position and P&L tracking for stock trading."""

import logging
from datetime import date
from typing import Any

from halal_trader.db.repos import StockPnlRepo, TradeRepo
from halal_trader.domain.ports import Broker
from halal_trader.market_hours import today_eastern

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """Tracks stock portfolio state and daily P&L via broker + local DB."""

    _DEFAULT_EQUITY: float = 100_000.0

    def __init__(
        self,
        broker: Broker,
        repo: TradeRepo,
        *,
        daily_loss_limit: float,
        pnl_repo: StockPnlRepo | None = None,
    ) -> None:
        self._daily_loss_limit = daily_loss_limit
        self._starting_equity: float | None = None
        # The Eastern trading date _starting_equity belongs to. A baseline
        # from yesterday must never be measured against today's equity.
        self._starting_date: date | None = None
        self._broker = broker
        self._repo = repo
        # When the caller passes a single shared Repository it satisfies
        # both protocols structurally; ``pnl_repo`` only exists for
        # callers that want to thread a narrower StockPnlRepo separately.
        self._pnl: StockPnlRepo = pnl_repo if pnl_repo is not None else repo  # type: ignore[assignment]

    # ── Broker / DB access ─────────────────────────────────────

    async def _get_equity(self) -> float:
        account = await self._broker.get_account_info()
        return account.effective_equity or self._DEFAULT_EQUITY

    async def _get_today_trades(self) -> list[dict[str, Any]]:
        return await self._repo.get_today_trades()

    async def _persist_day_start(self, equity: float) -> float | None:
        """Persist today's starting equity; return the figure on record.

        Returning the persisted value (when one already exists for today)
        is what keeps the daily loss limit anchored across restarts.
        ``None`` means "no persisted figure", and the fresh equity is used.
        """
        return await self._pnl.start_day(equity)

    async def _persist_day_end(self, equity: float, pnl: float, count: int) -> None:
        await self._pnl.end_day(equity, pnl, count)

    # ── Daily P&L ──────────────────────────────────────────────

    async def record_day_start(self) -> float:
        """Record the starting equity for today. Returns starting equity."""
        equity = await self._get_equity()
        on_record = await self._persist_day_start(equity)
        if on_record is not None and on_record != equity:
            logger.info(
                "Resuming today's baseline $%.2f from the database (equity now $%.2f)",
                on_record,
                equity,
            )
            equity = on_record
        self._starting_equity = equity
        self._starting_date = today_eastern()
        logger.info("Day started with equity: $%.2f", equity)
        return equity

    async def _ensure_baseline(self) -> None:
        """Make sure today's loss-limit baseline exists before it is used.

        Without this a failed pre-market left _starting_equity None and the
        loss limit measured equity against itself -- it could never trip --
        and a baseline left over from a previous day was silently reused.
        """
        if self._starting_equity is None or self._starting_date != today_eastern():
            await self.record_day_start()

    async def record_day_end(self) -> dict[str, Any]:
        """Record end-of-day stats. Returns summary dict.

        The win-rate and best/worst fields feed the Telegram
        daily-summary template.
        """
        equity = await self._get_equity()
        trades = await self._get_today_trades()
        trades_count = len(trades)

        realized_pnl = equity - (self._starting_equity or equity)
        await self._persist_day_end(equity, realized_pnl, trades_count)

        starting = self._starting_equity or equity
        return_pct = (equity - starting) / starting if starting else 0

        summary: dict[str, Any] = {
            "starting_equity": starting,
            "ending_equity": equity,
            "realized_pnl": realized_pnl,
            "return_pct": return_pct,
            "trades_count": trades_count,
        }
        # Win-rate / best+worst symbol derived from today's trades. Only
        # populated when trades include the necessary fields (pnl, symbol).
        wins = [t for t in trades if (t.get("pnl") or 0) > 0]
        losses = [t for t in trades if (t.get("pnl") or 0) < 0]
        if wins or losses:
            total_closed = len(wins) + len(losses)
            summary["win_rate"] = len(wins) / total_closed if total_closed else 0.0
        pnl_by_symbol: dict[str, float] = {}
        for t in trades:
            symbol = t.get("symbol")
            pnl = t.get("pnl")
            if symbol and isinstance(pnl, (int, float)):
                pnl_by_symbol[symbol] = pnl_by_symbol.get(symbol, 0.0) + float(pnl)
        if pnl_by_symbol:
            best = max(pnl_by_symbol.items(), key=lambda kv: kv[1])
            worst = min(pnl_by_symbol.items(), key=lambda kv: kv[1])
            summary["best_symbol"] = best[0]
            summary["best_symbol_pnl"] = best[1]
            if worst[0] != best[0]:
                summary["worst_symbol"] = worst[0]
                summary["worst_symbol_pnl"] = worst[1]

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
        await self._ensure_baseline()
        equity = await self._get_equity()
        starting = self._starting_equity or equity
        return equity - starting

    async def should_halt_trading(self) -> bool:
        """Check if daily loss limit has been breached."""
        pnl = await self.get_current_pnl()
        starting = self._starting_equity or self._DEFAULT_EQUITY
        loss_pct = abs(pnl) / starting if pnl < 0 else 0

        if loss_pct >= self._daily_loss_limit:
            logger.warning(
                "Daily loss limit breached: %.2f%% (limit: %.2f%%)",
                loss_pct * 100,
                self._daily_loss_limit * 100,
            )
            return True
        return False
