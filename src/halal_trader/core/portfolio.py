"""Base portfolio tracker with template-method P&L tracking."""

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BasePortfolioTracker(ABC):
    """Base class for portfolio state and daily P&L tracking.

    Subclasses provide broker-specific hooks for fetching equity, retrieving
    today's trades, and persisting day-start / day-end records. The base
    holds no repo reference — subclasses store their own narrow-typed
    repos and the base reaches them via the abstract hooks.
    """

    _DEFAULT_EQUITY: float = 100_000.0
    _label: str = ""

    def __init__(self, *, daily_loss_limit: float) -> None:
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
        """Record end-of-day stats. Returns summary dict.

        Subclasses may override to enrich the summary (e.g. with win-rate,
        best/worst pair, LLM spend) — those are consumed by the richer
        Telegram daily-summary template.
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
        # Win-rate / best+worst pair derived from today's trades. Only
        # populated when trades include the necessary fields (pnl, pair).
        wins = [t for t in trades if (t.get("pnl") or 0) > 0]
        losses = [t for t in trades if (t.get("pnl") or 0) < 0]
        if wins or losses:
            total_closed = len(wins) + len(losses)
            summary["win_rate"] = len(wins) / total_closed if total_closed else 0.0
        pnl_by_pair: dict[str, float] = {}
        for t in trades:
            pair = t.get("pair") or t.get("symbol")
            pnl = t.get("pnl")
            if pair and isinstance(pnl, (int, float)):
                pnl_by_pair[pair] = pnl_by_pair.get(pair, 0.0) + float(pnl)
        if pnl_by_pair:
            best = max(pnl_by_pair.items(), key=lambda kv: kv[1])
            worst = min(pnl_by_pair.items(), key=lambda kv: kv[1])
            summary["best_pair"] = best[0]
            summary["best_pair_pnl"] = best[1]
            if worst[0] != best[0]:
                summary["worst_pair"] = worst[0]
                summary["worst_pair_pnl"] = worst[1]

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
