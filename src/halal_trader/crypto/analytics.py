"""Performance analytics — computes rolling trading metrics from completed round-trips."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from halal_trader.db.repos import CryptoTradeRepo

logger = logging.getLogger(__name__)


@dataclass
class PerformanceStats:
    """Aggregated performance metrics over a lookback window."""

    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    total_pnl: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_hold_minutes: float = 0.0
    best_pair: str = ""
    best_pair_pnl: float = 0.0
    worst_pair: str = ""
    worst_pair_pnl: float = 0.0
    streak: int = 0
    streak_type: str = ""
    by_exit_reason: dict[str, int] = field(default_factory=dict)


class PerformanceAnalytics:
    """Computes trading performance metrics from the database."""

    def __init__(self, repo: CryptoTradeRepo) -> None:
        self._repo = repo

    async def compute_stats(self, lookback_days: int = 7) -> PerformanceStats:
        """Compute rolling performance metrics over the last N days."""
        round_trips = await self._repo.get_completed_round_trips(
            limit=500, lookback_days=lookback_days
        )

        stats = PerformanceStats()
        if not round_trips:
            return stats

        stats.total_trades = len(round_trips)

        win_pcts: list[float] = []
        loss_pcts: list[float] = []
        gross_wins = 0.0
        gross_losses = 0.0
        pair_pnl: dict[str, float] = {}
        durations: list[float] = []
        exit_reasons: dict[str, int] = {}

        for rt in round_trips:
            pnl: float = rt["pnl"]
            pnl_pct: float = rt["pnl_pct"]
            pair: str = rt["pair"]
            duration: float = rt["duration_minutes"]
            reason: str = rt.get("exit_reason") or "unknown"

            stats.total_pnl += pnl
            pair_pnl[pair] = pair_pnl.get(pair, 0) + pnl
            durations.append(duration)
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

            if pnl > 0:
                stats.wins += 1
                win_pcts.append(pnl_pct)
                gross_wins += pnl
            else:
                stats.losses += 1
                loss_pcts.append(pnl_pct)
                gross_losses += abs(pnl)

        stats.win_rate = stats.wins / stats.total_trades if stats.total_trades else 0
        stats.avg_win_pct = sum(win_pcts) / len(win_pcts) if win_pcts else 0
        stats.avg_loss_pct = sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0
        stats.profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")
        stats.avg_hold_minutes = sum(durations) / len(durations) if durations else 0
        stats.by_exit_reason = exit_reasons

        # Best/worst pair
        if pair_pnl:
            stats.best_pair = max(pair_pnl, key=pair_pnl.get)
            stats.best_pair_pnl = pair_pnl[stats.best_pair]
            stats.worst_pair = min(pair_pnl, key=pair_pnl.get)
            stats.worst_pair_pnl = pair_pnl[stats.worst_pair]

        # Max drawdown (peak-to-trough on cumulative P&L)
        stats.max_drawdown_pct = self._compute_max_drawdown(round_trips)

        # Current streak
        stats.streak, stats.streak_type = self._compute_streak(round_trips)

        return stats

    def format_for_prompt(self, stats: PerformanceStats) -> str:
        """Format performance stats as a text block for the LLM prompt."""
        if stats.total_trades == 0:
            return "No completed trades yet — no performance data available."

        hold_str = f"{stats.avg_hold_minutes:.0f}m"
        if stats.avg_hold_minutes >= 60:
            hold_str = f"{stats.avg_hold_minutes / 60:.1f}h"

        lines = [
            f"Total trades: {stats.total_trades} | "
            f"Win rate: {stats.win_rate:.0%} | "
            f"Avg win: {stats.avg_win_pct:+.2%} | "
            f"Avg loss: {stats.avg_loss_pct:+.2%}",
            f"Profit factor: {stats.profit_factor:.1f} | "
            f"Max drawdown: {stats.max_drawdown_pct:.2%} | "
            f"Total P&L: ${stats.total_pnl:+,.2f}",
            f"Avg hold time: {hold_str} | Current streak: {stats.streak} {stats.streak_type}",
        ]

        if stats.best_pair:
            lines.append(
                f"Best pair: {stats.best_pair} (${stats.best_pair_pnl:+,.2f}) | "
                f"Worst pair: {stats.worst_pair} (${stats.worst_pair_pnl:+,.2f})"
            )

        if stats.by_exit_reason:
            reasons = ", ".join(f"{k}: {v}" for k, v in stats.by_exit_reason.items())
            lines.append(f"Exit reasons: {reasons}")

        return "\n".join(lines)

    @staticmethod
    def _compute_max_drawdown(round_trips: list[dict[str, Any]]) -> float:
        """Compute max drawdown percentage from chronological round-trips."""
        sorted_trips = sorted(round_trips, key=lambda r: r.get("closed_at") or "")
        if not sorted_trips:
            return 0.0

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for rt in sorted_trips:
            cumulative += rt["pnl"]
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        # Express as percentage of peak (or initial equity if peak is 0)
        if peak > 0:
            return max_dd / peak
        return 0.0

    @staticmethod
    def _compute_streak(round_trips: list[dict[str, Any]]) -> tuple[int, str]:
        """Compute the current win/loss streak from most recent trades."""
        sorted_trips = sorted(round_trips, key=lambda r: r.get("closed_at") or "", reverse=True)
        if not sorted_trips:
            return 0, ""

        first_win = sorted_trips[0]["pnl"] > 0
        streak_type = "wins" if first_win else "losses"
        count = 0

        for rt in sorted_trips:
            is_win = rt["pnl"] > 0
            if is_win == first_win:
                count += 1
            else:
                break

        return count, streak_type
