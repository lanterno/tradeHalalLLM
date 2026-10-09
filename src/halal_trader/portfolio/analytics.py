"""Performance analytics — computes rolling trading metrics from completed round-trips.

Reads closed stock round-trips (``TradeRepo.get_completed_stock_round_trips``)
and reduces them to win rate, profit factor, drawdown, streak and the
per-symbol best/worst. The stock cycle's ``BuildPerformanceStage`` renders
the result into the prompt; ``/api/analytics`` serves it to the dashboard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from halal_trader.db.repos import TradeRepo

logger = logging.getLogger(__name__)

# The early-exit question (should an exit come this soon after the entry?).
FIRST_HOUR_MINUTES = 60.0


@dataclass
class ExitStats:
    """How the round trips that closed one way did: the evidence for an exit rule."""

    reason: str
    trades: int
    avg_pct: float
    total_pnl: float
    win_rate: float
    avg_hold_minutes: float
    first_hour_trades: int  # closed within FIRST_HOUR_MINUTES of the entry
    first_hour_avg_pct: float | None


def exit_breakdown(round_trips: list[dict[str, Any]]) -> list[ExitStats]:
    """Per exit reason, most frequent first."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for rt in round_trips:
        groups.setdefault(rt.get("exit_reason") or "unknown", []).append(rt)
    out = []
    for reason, rts in groups.items():
        pcts = [float(rt["pnl_pct"]) for rt in rts]
        early = [float(rt["pnl_pct"]) for rt in rts if rt["duration_minutes"] < FIRST_HOUR_MINUTES]
        out.append(
            ExitStats(
                reason=reason,
                trades=len(rts),
                avg_pct=sum(pcts) / len(pcts),
                total_pnl=sum(float(rt["pnl"]) for rt in rts),
                win_rate=sum(1 for p in pcts if p > 0) / len(pcts),
                avg_hold_minutes=sum(float(rt["duration_minutes"]) for rt in rts) / len(rts),
                first_hour_trades=len(early),
                first_hour_avg_pct=sum(early) / len(early) if early else None,
            )
        )
    return sorted(out, key=lambda e: (-e.trades, e.reason))


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
    best_symbol: str = ""
    best_symbol_pnl: float = 0.0
    worst_symbol: str = ""
    worst_symbol_pnl: float = 0.0
    streak: int = 0
    streak_type: str = ""
    by_exit_reason: dict[str, int] = field(default_factory=dict)
    exits: list[ExitStats] = field(default_factory=list)


class PerformanceAnalytics:
    """Computes trading performance metrics from the database."""

    def __init__(self, repo: TradeRepo) -> None:
        self._repo = repo

    async def compute_stats(self, lookback_days: int = 7) -> PerformanceStats:
        """Compute rolling performance metrics over the last N days."""
        round_trips = await self._repo.get_completed_stock_round_trips(
            limit=500, lookback_days=lookback_days
        )
        return self.stats_from_round_trips(round_trips)

    def stats_from_round_trips(self, round_trips: list[dict[str, Any]]) -> PerformanceStats:
        """Reduce a list of round-trip dicts to :class:`PerformanceStats`."""
        stats = PerformanceStats()
        if not round_trips:
            return stats

        stats.total_trades = len(round_trips)

        win_pcts: list[float] = []
        loss_pcts: list[float] = []
        gross_wins = 0.0
        gross_losses = 0.0
        symbol_pnl: dict[str, float] = {}
        durations: list[float] = []
        exit_reasons: dict[str, int] = {}

        for rt in round_trips:
            pnl: float = rt["pnl"]
            pnl_pct: float = rt["pnl_pct"]
            symbol: str = rt["symbol"]
            duration: float = rt["duration_minutes"]
            reason: str = rt.get("exit_reason") or "unknown"

            stats.total_pnl += pnl
            symbol_pnl[symbol] = symbol_pnl.get(symbol, 0) + pnl
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
        stats.exits = exit_breakdown(round_trips)

        # Best/worst symbol
        if symbol_pnl:
            stats.best_symbol = max(symbol_pnl, key=lambda k: symbol_pnl[k])
            stats.best_symbol_pnl = symbol_pnl[stats.best_symbol]
            stats.worst_symbol = min(symbol_pnl, key=lambda k: symbol_pnl[k])
            stats.worst_symbol_pnl = symbol_pnl[stats.worst_symbol]

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

        if stats.best_symbol:
            lines.append(
                f"Best symbol: {stats.best_symbol} (${stats.best_symbol_pnl:+,.2f}) | "
                f"Worst symbol: {stats.worst_symbol} (${stats.worst_symbol_pnl:+,.2f})"
            )

        if stats.by_exit_reason:
            reasons = ", ".join(f"{k}: {v}" for k, v in stats.by_exit_reason.items())
            lines.append(f"Exit reasons: {reasons}")

        return "\n".join(lines)

    @staticmethod
    def _compute_max_drawdown(round_trips: list[dict[str, Any]]) -> float:
        """Compute max drawdown as a fraction from chronological round-trips.

        Compounds each trade's ``pnl_pct`` (already a fraction in the
        round-trip dicts) into a unit equity curve and takes
        the worst peak-to-trough drop relative to the running peak — bounded
        in [0, 1) by construction.

        The previous version summed DOLLAR pnl and normalized by the peak of
        cumulative P&L (not equity). With small per-trade P&L the cumulative
        peak is tiny, so any losing streak that drove the curve below zero
        produced ratios > 1 — the live stock prompt printed "Max drawdown:
        242.52%" (peak +$50, trough −$71), telling the LLM the account was
        deep underwater while real equity moved a fraction of a percent. It
        also degenerated to 0.0 whenever the curve never went positive,
        hiding genuine drawdowns. Per-trade compounding slightly overstates
        true account-equity drawdown (it assumes sequential full-capital
        allocation) — acceptable conservatism for a prompt signal.
        """
        sorted_trips = sorted(round_trips, key=lambda r: r.get("closed_at") or "")
        if not sorted_trips:
            return 0.0

        equity = 1.0
        peak = 1.0
        max_dd = 0.0

        for rt in sorted_trips:
            equity *= 1.0 + rt.get("pnl_pct", 0.0)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd

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
