"""Asset-class-aware performance analytics.

Promoted from :mod:`crypto.analytics` so the stock pipeline can consume
the same surface. The math is identical between markets — closed-trade
P&L, win rate, profit factor, drawdown, streak — the only thing that
varies is *which* completed-round-trip getter to call.

The crypto-specific class still exists at ``crypto.analytics`` for
existing callers (cycle, web/app); over time those can adopt this
module directly.
"""

from __future__ import annotations

from typing import Literal

from halal_trader.crypto.analytics import PerformanceAnalytics, PerformanceStats
from halal_trader.db.repository import Repository

AssetClass = Literal["crypto", "stock"]


class CrossAssetAnalytics:
    """Wraps :class:`PerformanceAnalytics` with an asset-class switch.

    The crypto class hard-codes ``repo.get_completed_round_trips``; this
    facade swaps in the stock equivalent when ``asset_class="stock"``.
    Format/output is identical so dashboards can render either with the
    same template.
    """

    def __init__(self, repo: Repository, *, asset_class: AssetClass = "crypto") -> None:
        self._repo = repo
        self._asset_class = asset_class
        self._inner = PerformanceAnalytics(repo)

    async def compute_stats(self, lookback_days: int = 7) -> PerformanceStats:
        if self._asset_class == "stock":
            round_trips = await self._repo.get_completed_stock_round_trips(
                limit=500, lookback_days=lookback_days
            )
            return _stats_from_trips(self._inner, round_trips)
        return await self._inner.compute_stats(lookback_days=lookback_days)

    def format_for_prompt(self, stats: PerformanceStats) -> str:
        return self._inner.format_for_prompt(stats)


def _stats_from_trips(inner: PerformanceAnalytics, round_trips: list[dict]) -> PerformanceStats:
    """Recreate the crypto analytics flow over a precomputed round-trip list.

    We can't reuse ``inner.compute_stats`` directly because it always
    fetches crypto trades. Re-running the same arithmetic on the trip
    list keeps a single source of truth (the inner methods do the
    drawdown / streak math); we just bypass the fetch step.
    """
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
        pnl = rt["pnl"]
        pnl_pct = rt["pnl_pct"]
        pair = rt["pair"]
        duration = rt["duration_minutes"]
        reason = rt.get("exit_reason") or "unknown"

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

    if pair_pnl:
        stats.best_pair = max(pair_pnl, key=pair_pnl.get)
        stats.best_pair_pnl = pair_pnl[stats.best_pair]
        stats.worst_pair = min(pair_pnl, key=pair_pnl.get)
        stats.worst_pair_pnl = pair_pnl[stats.worst_pair]

    stats.max_drawdown_pct = inner._compute_max_drawdown(round_trips)
    stats.streak, stats.streak_type = inner._compute_streak(round_trips)
    return stats
