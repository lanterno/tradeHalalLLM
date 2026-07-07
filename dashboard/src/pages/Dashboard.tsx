import { useAnalytics, useDailyPnl } from "../hooks/useAnalytics";
import { useTrades } from "../hooks/useTrades";
import { useStockOfTheDay } from "../hooks/useRecommendation";
import { StatCard } from "../components/StatCard";
import { EquityCurve } from "../components/EquityCurve";
import { TradesTable } from "../components/TradesTable";
import { RecommendationCard } from "../components/RecommendationCard";
import { ErrorState } from "../components/ErrorState";
import { formatUsd, formatPct, pnlColor } from "../lib/utils";

export default function Dashboard() {
  const {
    data: stats,
    isLoading: statsLoading,
    isError: statsIsError,
    error: statsError,
    refetch: statsRefetch,
  } = useAnalytics(7);
  const {
    data: pnl,
    isError: pnlIsError,
    error: pnlError,
    refetch: pnlRefetch,
  } = useDailyPnl(30);
  const {
    data: trades,
    isError: tradesIsError,
    error: tradesError,
    refetch: tradesRefetch,
  } = useTrades({ limit: 10 });
  const { data: pick, isLoading: pickLoading } = useStockOfTheDay();

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <p className="text-xs text-muted">Last 7 days</p>
      </div>

      {/* Daily halal recommendation (advisory) */}
      <RecommendationCard pick={pick} isLoading={pickLoading} />

      {/* Stats grid */}
      {statsIsError ? (
        <div className="rounded-xl border border-border bg-surface p-4">
          <ErrorState compact error={statsError} onRetry={statsRefetch} />
        </div>
      ) : statsLoading ? (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
          {Array.from({ length: 6 }).map((_, i) => (
            <div
              key={i}
              className="h-24 animate-pulse rounded-xl border border-border bg-surface"
            />
          ))}
        </div>
      ) : stats ? (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
          <StatCard
            label="Total P&L"
            value={
              <span className={pnlColor(stats.total_pnl)}>
                {formatUsd(stats.total_pnl)}
              </span>
            }
          />
          <StatCard
            label="Win Rate"
            value={
              <span className={stats.win_rate >= 0.5 ? "text-accent" : "text-loss"}>
                {formatPct(stats.win_rate)}
              </span>
            }
            sub={`${stats.wins}W / ${stats.losses}L`}
          />
          <StatCard label="Total Trades" value={stats.total_trades} />
          <StatCard
            label="Profit Factor"
            value={
              stats.profit_factor === Infinity
                ? "∞"
                : stats.profit_factor.toFixed(2)
            }
          />
          <StatCard
            label="Max Drawdown"
            value={
              <span className="text-loss">
                {formatPct(stats.max_drawdown_pct)}
              </span>
            }
          />
          <StatCard
            label="Streak"
            value={`${stats.streak} ${stats.streak_type}`}
            sub={
              stats.streak_type === "wins" ? (
                <span className="text-accent">Winning</span>
              ) : stats.streak_type === "losses" ? (
                <span className="text-loss">Losing</span>
              ) : null
            }
          />
        </div>
      ) : null}

      {/* Equity curve */}
      <div className="rounded-xl border border-border bg-surface p-4">
        <h2 className="mb-4 text-sm font-medium uppercase tracking-wider text-muted">
          Equity Curve (30d)
        </h2>
        {pnlIsError ? (
          <ErrorState compact error={pnlError} onRetry={pnlRefetch} />
        ) : pnl ? (
          <EquityCurve data={pnl} />
        ) : (
          <p className="text-sm text-muted">Loading…</p>
        )}
      </div>

      {/* Recent trades */}
      <div className="rounded-xl border border-border bg-surface p-4">
        <h2 className="mb-4 text-sm font-medium uppercase tracking-wider text-muted">
          Recent Trades
        </h2>
        {tradesIsError ? (
          <ErrorState compact error={tradesError} onRetry={tradesRefetch} />
        ) : trades ? (
          <TradesTable trades={trades} compact />
        ) : (
          <p className="text-sm text-muted">Loading…</p>
        )}
      </div>
    </div>
  );
}
