import { useAnalytics, useDailyPnl } from "../hooks/useAnalytics";
import { useTrades } from "../hooks/useTrades";
import { StatCard } from "../components/StatCard";
import { EquityCurve } from "../components/EquityCurve";
import { TradesTable } from "../components/TradesTable";
import { formatUsd, formatPct, pnlColor } from "../lib/utils";

export default function Dashboard() {
  const { data: stats, isLoading: statsLoading } = useAnalytics(7);
  const { data: pnl } = useDailyPnl(30);
  const { data: trades } = useTrades({ limit: 10 });

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <p className="text-xs text-muted">Last 7 days</p>
      </div>

      {/* Stats grid */}
      {statsLoading ? (
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
        {pnl ? <EquityCurve data={pnl} /> : <p className="text-sm text-muted">Loading...</p>}
      </div>

      {/* Recent trades */}
      <div className="rounded-xl border border-border bg-surface p-4">
        <h2 className="mb-4 text-sm font-medium uppercase tracking-wider text-muted">
          Recent Trades
        </h2>
        {trades ? (
          <TradesTable trades={trades} compact />
        ) : (
          <p className="text-sm text-muted">Loading...</p>
        )}
      </div>
    </div>
  );
}
