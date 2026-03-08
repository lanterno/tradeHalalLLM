import { useState } from "react";
import { useAnalytics, useDailyPnl } from "../hooks/useAnalytics";
import { useTrades } from "../hooks/useTrades";
import { StatCard } from "../components/StatCard";
import { PnlBarChart } from "../components/PnlBarChart";
import { EquityCurve } from "../components/EquityCurve";
import { ExitReasonsChart } from "../components/ExitReasonsChart";
import { PairBreakdown } from "../components/PairBreakdown";
import { formatUsd, formatPct, pnlColor } from "../lib/utils";

const RANGES = [
  { label: "7d", days: 7 },
  { label: "30d", days: 30 },
  { label: "90d", days: 90 },
  { label: "All", days: 365 },
] as const;

export default function Analytics() {
  const [days, setDays] = useState(30);
  const { data: stats, isLoading } = useAnalytics(days);
  const { data: pnl } = useDailyPnl(days);
  const { data: trades } = useTrades({ limit: 500 });

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Analytics</h1>
        <div className="flex gap-1 rounded-lg border border-border bg-surface p-0.5">
          {RANGES.map((r) => (
            <button
              key={r.days}
              onClick={() => setDays(r.days)}
              className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                days === r.days
                  ? "bg-accent/15 text-accent"
                  : "text-muted hover:text-white"
              }`}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4 lg:grid-cols-5">
          {Array.from({ length: 10 }).map((_, i) => (
            <div
              key={i}
              className="h-24 animate-pulse rounded-xl border border-border bg-surface"
            />
          ))}
        </div>
      ) : stats ? (
        <>
          {/* KPI Cards */}
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4 lg:grid-cols-5">
            <StatCard
              label="Total Trades"
              value={stats.total_trades}
              sub={`${stats.wins}W / ${stats.losses}L`}
            />
            <StatCard
              label="Win Rate"
              value={
                <span className={stats.win_rate >= 0.5 ? "text-accent" : "text-loss"}>
                  {formatPct(stats.win_rate)}
                </span>
              }
            />
            <StatCard
              label="Total P&L"
              value={
                <span className={pnlColor(stats.total_pnl)}>
                  {formatUsd(stats.total_pnl)}
                </span>
              }
            />
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
              label="Avg Win"
              value={
                <span className="text-accent">
                  {formatPct(stats.avg_win_pct)}
                </span>
              }
            />
            <StatCard
              label="Avg Loss"
              value={
                <span className="text-loss">
                  {formatPct(stats.avg_loss_pct)}
                </span>
              }
            />
            <StatCard
              label="Avg Hold"
              value={
                stats.avg_hold_minutes >= 60
                  ? `${(stats.avg_hold_minutes / 60).toFixed(1)}h`
                  : `${stats.avg_hold_minutes.toFixed(0)}m`
              }
            />
            <StatCard
              label="Best Pair"
              value={
                <span className="text-accent text-lg">
                  {stats.best_pair || "N/A"}
                </span>
              }
            />
            <StatCard
              label="Worst Pair"
              value={
                <span className="text-loss text-lg">
                  {stats.worst_pair || "N/A"}
                </span>
              }
            />
          </div>

          {/* Charts */}
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="rounded-xl border border-border bg-surface p-4">
              <h3 className="mb-4 text-sm font-medium uppercase tracking-wider text-muted">
                Daily P&L
              </h3>
              {pnl ? <PnlBarChart data={pnl} /> : null}
            </div>
            <div className="rounded-xl border border-border bg-surface p-4">
              <h3 className="mb-4 text-sm font-medium uppercase tracking-wider text-muted">
                Cumulative P&L
              </h3>
              {pnl ? <EquityCurve data={pnl} /> : null}
            </div>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <div className="rounded-xl border border-border bg-surface p-4">
              <h3 className="mb-4 text-sm font-medium uppercase tracking-wider text-muted">
                Exit Reasons
              </h3>
              <ExitReasonsChart data={stats.by_exit_reason} />
            </div>
            <div className="rounded-xl border border-border bg-surface p-4">
              <h3 className="mb-4 text-sm font-medium uppercase tracking-wider text-muted">
                P&L by Pair
              </h3>
              {trades ? <PairBreakdown trades={trades} /> : null}
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}
