import { useState } from "react";
import { useAnalytics, useDailyPnl } from "../hooks/useAnalytics";
import { useTrades } from "../hooks/useTrades";
import { StatCard } from "../components/StatCard";
import { PnlBarChart } from "../components/PnlBarChart";
import { EquityCurve } from "../components/EquityCurve";
import { ExitsTable } from "../components/ExitsTable";
import { SymbolBreakdown } from "../components/SymbolBreakdown";
import { ErrorState } from "../components/ErrorState";
import { formatUsd, formatPct, pnlColor } from "../lib/utils";

// Calendar days back from today (New York), for both the closed trades and
// the daily ledger.
const RANGES = [
  { label: "7d", days: 7 },
  { label: "30d", days: 30 },
  { label: "90d", days: 90 },
  { label: "1y", days: 365 },
  { label: "All", days: 3650 },
] as const;

export default function Analytics() {
  // 90 days: enough closed round-trips to read a record from.
  const [days, setDays] = useState(90);
  const { data: stats, isLoading, isError, error, refetch } = useAnalytics(days);
  const {
    data: pnl,
    isError: pnlError,
    error: pnlErr,
    refetch: pnlRefetch,
  } = useDailyPnl(days);
  const {
    data: trades,
    isError: tradesError,
    error: tradesErr,
    refetch: tradesRefetch,
  } = useTrades({ limit: 500 });

  if (isError)
    return (
      <div className="p-4 sm:p-6">
        <ErrorState error={error} onRetry={refetch} />
      </div>
    );

  return (
    <div className="space-y-6 p-4 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-white">Day-trader analytics</h1>
          <p className="mt-1 max-w-2xl text-xs text-muted">
            The intraday day-trader's record on its own paper account. The core portfolio has its own
            page.
          </p>
        </div>
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
              label="Closed trades"
              value={stats.total_trades}
              sub={`${stats.wins}W / ${stats.losses}L · last ${days} days`}
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
              label="Closed-trade P&L"
              value={
                <span className={pnlColor(stats.total_pnl)}>
                  {formatUsd(stats.total_pnl)}
                </span>
              }
              sub="realized, on round trips"
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
              label="Best Symbol"
              value={
                <span className="text-accent text-lg">
                  {stats.best_symbol || "N/A"}
                </span>
              }
            />
            <StatCard
              label="Worst Symbol"
              value={
                <span className="text-loss text-lg">
                  {stats.worst_symbol || "N/A"}
                </span>
              }
            />
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <div className="rounded-xl border border-border bg-surface p-4">
              <h3 className="mb-1 text-sm font-medium uppercase tracking-wider text-muted">
                Daily equity change
              </h3>
              <p className="mb-4 text-xs text-muted">
                Ending minus starting account equity: open positions' marks included, so not realized P&L.
              </p>
              {pnlError ? (
                <ErrorState compact error={pnlErr} onRetry={pnlRefetch} />
              ) : pnl ? (
                <PnlBarChart data={pnl} />
              ) : null}
            </div>
            <div className="rounded-xl border border-border bg-surface p-4">
              <h3 className="mb-1 text-sm font-medium uppercase tracking-wider text-muted">
                Cumulative equity change
              </h3>
              <p className="mb-4 text-xs text-muted">Summed over the closed days in the window.</p>
              {pnlError ? (
                <ErrorState compact error={pnlErr} onRetry={pnlRefetch} />
              ) : pnl ? (
                <EquityCurve data={pnl} mode="cumulative" />
              ) : null}
            </div>
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <div className="rounded-xl border border-border bg-surface p-4">
              <h3 className="mb-4 text-sm font-medium uppercase tracking-wider text-muted">
                Exits
              </h3>
              <ExitsTable exits={stats.exits ?? []} />
            </div>
            <div className="rounded-xl border border-border bg-surface p-4">
              <h3 className="mb-4 text-sm font-medium uppercase tracking-wider text-muted">
                P&L by Symbol
                <span className="ml-1 normal-case tracking-normal">· its last 500 trades</span>
              </h3>
              {tradesError ? (
                <ErrorState compact error={tradesErr} onRetry={tradesRefetch} />
              ) : trades ? (
                <SymbolBreakdown trades={trades} />
              ) : null}
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}
