import { useMemo } from "react";
import { useTrades } from "../hooks/useTrades";
import { StatCard } from "./StatCard";
import { ErrorState } from "./ErrorState";
import type { Trade } from "../api/types";
import { cn, formatTime } from "../lib/utils";

const FILLED = new Set(["filled", "closed"]);

/**
 * The EOD close-all's sell rows are bookkeeping, not orders: the executor
 * writes one per flattened position with submitted_at == filled_at (a zero
 * latency) and the close price as the fill. Counted, they drag the latency
 * and slippage figures toward zero.
 */
function isSyntheticExit(t: Trade): boolean {
  if (t.llm_reasoning?.startsWith("EOD close-all")) return true;
  return t.side === "sell" && !!t.submitted_at && t.submitted_at === t.filled_at;
}

function latencyMs(t: Trade): number | null {
  if (!t.submitted_at || !t.filled_at) return null;
  const ms = new Date(t.filled_at).getTime() - new Date(t.submitted_at).getTime();
  return Number.isFinite(ms) && ms >= 0 ? ms : null;
}

function slippageBps(t: Trade): number | null {
  // paper_slippage_pct is a fraction (e.g. 3.2e-5); ×10000 → basis points.
  if (t.paper_slippage_pct != null) return t.paper_slippage_pct * 10000;
  if (t.filled_price != null && t.price) {
    return ((t.filled_price - t.price) / t.price) * 10000;
  }
  return null;
}

function median(xs: number[]): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

function fmtLatency(ms: number | null): string {
  if (ms == null) return "—";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

/**
 * Execution quality for recent orders: fill latency (submitted → filled),
 * slippage (filled vs intended price), and how many orders never filled
 * (surfaces the market-open cold-start fill-timeout pattern). Self-contained
 * over the last 100 trades.
 */
export function FillQuality() {
  const { data, isLoading, isError, error, refetch } = useTrades({ limit: 100 });

  const stats = useMemo(() => {
    const all = data ?? [];
    const rows = all.filter((t) => !isSyntheticExit(t));
    const filled = rows.filter(
      (t) => FILLED.has(t.status) && (t.filled_quantity ?? t.quantity) > 0,
    );
    const unfilled = rows.filter((t) => !FILLED.has(t.status));
    const latencies = filled
      .map(latencyMs)
      .filter((x): x is number => x != null);
    const slippages = filled
      .map(slippageBps)
      .filter((x): x is number => x != null);
    const avgSlip = slippages.length
      ? slippages.reduce((a, b) => a + b, 0) / slippages.length
      : null;
    return {
      total: rows.length,
      synthetic: all.length - rows.length,
      filledCount: filled.length,
      unfilledCount: unfilled.length,
      medLatency: median(latencies),
      maxLatency: latencies.length ? Math.max(...latencies) : null,
      avgSlip,
      recent: filled.slice(0, 10),
    };
  }, [data]);

  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <h3 className="mb-1 text-sm font-medium uppercase tracking-wider text-muted">
        Fill Quality
      </h3>
      <p className="mb-4 text-xs text-muted">
        The day-trader's execution latency + slippage over its last {stats.total} orders
        {stats.synthetic
          ? ` (${stats.synthetic} synthetic end-of-day close-all row${stats.synthetic === 1 ? "" : "s"} left out: bookkeeping, not orders)`
          : ""}
        .
      </p>
      {isError ? (
        <ErrorState compact error={error} onRetry={refetch} />
      ) : isLoading ? (
        <p className="text-sm text-muted">Loading…</p>
      ) : !stats.total ? (
        <p className="text-sm text-muted">No orders yet.</p>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
            <StatCard label="Fills" value={stats.filledCount} />
            <StatCard
              label="Unfilled"
              value={
                <span className={cn(stats.unfilledCount ? "text-warning" : "text-accent")}>
                  {stats.unfilledCount}
                </span>
              }
              sub="never reached fill"
            />
            <StatCard label="Median latency" value={fmtLatency(stats.medLatency)} />
            <StatCard
              label="Max latency"
              value={
                <span className={cn((stats.maxLatency ?? 0) > 10_000 && "text-warning")}>
                  {fmtLatency(stats.maxLatency)}
                </span>
              }
            />
            <StatCard
              label="Avg slippage"
              value={stats.avgSlip == null ? "—" : `${stats.avgSlip.toFixed(2)} bps`}
            />
          </div>

          {stats.recent.length > 0 && (
            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                    <th className="px-3 py-2">Time</th>
                    <th className="px-3 py-2">Symbol</th>
                    <th className="px-3 py-2">Side</th>
                    <th className="px-3 py-2 text-right">Latency</th>
                    <th className="px-3 py-2 text-right">Slippage</th>
                  </tr>
                </thead>
                <tbody>
                  {stats.recent.map((t) => {
                    const slip = slippageBps(t);
                    return (
                      <tr
                        key={t.id}
                        className="border-b border-border/50 hover:bg-surface-hover/50 transition-colors"
                      >
                        <td className="whitespace-nowrap px-3 py-2 text-muted">
                          {formatTime(t.filled_at ?? t.timestamp)}
                        </td>
                        <td className="px-3 py-2 font-medium">{t.symbol}</td>
                        <td className="px-3 py-2">{t.side}</td>
                        <td className="px-3 py-2 text-right font-mono">
                          {fmtLatency(latencyMs(t))}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-muted">
                          {slip == null ? "—" : `${slip.toFixed(2)} bps`}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
