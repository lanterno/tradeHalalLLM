import { useMemo } from "react";
import { usePositions } from "../hooks/usePositions";
import { usePriceStream } from "../hooks/usePriceStream";
import { StatCard } from "../components/StatCard";
import { cn, formatUsd, formatQty, formatTime, pnlColor } from "../lib/utils";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";

const COLORS = ["#4ade80", "#60a5fa", "#facc15", "#c084fc", "#fb923c", "#f87171"];

export default function Positions() {
  const { data: positions, isLoading } = usePositions();
  const symbols = useMemo(
    () => (positions ?? []).map((p) => p.pair),
    [positions],
  );
  const { prices, connected } = usePriceStream(symbols);

  const enriched = useMemo(() => {
    if (!positions) return [];
    return positions.map((p) => {
      const current = prices[p.pair] ?? p.current_price ?? p.entry_price;
      const unrealizedPnl = (current - p.entry_price) * p.quantity;
      const unrealizedPct = p.entry_price
        ? (current - p.entry_price) / p.entry_price
        : 0;
      return { ...p, current_price: current, unrealized_pnl: unrealizedPnl, unrealized_pnl_pct: unrealizedPct };
    });
  }, [positions, prices]);

  const totalUnrealized = enriched.reduce(
    (s, p) => s + (p.unrealized_pnl ?? 0),
    0,
  );

  const allocationData = enriched.map((p) => ({
    name: p.pair,
    value: p.entry_price * p.quantity,
  }));

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Open Positions</h1>
        <div className="flex items-center gap-2 text-xs text-muted">
          <span
            className={cn(
              "h-2 w-2 rounded-full",
              connected ? "bg-accent animate-pulse" : "bg-muted",
            )}
          />
          {connected ? "Live prices" : "Reconnecting..."}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
        <StatCard
          label="Open Positions"
          value={enriched.length}
        />
        <StatCard
          label="Unrealized P&L"
          value={
            <span className={pnlColor(totalUnrealized)}>
              {formatUsd(totalUnrealized)}
            </span>
          }
        />
        <StatCard
          label="Total Exposure"
          value={formatUsd(
            enriched.reduce((s, p) => s + p.entry_price * p.quantity, 0),
          )}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Positions table */}
        <div className="lg:col-span-2 rounded-xl border border-border bg-surface p-4">
          {isLoading ? (
            <p className="py-8 text-center text-sm text-muted">Loading...</p>
          ) : !enriched.length ? (
            <p className="py-8 text-center text-sm text-muted">
              No open positions.
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                    <th className="px-3 py-2">Pair</th>
                    <th className="px-3 py-2 text-right">Qty</th>
                    <th className="px-3 py-2 text-right">Entry</th>
                    <th className="px-3 py-2 text-right">Current</th>
                    <th className="px-3 py-2 text-right">P&L</th>
                    <th className="px-3 py-2 text-right">SL</th>
                    <th className="px-3 py-2 text-right">TP</th>
                    <th className="px-3 py-2">Opened</th>
                  </tr>
                </thead>
                <tbody>
                  {enriched.map((p) => (
                    <tr
                      key={p.id}
                      className="border-b border-border/50 hover:bg-surface-hover/50 transition-colors"
                    >
                      <td className="px-3 py-2 font-medium">{p.pair}</td>
                      <td className="px-3 py-2 text-right font-mono">
                        {formatQty(p.quantity)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono">
                        {formatUsd(p.entry_price)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono">
                        {formatUsd(p.current_price ?? 0)}
                      </td>
                      <td
                        className={cn(
                          "px-3 py-2 text-right font-mono font-semibold",
                          pnlColor(p.unrealized_pnl ?? 0),
                        )}
                      >
                        {formatUsd(p.unrealized_pnl ?? 0)}
                        <span className="ml-1 text-[10px] font-normal">
                          ({(p.unrealized_pnl_pct! * 100).toFixed(2)}%)
                        </span>
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-loss">
                        {p.stop_loss ? formatUsd(p.stop_loss) : "—"}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-accent">
                        {p.target_price ? formatUsd(p.target_price) : "—"}
                      </td>
                      <td className="px-3 py-2 text-muted whitespace-nowrap">
                        {formatTime(p.timestamp)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Allocation pie */}
        <div className="rounded-xl border border-border bg-surface p-4">
          <h3 className="mb-4 text-sm font-medium uppercase tracking-wider text-muted">
            Allocation
          </h3>
          {allocationData.length ? (
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie
                  data={allocationData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={80}
                  paddingAngle={3}
                  dataKey="value"
                  stroke="none"
                >
                  {allocationData.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: "#111118",
                    border: "1px solid #1a1a2e",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={(value) => [formatUsd(Number(value)), "Value"]}
                />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <p className="py-12 text-center text-sm text-muted">No positions.</p>
          )}
        </div>
      </div>
    </div>
  );
}
