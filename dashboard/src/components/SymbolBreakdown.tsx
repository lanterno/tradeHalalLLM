import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  CartesianGrid,
} from "recharts";
import type { Trade } from "../api/types";
import { formatUsd } from "../lib/utils";
import { AXIS_TICK, CHART, CHART_TOOLTIP, pnlFill } from "../lib/charts";

interface SymbolBreakdownProps {
  trades: Trade[];
}

export function SymbolBreakdown({ trades }: SymbolBreakdownProps) {
  const pnlBySymbol: Record<string, number> = {};
  for (const t of trades) {
    const entry = t.filled_price ?? t.price;
    if (t.exit_price != null && entry != null) {
      const pnl = (t.exit_price - entry) * t.quantity;
      pnlBySymbol[t.symbol] = (pnlBySymbol[t.symbol] ?? 0) + pnl;
    }
  }

  const data = Object.entries(pnlBySymbol)
    .map(([symbol, pnl]) => ({ symbol, pnl }))
    .sort((a, b) => b.pnl - a.pnl);

  if (!data.length) {
    return (
      <p className="py-12 text-center text-sm text-muted">
        No per-symbol P&L yet.
      </p>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={Math.max(200, data.length * 36)}>
      <BarChart data={data} layout="vertical">
        <CartesianGrid strokeDasharray="3 3" stroke={CHART.gridStroke} horizontal={false} />
        <XAxis
          type="number"
          tickFormatter={(v: number) => `$${v.toFixed(0)}`}
          tick={AXIS_TICK}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          dataKey="symbol"
          type="category"
          tick={{ fill: CHART.axisText, fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={90}
        />
        <Tooltip
          contentStyle={CHART_TOOLTIP}
          formatter={(value) => [formatUsd(Number(value)), "P&L"]}
        />
        <Bar dataKey="pnl" radius={[0, 4, 4, 0]} barSize={20}>
          {data.map((entry, i) => (
            <Cell key={i} fill={pnlFill(entry.pnl)} fillOpacity={0.8} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
