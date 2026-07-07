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
import { entityOf, formatUsd } from "../lib/utils";
import { AXIS_TICK, CHART, CHART_TOOLTIP, pnlFill } from "../lib/charts";

interface PairBreakdownProps {
  trades: Trade[];
}

export function PairBreakdown({ trades }: PairBreakdownProps) {
  const pnlByEntity: Record<string, number> = {};
  for (const t of trades) {
    // Stocks carry filled_price as the entry basis; crypto carry entry_price.
    const entry = t.entry_price ?? t.filled_price ?? t.price;
    if (t.exit_price != null && entry != null) {
      const key = entityOf(t);
      const pnl = (t.exit_price - entry) * t.quantity;
      pnlByEntity[key] = (pnlByEntity[key] ?? 0) + pnl;
    }
  }

  const data = Object.entries(pnlByEntity)
    .map(([pair, pnl]) => ({ pair, pnl }))
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
          dataKey="pair"
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
