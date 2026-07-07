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
        <CartesianGrid strokeDasharray="3 3" stroke="#1a1a2e" horizontal={false} />
        <XAxis
          type="number"
          tickFormatter={(v: number) => `$${v.toFixed(0)}`}
          tick={{ fill: "#6b7280", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          dataKey="pair"
          type="category"
          tick={{ fill: "#e0e0e0", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={90}
        />
        <Tooltip
          contentStyle={{
            background: "#111118",
            border: "1px solid #1a1a2e",
            borderRadius: 8,
            fontSize: 12,
          }}
          formatter={(value) => [formatUsd(Number(value)), "P&L"]}
        />
        <Bar dataKey="pnl" radius={[0, 4, 4, 0]} barSize={20}>
          {data.map((entry, i) => (
            <Cell
              key={i}
              fill={entry.pnl >= 0 ? "#4ade80" : "#f87171"}
              fillOpacity={0.8}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
