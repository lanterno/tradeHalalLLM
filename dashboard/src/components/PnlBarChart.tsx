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
import type { DailyPnl } from "../api/types";
import { formatDate, formatUsd } from "../lib/utils";

interface PnlBarChartProps {
  data: DailyPnl[];
}

export function PnlBarChart({ data }: PnlBarChartProps) {
  const sorted = data.slice().sort((a, b) => a.date.localeCompare(b.date));

  if (!sorted.length) {
    return (
      <p className="py-12 text-center text-sm text-muted">No P&L data yet.</p>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={sorted}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1a1a2e" />
        <XAxis
          dataKey="date"
          tickFormatter={formatDate}
          tick={{ fill: "#6b7280", fontSize: 11 }}
          axisLine={{ stroke: "#1a1a2e" }}
          tickLine={false}
        />
        <YAxis
          tickFormatter={(v: number) => `$${v.toFixed(0)}`}
          tick={{ fill: "#6b7280", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={60}
        />
        <Tooltip
          contentStyle={{
            background: "#111118",
            border: "1px solid #1a1a2e",
            borderRadius: 8,
            fontSize: 12,
          }}
          formatter={(value) => [formatUsd(Number(value)), "Daily P&L"]}
          labelFormatter={(label) => formatDate(String(label))}
        />
        <Bar dataKey="realized_pnl" radius={[3, 3, 0, 0]}>
          {sorted.map((entry, i) => (
            <Cell
              key={i}
              fill={entry.realized_pnl >= 0 ? "#4ade80" : "#f87171"}
              fillOpacity={0.8}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
