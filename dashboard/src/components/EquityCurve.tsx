import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import type { DailyPnl } from "../api/types";
import { formatDate, formatUsd } from "../lib/utils";

interface EquityCurveProps {
  data: DailyPnl[];
}

export function EquityCurve({ data }: EquityCurveProps) {
  const chartData = data
    .slice()
    .sort((a, b) => a.date.localeCompare(b.date))
    .reduce<{ date: string; pnl: number; cumulative: number }[]>(
      (acc, d) => {
        const prev = acc.length ? acc[acc.length - 1].cumulative : 0;
        acc.push({
          date: d.date,
          pnl: d.realized_pnl,
          cumulative: prev + d.realized_pnl,
        });
        return acc;
      },
      [],
    );

  if (!chartData.length) {
    return (
      <p className="py-12 text-center text-sm text-muted">
        No P&L data yet.
      </p>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <AreaChart data={chartData}>
        <defs>
          <linearGradient id="pnlGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#4ade80" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#4ade80" stopOpacity={0} />
          </linearGradient>
        </defs>
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
          formatter={(value, name) => [
            formatUsd(Number(value)),
            name === "cumulative" ? "Cumulative" : "Daily P&L",
          ]}
          labelFormatter={(label) => formatDate(String(label))}
        />
        <Area
          type="monotone"
          dataKey="cumulative"
          stroke="#4ade80"
          fill="url(#pnlGrad)"
          strokeWidth={2}
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
