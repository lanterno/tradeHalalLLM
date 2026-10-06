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
import { AXIS_TICK, CHART, CHART_TOOLTIP, pnlFill } from "../lib/charts";

interface PnlBarChartProps {
  data: DailyPnl[];
}

export function PnlBarChart({ data }: PnlBarChartProps) {
  // Closed days only: an open day has no ending equity, hence no change yet.
  const sorted = data
    .filter((d) => d.equity_change != null)
    .sort((a, b) => a.date.localeCompare(b.date));

  if (!sorted.length) {
    return (
      <p className="py-12 text-center text-sm text-muted">No closed days in this window.</p>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={sorted}>
        <CartesianGrid strokeDasharray="3 3" stroke={CHART.gridStroke} />
        <XAxis
          dataKey="date"
          tickFormatter={formatDate}
          tick={AXIS_TICK}
          axisLine={{ stroke: CHART.gridStroke }}
          tickLine={false}
        />
        <YAxis
          tickFormatter={(v: number) => `$${v.toFixed(0)}`}
          tick={AXIS_TICK}
          axisLine={false}
          tickLine={false}
          width={60}
        />
        <Tooltip
          contentStyle={CHART_TOOLTIP}
          formatter={(value) => [formatUsd(Number(value)), "Equity change"]}
          labelFormatter={(label) => formatDate(String(label))}
        />
        <Bar dataKey="equity_change" radius={[3, 3, 0, 0]}>
          {sorted.map((entry, i) => (
            <Cell key={i} fill={pnlFill(entry.equity_change ?? 0)} fillOpacity={0.8} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
