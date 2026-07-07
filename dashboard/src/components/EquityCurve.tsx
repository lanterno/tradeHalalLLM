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
import { AXIS_TICK, CHART, CHART_TOOLTIP } from "../lib/charts";

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
            <stop offset="5%" stopColor={CHART.accent} stopOpacity={0.3} />
            <stop offset="95%" stopColor={CHART.accent} stopOpacity={0} />
          </linearGradient>
        </defs>
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
          formatter={(value, name) => [
            formatUsd(Number(value)),
            name === "cumulative" ? "Cumulative" : "Daily P&L",
          ]}
          labelFormatter={(label) => formatDate(String(label))}
        />
        <Area
          type="monotone"
          dataKey="cumulative"
          stroke={CHART.accent}
          fill="url(#pnlGrad)"
          strokeWidth={2}
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
