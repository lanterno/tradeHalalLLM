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
  /**
   * "equity" (default) plots the real account value (`ending_equity`) over
   * time — a mark-to-market equity curve. "cumulative" plots the running sum
   * of realized P&L (the old behavior, used by the Analytics "Cumulative P&L"
   * panel).
   */
  mode?: "equity" | "cumulative";
}

function compactUsd(v: number): string {
  return Math.abs(v) >= 1000 ? `$${(v / 1000).toFixed(1)}k` : `$${v.toFixed(0)}`;
}

export function EquityCurve({ data, mode = "equity" }: EquityCurveProps) {
  const sorted = data.slice().sort((a, b) => a.date.localeCompare(b.date));

  let chartData: { date: string; value: number }[];
  if (mode === "equity") {
    // Only days the ledger actually closed an equity snapshot; null/0 rows
    // (e.g. non-trading days before the bot started) would flatten the curve.
    chartData = sorted
      .filter((d) => d.ending_equity != null && d.ending_equity > 0)
      .map((d) => ({ date: d.date, value: d.ending_equity }));
  } else {
    let cum = 0;
    chartData = sorted.map((d) => {
      cum += d.realized_pnl;
      return { date: d.date, value: cum };
    });
  }

  const label = mode === "equity" ? "Account equity" : "Cumulative P&L";

  if (!chartData.length) {
    return (
      <p className="py-12 text-center text-sm text-muted">
        {mode === "equity" ? "No equity history yet." : "No P&L data yet."}
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
          // Equity values are large + tightly clustered — auto domain keeps
          // the day-to-day variation visible instead of a flat line near the top.
          domain={mode === "equity" ? ["auto", "auto"] : undefined}
          tickFormatter={compactUsd}
          tick={AXIS_TICK}
          axisLine={false}
          tickLine={false}
          width={52}
        />
        <Tooltip
          contentStyle={CHART_TOOLTIP}
          formatter={(value) => [formatUsd(Number(value)), label]}
          labelFormatter={(l) => formatDate(String(l))}
        />
        <Area
          type="monotone"
          dataKey="value"
          stroke={CHART.accent}
          fill="url(#pnlGrad)"
          strokeWidth={2}
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
