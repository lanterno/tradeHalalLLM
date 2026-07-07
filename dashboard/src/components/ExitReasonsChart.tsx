import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { CHART, CHART_COLORS, CHART_TOOLTIP } from "../lib/charts";

interface ExitReasonsChartProps {
  data: Record<string, number>;
}

export function ExitReasonsChart({ data }: ExitReasonsChartProps) {
  const entries = Object.entries(data).map(([name, value]) => ({
    name,
    value,
  }));

  if (!entries.length) {
    return (
      <p className="py-12 text-center text-sm text-muted">No data.</p>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <PieChart>
        <Pie
          data={entries}
          cx="50%"
          cy="50%"
          innerRadius={60}
          outerRadius={100}
          paddingAngle={3}
          dataKey="value"
          stroke="none"
        >
          {entries.map((_, i) => (
            <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
          ))}
        </Pie>
        <Tooltip contentStyle={CHART_TOOLTIP} />
        <Legend
          wrapperStyle={{ fontSize: 11, color: CHART.muted }}
          iconType="circle"
          iconSize={8}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}
