import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { CoreStatus } from "../../api/types";
import { AXIS_TICK, CHART, CHART_TOOLTIP } from "../../lib/charts";
import { formatDate, formatDay } from "../../lib/utils";

import { Panel } from "../Panel";
import { SignedPct } from "./shared";

const LINES = [
  { key: "account", name: "Core", color: CHART.accent, dash: undefined },
  { key: "book", name: "Forward book", color: "#60a5fa", dash: "4 3" },
  { key: "spus", name: "SPUS", color: "#facc15", dash: undefined },
  { key: "hlal", name: "HLAL", color: "#c084fc", dash: undefined },
  { key: "spy", name: "SPY", color: "#6b7280", dash: "2 3" },
] as const;

export function PerformancePanel({ data }: { data: CoreStatus }) {
  const s = data.since;
  const rows = [
    {
      label: "Today",
      core: data.change_pct,
      book: null as number | null,
      spus: data.today_vs.find((b) => b.symbol === "SPUS")?.change_pct ?? null,
      hlal: data.today_vs.find((b) => b.symbol === "HLAL")?.change_pct ?? null,
    },
    ...(s
      ? [
          {
            label: `Since ${formatDay(s.since)}`,
            core: s.change_pct,
            book: s.book_pct,
            spus: s.spus_pct ?? null,
            hlal: s.hlal_pct ?? null,
          },
        ]
      : []),
  ];
  return (
    <Panel title="Performance" right="rebased to 100 on the account's first day">
      {data.series.length < 2 ? (
        <p className="py-10 text-center text-sm text-muted">
          A chart needs two closes of the account; the first came {s ? formatDay(s.since) : "—"}.
        </p>
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data.series}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART.gridStroke} />
            <XAxis
              dataKey="date"
              tickFormatter={formatDate}
              tick={AXIS_TICK}
              axisLine={{ stroke: CHART.gridStroke }}
              tickLine={false}
            />
            <YAxis domain={["auto", "auto"]} tick={AXIS_TICK} axisLine={false} tickLine={false} width={44} />
            <ReferenceLine y={100} stroke={CHART.gridStroke} />
            <Tooltip contentStyle={CHART_TOOLTIP} labelFormatter={(v) => formatDate(String(v))} />
            {LINES.map((l) => (
              <Line
                key={l.key}
                type="monotone"
                dataKey={l.key}
                name={l.name}
                stroke={l.color}
                strokeDasharray={l.dash}
                strokeWidth={l.key === "account" ? 2.2 : 1.4}
                dot={false}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted">
        {LINES.map((l) => (
          <span key={l.key} className="flex items-center gap-1.5">
            <i className="inline-block h-0.5 w-4" style={{ background: l.color }} />
            {l.name}
          </span>
        ))}
      </div>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-sm tabular-nums">
          <thead className="text-left text-[11px] uppercase tracking-wider text-muted">
            <tr>
              <th className="py-1.5">Period</th>
              <th className="py-1.5 text-right">Core</th>
              <th className="py-1.5 text-right">Book</th>
              <th className="py-1.5 text-right">SPUS</th>
              <th className="py-1.5 text-right">HLAL</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {rows.map((r) => (
              <tr key={r.label}>
                <td className="py-1.5 text-white">{r.label}</td>
                <td className="py-1.5 text-right font-semibold">
                  <SignedPct v={r.core} />
                </td>
                <td className="py-1.5 text-right">
                  {r.book == null ? <span className="text-muted">—</span> : <SignedPct v={r.book} />}
                </td>
                <td className="py-1.5 text-right">
                  <SignedPct v={r.spus} />
                </td>
                <td className="py-1.5 text-right">
                  <SignedPct v={r.hlal} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-2 text-[11px] text-muted">
          Points, not a trend: a line chart starts to mean something after a few weeks. The book
          is what the strategy would hold with no frictions; the gate measures the account
          against it.
        </p>
      </div>
    </Panel>
  );
}
