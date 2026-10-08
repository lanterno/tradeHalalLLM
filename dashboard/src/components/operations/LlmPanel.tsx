import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { OperationsStatus, OpsPool } from "../../api/types";
import { useLlmMetrics } from "../../hooks/useMetrics";
import { AXIS_TICK, CHART, CHART_TOOLTIP } from "../../lib/charts";
import { cn, formatDate } from "../../lib/utils";
import { Panel } from "../Panel";
import { usd } from "./format";

const CONSUMERS: Record<string, { label: string; what: string; color: string }> = {
  stock: { label: "Stock bot", what: "strategy, news classifier, stock of the day", color: "#60a5fa" },
  shadow: { label: "Shadow engine", what: "belief theses for the Belief Board", color: "#fb923c" },
  research: { label: "Research", what: "evening run: headline scoring", color: "#c084fc" },
};
const consumer = (c: string) => CONSUMERS[c] ?? { label: c, what: "", color: "#6b7280" };

function Bar1({ value, cap }: { value: number; cap: number }) {
  const share = cap ? value / cap : 0;
  return (
    <div className="h-1.5 rounded bg-border">
      <div
        className={cn("h-1.5 rounded", share >= 0.8 ? "bg-warning" : "bg-accent")}
        style={{ width: `${Math.min(share, 1) * 100}%` }}
      />
    </div>
  );
}

function PoolCard({ p }: { p: OpsPool }) {
  return (
    <div className="rounded-lg border border-border p-3 text-xs">
      <div className="flex items-baseline justify-between">
        <p className="font-semibold capitalize text-white">{p.pool} pool</p>
        <span className="text-muted">{p.members.map((m) => consumer(m).label).join(" + ")}</span>
      </div>
      <div className="mt-2 flex justify-between text-muted">
        <span>today</span>
        <span>
          <span className="font-semibold text-white">{usd(p.today)}</span>
          {p.daily_cap ? ` / ${usd(p.daily_cap)}` : " · no daily cap"}
        </span>
      </div>
      {p.daily_cap ? <Bar1 value={p.today} cap={p.daily_cap} /> : null}
      <div className="mt-2 flex justify-between text-muted">
        <span>this month</span>
        <span>
          <span className="font-semibold text-white">{usd(p.month)}</span>
          {p.monthly_cap ? ` / ${usd(p.monthly_cap, 0)}` : ""}
        </span>
      </div>
      {p.monthly_cap ? <Bar1 value={p.month} cap={p.monthly_cap} /> : null}
      <p className="mt-1 text-muted">on pace for ~{usd(p.pace)} by the month's end</p>
    </div>
  );
}

export function LlmPanel({ data }: { data: OperationsStatus }) {
  const llm = data.llm;
  const week = useLlmMetrics(604_800);
  const names = [...new Set(llm.days.flatMap((d) => Object.keys(d).filter((k) => k !== "day")))];
  const total = (k: "today" | "yesterday" | "month") => llm.consumers.reduce((s, c) => s + c[k], 0);
  return (
    <Panel title="LLM spend" right="from the spend meter the caps enforce · UTC days">
      <div className="grid gap-3 sm:grid-cols-2">
        {llm.pools.map((p) => (
          <PoolCard key={p.pool} p={p} />
        ))}
      </div>

      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-sm tabular-nums">
          <thead className="text-left text-[11px] uppercase tracking-wider text-muted">
            <tr>
              <th className="py-1.5 pr-3">Consumer</th>
              <th className="py-1.5 pr-3 text-right">Today</th>
              <th className="py-1.5 pr-3 text-right">Yesterday</th>
              <th className="py-1.5 text-right">Month</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {llm.consumers.map((c) => (
              <tr key={c.consumer} className="align-top">
                <td className="whitespace-normal py-2 pr-3">
                  <p className="flex items-center gap-2 font-semibold text-white">
                    <i
                      className="inline-block h-2 w-2 rounded-sm"
                      style={{ background: consumer(c.consumer).color }}
                    />
                    {consumer(c.consumer).label}
                  </p>
                  <p className="text-[11px] text-muted">{consumer(c.consumer).what}</p>
                </td>
                <td className="py-2 pr-3 text-right">
                  {usd(c.today)}
                  <p className="text-[11px] text-muted">{c.calls_today.toLocaleString()} calls</p>
                </td>
                <td className="py-2 pr-3 text-right">{usd(c.yesterday)}</td>
                <td className="py-2 text-right">
                  {usd(c.month)}
                  <p className="text-[11px] text-muted">{c.calls_month.toLocaleString()} calls</p>
                </td>
              </tr>
            ))}
            <tr className="font-semibold text-white">
              <td className="py-2 pr-3">All</td>
              <td className="py-2 pr-3 text-right">{usd(total("today"))}</td>
              <td className="py-2 pr-3 text-right">{usd(total("yesterday"))}</td>
              <td className="py-2 text-right">{usd(total("month"))}</td>
            </tr>
          </tbody>
        </table>
        {llm.consumers.length === 0 && (
          <p className="py-3 text-center text-xs text-muted">No LLM call metered this month.</p>
        )}
      </div>

      <p className="mt-4 text-[11px] uppercase tracking-wider text-muted">
        Per day, last {llm.days.length} days
      </p>
      <ResponsiveContainer width="100%" height={150}>
        <BarChart data={llm.days} margin={{ top: 8, right: 4, left: -12, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART.gridStroke} vertical={false} />
          <XAxis
            dataKey="day"
            tickFormatter={formatDate}
            tick={AXIS_TICK}
            axisLine={{ stroke: CHART.gridStroke }}
            tickLine={false}
            interval="preserveStartEnd"
          />
          <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} tickFormatter={(v) => `$${v}`} />
          <Tooltip
            contentStyle={CHART_TOOLTIP}
            labelFormatter={(v) => formatDate(String(v))}
            formatter={(v, name) => [usd(Number(v), 3), consumer(String(name)).label]}
          />
          {names.map((n) => (
            <Bar key={n} dataKey={n} stackId="spend" fill={consumer(n).color} />
          ))}
        </BarChart>
      </ResponsiveContainer>

      <p className="mt-3 text-[11px] leading-relaxed text-muted">
        Tokens and latency are not in the meter. From the bot's log, last 7 days:{" "}
        {week.data ? (
          <span className="text-white">
            {week.data.calls.toLocaleString()} calls, {(week.data.total_tokens / 1000).toFixed(0)}k
            tokens, p50 {((week.data.p50_ms ?? 0) / 1000).toFixed(2)} s, p95{" "}
            {((week.data.p95_ms ?? 0) / 1000).toFixed(2)} s
          </span>
        ) : (
          "—"
        )}
        . The meter, which the caps enforce, is the bill.
      </p>
    </Panel>
  );
}
