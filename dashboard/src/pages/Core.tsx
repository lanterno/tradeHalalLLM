import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { CheckCircle2, Circle } from "lucide-react";
import { useCore } from "../hooks/useCore";
import { StatCard } from "../components/StatCard";
import { ErrorState } from "../components/ErrorState";
import { cn, formatDate, formatPct, formatQty, formatTime, formatUsd } from "../lib/utils";
import { AXIS_TICK, CHART, CHART_TOOLTIP } from "../lib/charts";
import type { CoreStatus } from "../api/types";

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-muted">{title}</h2>
      {children}
    </section>
  );
}

function Check({ ok, label, detail }: { ok: boolean; label: string; detail: string }) {
  const Icon = ok ? CheckCircle2 : Circle;
  return (
    <li className="flex items-start gap-3 py-2">
      <Icon className={cn("mt-0.5 h-4 w-4 shrink-0", ok ? "text-accent" : "text-muted")} />
      <div>
        <p className="text-sm text-white">{label}</p>
        <p className="text-xs text-muted">{detail}</p>
      </div>
    </li>
  );
}

function Gate({ r }: { r: CoreStatus["readiness"] }) {
  const te = r.tracking_error;
  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <p className={cn("text-sm font-semibold", r.ready ? "text-accent" : "text-warning")}>
        {r.ready ? "Ready for live keys" : "Not yet ready for real money"}
      </p>
      <ul className="mt-2 divide-y divide-border">
        <Check
          ok={r.days >= r.min_days}
          label="Long enough on paper"
          detail={`${r.days} of ${r.min_days} trading days`}
        />
        <Check
          ok={r.monthly_runs >= 1}
          label="A monthly rebalance executed"
          detail={`${r.monthly_runs} so far`}
        />
        <Check
          ok={te !== null && te <= r.max_tracking_error}
          label="Follows its forward book"
          detail={
            te === null
              ? "tracking error not yet measurable"
              : `tracking error ${formatPct(te)} (limit ${formatPct(r.max_tracking_error)})`
          }
        />
        <Check
          ok={r.gap !== null && Math.abs(r.gap) <= r.max_gap}
          label="No cumulative gap to the book"
          detail={
            r.gap === null
              ? "not yet measurable"
              : `${(r.gap * 100).toFixed(2)}% (limit ±${formatPct(r.max_gap)})`
          }
        />
        <Check
          ok={r.refused === 0 && r.unfilled === 0 && r.halted === 0}
          label="Clean"
          detail={`${r.refused} refused, ${r.unfilled} unfilled order(s), ${r.halted} halted run(s) recently`}
        />
      </ul>
    </div>
  );
}

function TrackingChart({ series }: { series: CoreStatus["series"] }) {
  if (series.length < 2) {
    return <p className="py-12 text-center text-sm text-muted">Not enough history yet.</p>;
  }
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={series}>
        <CartesianGrid strokeDasharray="3 3" stroke={CHART.gridStroke} />
        <XAxis
          dataKey="date"
          tickFormatter={formatDate}
          tick={AXIS_TICK}
          axisLine={{ stroke: CHART.gridStroke }}
          tickLine={false}
        />
        <YAxis domain={["auto", "auto"]} tick={AXIS_TICK} axisLine={false} tickLine={false} />
        <Tooltip contentStyle={CHART_TOOLTIP} />
        <Legend />
        <Line type="monotone" dataKey="account" name="Account" stroke={CHART.accent} dot={false} />
        <Line
          type="monotone"
          dataKey="book"
          name="Forward book"
          stroke="#60a5fa"
          dot={false}
          strokeDasharray="4 3"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

const bps = (v: number | null) => (v === null ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(1)} bps`);

function Execution({ e }: { e: NonNullable<CoreStatus["execution"]> }) {
  const closeCls =
    e.vs_close_bps === null
      ? "text-white"
      : e.vs_close_bps <= e.book_cost_bps
        ? "text-accent"
        : "text-warning";
  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <div>
          <p className="text-xs uppercase tracking-wider text-muted">Filled</p>
          <p className="mt-1 text-xl font-bold text-white">
            {e.filled} / {e.orders}
          </p>
          <p className="text-xs text-muted">
            {e.partial} partial · {e.unfilled} unfilled
          </p>
        </div>
        <div>
          <p className="text-xs uppercase tracking-wider text-muted">Traded</p>
          <p className="mt-1 text-xl font-bold text-white">{formatUsd(e.filled_notional)}</p>
          <p className="text-xs text-muted">since {e.start}</p>
        </div>
        <div>
          <p className="text-xs uppercase tracking-wider text-muted">Vs arrival</p>
          <p className="mt-1 text-xl font-bold text-white">{bps(e.vs_arrival_bps)}</p>
          <p className="text-xs text-muted">vs the price the plan used</p>
        </div>
        <div>
          <p className="text-xs uppercase tracking-wider text-muted">Vs close</p>
          <p className={cn("mt-1 text-xl font-bold", closeCls)}>{bps(e.vs_close_bps)}</p>
          <p className="text-xs text-muted">
            {e.vs_close_bps === null
              ? "after the evening run"
              : `the book assumes +${e.book_cost_bps} bps`}
          </p>
        </div>
      </div>
      <p className="mt-3 text-[11px] leading-relaxed text-muted">
        Positive is a cost: a buy filled above the reference price, or a sell below it. Weighted
        by filled value.
      </p>
    </div>
  );
}

export default function Core() {
  const { data, isLoading, isError, error, refetch } = useCore();
  const held = data?.holdings.filter((h) => h.shares) ?? [];

  return (
    <div className="space-y-6 p-4 sm:p-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Core portfolio</h1>
        <p className="mt-1 text-xs text-muted">
          The largest 100 names the strict screen passes, cap-weighted, rebalanced monthly
          within bands, on its own account. A holding that stops passing is sold at the next
          run.
        </p>
      </div>

      {isError ? (
        <ErrorState error={error} onRetry={refetch} />
      ) : isLoading || !data ? (
        <p className="py-8 text-center text-sm text-muted">Loading…</p>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <StatCard
              label="Equity"
              value={
                data.equity !== null
                  ? formatUsd(data.equity)
                  : data.runs[0]?.equity != null
                    ? formatUsd(data.runs[0].equity)
                    : "—"
              }
              sub={
                data.equity_day
                  ? `as of ${data.equity_day}`
                  : data.runs[0]
                    ? `at the ${data.runs[0].run_on} run`
                    : "no snapshot yet"
              }
            />
            <StatCard
              label="Mode"
              value={
                <span className={data.paper ? "text-warning" : "text-accent"}>
                  {data.paper ? "Paper" : "Live"}
                </span>
              }
              sub={data.enabled ? "trading at 15:40 ET" : "disabled"}
            />
            <StatCard label="Holdings" value={held.length} sub={`${data.holdings.length} targeted`} />
            <StatCard
              label="Last run"
              value={data.runs[0] ? formatDate(data.runs[0].run_on) : "—"}
              sub={
                data.runs[0]
                  ? data.runs[0].halted
                    ? `halted: ${data.runs[0].halted}`
                    : `${data.runs[0].orders} order(s)${data.runs[0].monthly ? ", monthly" : ""}${
                        data.runs[0].executed ? "" : " · plan only"
                      }`
                  : "none yet"
              }
            />
          </div>

          <Section title="Live-money gate">
            <Gate r={data.readiness} />
          </Section>

          <Section title="Account vs forward book (rebased to 100)">
            <div className="rounded-xl border border-border bg-surface p-4">
              <TrackingChart series={data.series} />
            </div>
          </Section>

          <Section title="Holdings vs targets">
            {data.holdings.length === 0 ? (
              <p className="text-sm text-muted">No holdings or targets yet.</p>
            ) : (
              <div className="overflow-x-auto rounded-xl border border-border bg-surface">
                <table className="w-full text-sm">
                  <thead className="text-left text-xs uppercase tracking-wider text-muted">
                    <tr>
                      <th className="px-4 py-2">Symbol</th>
                      <th className="px-4 py-2 text-right">Value</th>
                      <th className="px-4 py-2 text-right">Weight</th>
                      <th className="px-4 py-2 text-right">Target</th>
                      <th className="px-4 py-2 text-right">Drift</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {data.holdings.map((h) => (
                      <tr key={h.symbol}>
                        <td className="px-4 py-2 font-medium text-white">{h.symbol}</td>
                        <td className="px-4 py-2 text-right">
                          {h.value ? formatUsd(h.value) : "—"}
                        </td>
                        <td className="px-4 py-2 text-right">{formatPct(h.weight)}</td>
                        <td className="px-4 py-2 text-right">{formatPct(h.target)}</td>
                        <td
                          className={cn(
                            "px-4 py-2 text-right",
                            h.drift !== null &&
                              h.target &&
                              Math.abs(h.drift) > Math.max(0.25 * h.target, 0.002)
                              ? "text-warning"
                              : "text-muted",
                          )}
                        >
                          {h.drift === null ? "—" : `${(h.drift * 100).toFixed(2)}%`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="px-4 py-2 text-[11px] text-muted">
                  Values at the last close. Drift beyond the band (a quarter of the target, at
                  least 0.2 points) is traded at the next monthly rebalance.
                </p>
              </div>
            )}
          </Section>

          {data.execution && (
            <Section title="Execution quality (30 days)">
              <Execution e={data.execution} />
            </Section>
          )}

          <Section title="Recent orders">
            {data.orders.length === 0 ? (
              <p className="text-sm text-muted">No orders yet.</p>
            ) : (
              <div className="overflow-x-auto rounded-xl border border-border bg-surface">
                <table className="w-full text-sm">
                  <thead className="text-left text-xs uppercase tracking-wider text-muted">
                    <tr>
                      <th className="px-4 py-2">When</th>
                      <th className="px-4 py-2">Order</th>
                      <th className="px-4 py-2 text-right">Notional</th>
                      <th className="px-4 py-2 text-right">Fill</th>
                      <th className="px-4 py-2 text-right">Slippage</th>
                      <th className="px-4 py-2">Reason</th>
                      <th className="px-4 py-2">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {data.orders.map((o, i) => (
                      <tr key={`${o.at}-${o.symbol}-${i}`}>
                        <td className="whitespace-nowrap px-4 py-2 text-muted">{formatTime(o.at)}</td>
                        <td className="whitespace-nowrap px-4 py-2 text-white">
                          <span className={o.side === "buy" ? "text-accent" : "text-loss"}>
                            {o.side}
                          </span>{" "}
                          {o.qty === null ? "" : formatQty(o.qty, 4)} {o.symbol}
                        </td>
                        <td className="px-4 py-2 text-right">
                          {o.notional === null ? "—" : formatUsd(o.notional)}
                        </td>
                        <td className="px-4 py-2 text-right">
                          {o.fill_price === null ? (
                            <span className="text-muted">{o.fill_status ?? "—"}</span>
                          ) : (
                            formatUsd(o.fill_price)
                          )}
                        </td>
                        <td
                          className={cn(
                            "px-4 py-2 text-right",
                            (o.vs_arrival_bps ?? 0) > 25 ? "text-warning" : "text-muted",
                          )}
                        >
                          {bps(o.vs_arrival_bps)}
                        </td>
                        <td className="px-4 py-2 text-muted">{o.reason}</td>
                        <td
                          className={cn(
                            "px-4 py-2",
                            o.status === "submitted" ? "text-muted" : "text-warning",
                          )}
                        >
                          {o.fill_status ?? o.status}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Section>
        </>
      )}
    </div>
  );
}
