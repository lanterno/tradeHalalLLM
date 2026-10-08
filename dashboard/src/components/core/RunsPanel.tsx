import { useState } from "react";
import type { CoreExecution, CoreRun, CoreStatus } from "../../api/types";
import { cn, formatQty, formatUsd, formatDay } from "../../lib/utils";
import { bps, costTone } from "./format";
import { Panel } from "../Panel";

function kind(r: CoreRun): { label: string; style: string } {
  if (r.halted) return { label: "Halted", style: "bg-loss/15 text-loss" };
  if (!r.executed) return { label: "Plan only", style: "bg-border text-muted" };
  if (r.monthly) return { label: "Monthly", style: "bg-accent/15 text-accent" };
  return { label: "Daily", style: "bg-border text-white" };
}

function Orders({ run, budget }: { run: CoreRun; budget: number }) {
  if (run.order_rows.length === 0) return null;
  return (
    <tr>
      <td colSpan={7} className="bg-bg/60 px-2 pb-3">
        <div className="overflow-x-auto">
          <table className="w-full text-xs tabular-nums">
            <thead className="text-left text-[10px] uppercase tracking-wider text-muted">
              <tr>
                <th className="py-1">Symbol</th>
                <th className="py-1">Side</th>
                <th className="py-1 text-right">Qty</th>
                <th className="py-1 text-right">Filled value</th>
                <th className="py-1 text-right">Arrival</th>
                <th className="py-1 text-right">Fill</th>
                <th className="py-1 text-right">vs arrival</th>
                <th className="py-1 text-right">Close</th>
                <th className="py-1 text-right">vs close</th>
                <th className="py-1">Reason</th>
                <th className="py-1">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {run.order_rows.map((o, i) => (
                <tr key={`${o.symbol}-${i}`}>
                  <td className="py-1 font-semibold text-white">{o.symbol}</td>
                  <td className={cn("py-1", o.side === "buy" ? "text-accent" : "text-loss")}>{o.side}</td>
                  <td className="py-1 text-right">{o.qty == null ? "—" : formatQty(o.qty, 4)}</td>
                  <td className="py-1 text-right">
                    {o.fill_price && o.filled_qty ? formatUsd(o.fill_price * o.filled_qty) : formatUsd(o.notional)}
                  </td>
                  <td className="py-1 text-right">{formatUsd(o.price)}</td>
                  <td className="py-1 text-right">{o.fill_price == null ? "—" : formatUsd(o.fill_price)}</td>
                  <td className={cn("py-1 text-right", costTone(o.vs_arrival_bps, budget))}>
                    {bps(o.vs_arrival_bps)}
                  </td>
                  <td className="py-1 text-right">{o.close == null ? "—" : formatUsd(o.close)}</td>
                  <td className={cn("py-1 text-right", costTone(o.vs_close_bps, budget))}>
                    {bps(o.vs_close_bps)}
                  </td>
                  <td className="whitespace-normal py-1 text-muted">{o.reason}</td>
                  <td className={cn("py-1", o.fill_status === "filled" ? "text-accent" : "text-warning")}>
                    {o.fill_status ?? o.status}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </td>
    </tr>
  );
}

export function RunsPanel({ data }: { data: CoreStatus }) {
  const [open, setOpen] = useState<string | null>(
    data.runs.find((r) => r.order_rows.length > 0)?.run_on ?? null,
  );
  const budget = data.execution?.book_cost_bps ?? 5;
  return (
    <Panel title="Runs and their orders" right="one row per run · 15:40 ET each trading day">
      {data.runs.length === 0 ? (
        <p className="py-6 text-center text-sm text-muted">
          No run recorded yet: the first is {formatDay(data.next_check)} at 15:40 ET.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm tabular-nums">
            <thead className="text-left text-[11px] uppercase tracking-wider text-muted">
              <tr>
                <th className="py-1.5">Run</th>
                <th className="py-1.5">Kind</th>
                <th className="py-1.5">Orders</th>
                <th className="py-1.5 text-right">Notional</th>
                <th className="py-1.5 text-right">vs arrival</th>
                <th className="py-1.5 text-right">vs close</th>
                <th className="py-1.5 text-right">Filled</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-t border-border text-muted">
                <td className="py-2">{formatDay(data.next_check)}</td>
                <td className="py-2">
                  <span className="rounded bg-border px-1.5 py-0.5 text-xs">
                    {data.monthly_due ? "Monthly" : "Daily"}
                  </span>
                </td>
                <td className="whitespace-normal py-2" colSpan={5}>
                  scheduled
                  {data.to_sell.length > 0 &&
                    ` · sells ${data.to_sell.map((s) => s.symbol).join(", ")} (fails the screen)`}
                </td>
              </tr>
              {data.runs.map((r) => {
                const k = kind(r);
                const isOpen = open === r.run_on;
                const sells = r.order_rows.filter((o) => o.side === "sell").length;
                const buys = r.order_rows.length - sells;
                return [
                  <tr
                    key={r.run_on}
                    className={cn(
                      "border-t border-border",
                      r.order_rows.length > 0 && "cursor-pointer hover:bg-surface-hover",
                    )}
                    onClick={() => r.order_rows.length > 0 && setOpen(isOpen ? null : r.run_on)}
                  >
                    <td className="py-2 text-white">
                      {r.order_rows.length > 0 && <span className="mr-1 text-muted">{isOpen ? "▾" : "▸"}</span>}
                      {formatDay(r.run_on)}
                    </td>
                    <td className="py-2">
                      <span className={cn("rounded px-1.5 py-0.5 text-xs", k.style)}>{k.label}</span>
                    </td>
                    <td className="whitespace-normal py-2 text-xs">
                      {r.halted
                        ? r.halted
                        : r.order_rows.length
                          ? `${buys ? `${buys} buys` : ""}${buys && sells ? " · " : ""}${sells ? `${sells} sells` : ""}`
                          : r.orders
                            ? `${r.orders} planned, none sent`
                            : "nothing to trade"}
                    </td>
                    <td className="py-2 text-right">{formatUsd(r.notional)}</td>
                    <td className={cn("py-2 text-right", costTone(r.vs_arrival_bps, budget))}>
                      {bps(r.vs_arrival_bps)}
                    </td>
                    <td className={cn("py-2 text-right", costTone(r.vs_close_bps, budget))}>
                      {bps(r.vs_close_bps)}
                    </td>
                    <td className="py-2 text-right">
                      {r.order_rows.length ? `${r.filled} / ${r.order_rows.length}` : "—"}
                    </td>
                  </tr>,
                  isOpen ? <Orders key={`${r.run_on}-orders`} run={r} budget={budget} /> : null,
                ];
              })}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

export function ExecutionPanel({ e }: { e: CoreExecution }) {
  const peak = Math.max(...e.histogram.map((b) => b.value), 1);
  return (
    <Panel title="Execution quality" right={`since ${formatDay(e.start)}`}>
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-lg border border-border p-3">
          <p className="text-[11px] uppercase tracking-wider text-muted">Filled</p>
          <p className="text-lg font-bold text-white">
            {e.filled} / {e.orders}
          </p>
          <p className="text-[11px] text-muted">
            {e.partial} partial · {e.unfilled} unfilled
          </p>
        </div>
        <div className="rounded-lg border border-border p-3">
          <p className="text-[11px] uppercase tracking-wider text-muted">Traded</p>
          <p className="text-lg font-bold text-white">{formatUsd(e.filled_notional)}</p>
        </div>
        <div className="rounded-lg border border-border p-3">
          <p className="text-[11px] uppercase tracking-wider text-muted">vs arrival</p>
          <p className={cn("text-lg font-bold", costTone(e.vs_arrival_bps, e.book_cost_bps))}>
            {bps(e.vs_arrival_bps)}
          </p>
          <p className="text-[11px] text-muted">vs the price the plan used</p>
        </div>
        <div className="rounded-lg border border-border p-3">
          <p className="text-[11px] uppercase tracking-wider text-muted">vs close</p>
          <p className={cn("text-lg font-bold", costTone(e.vs_close_bps, e.book_cost_bps))}>
            {bps(e.vs_close_bps)}
          </p>
          <p className="text-[11px] text-muted">the book assumes +{e.book_cost_bps} bps</p>
        </div>
      </div>
      <p className="mt-4 text-[11px] uppercase tracking-wider text-muted">Fills vs the close, by value</p>
      <div className="mt-2 flex h-20 items-end gap-1">
        {e.histogram.map((b) => (
          <div key={b.from_bps} className="flex flex-1 flex-col items-center gap-1">
            <div
              className={cn("w-full rounded-t", b.from_bps >= e.book_cost_bps ? "bg-warning/70" : "bg-accent/70")}
              style={{ height: `${(b.value / peak) * 100}%`, minHeight: b.value ? 2 : 0 }}
              title={`${b.from_bps} to ${b.to_bps} bps: ${formatUsd(b.value)}`}
            />
          </div>
        ))}
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-muted">
        <span>≤ {e.histogram[0]?.from_bps + 10} bps</span>
        <span>0</span>
        <span>≥ {e.histogram[e.histogram.length - 1]?.from_bps} bps</span>
      </div>
      {e.worst.length > 0 && (
        <>
          <p className="mt-4 text-[11px] uppercase tracking-wider text-muted">Worst vs arrival</p>
          <ul className="mt-1 divide-y divide-border text-xs tabular-nums">
            {e.worst.map((w) => (
              <li key={`${w.symbol}-${w.side}`} className="flex justify-between py-1">
                <span className="text-white">
                  {w.symbol} <span className="text-muted">{w.side}</span>
                </span>
                <span className={costTone(w.vs_arrival_bps, e.book_cost_bps)}>{bps(w.vs_arrival_bps)}</span>
              </li>
            ))}
          </ul>
        </>
      )}
      <p className="mt-3 text-[11px] leading-relaxed text-muted">
        A positive figure is a cost: a buy filled above the reference, or a sell below it.
        Weighted by filled value.
      </p>
    </Panel>
  );
}
