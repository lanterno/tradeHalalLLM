import type { OperationsStatus, OpsProcess } from "../../api/types";
import { useCycleMetrics } from "../../hooks/useMetrics";
import { cn, formatDay, formatTime } from "../../lib/utils";
import { Panel } from "../Panel";
import { ago } from "./format";
import { Dot, StatusBadge } from "./StatusBadge";

/** What a process's beat carries, in words. */
function detail(p: OpsProcess): string | null {
  const d = p.detail ?? {};
  if (p.component === "stock.monitor" && "market_open" in d) {
    return d.market_open ? "market open: guarding positions" : "market closed: nothing to guard";
  }
  if (p.component === "market.snapshot" && Array.isArray(d.accounts)) {
    return `accounts: ${(d.accounts as string[]).join(", ")}`;
  }
  if (p.component === "web.watchdog") {
    const alerting = (d.alerting as string[] | undefined) ?? [];
    const suspect = (d.suspect as string[] | undefined) ?? [];
    if (alerting.length) return `alerting: ${alerting.join(", ")}`;
    if (suspect.length) return `suspect: ${suspect.join(", ")}`;
    return "nothing suspect, nothing alerting";
  }
  if (p.component === "stock.process" && "core" in d) {
    return d.core ? "runs the core too" : "the core is off: no keys";
  }
  return null;
}

const ms = (v: number | null | undefined) =>
  v == null ? "—" : v < 1000 ? `${v.toFixed(0)} ms` : `${(v / 1000).toFixed(1)} s`;

export function ProcessesPanel({ data }: { data: OperationsStatus }) {
  const cycles = useCycleMetrics(86_400);
  return (
    <Panel title="Processes" right="judged by heartbeat age">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-[11px] uppercase tracking-wider text-muted">
            <tr>
              <th className="py-1.5 pr-3">Component</th>
              <th className="py-1.5 pr-3">Status</th>
              <th className="py-1.5 pr-3">Last beat</th>
              <th className="hidden py-1.5 sm:table-cell">Cadence · detail</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {data.processes.map((p) => {
              const idle = !p.due && p.status === "ok";
              return (
                <tr key={p.component} className="align-top">
                  <td className="whitespace-normal py-2 pr-3">
                    <p className="font-semibold text-white">{p.label}</p>
                    <p className="font-mono text-[10px] text-muted">{p.component}</p>
                  </td>
                  <td className="py-2 pr-3">
                    <StatusBadge status={idle ? "idle" : p.status} title={p.reason} />
                  </td>
                  <td className="py-2 pr-3 tabular-nums">{ago(p.last, data.now)}</td>
                  <td className="hidden whitespace-normal py-2 text-xs text-muted sm:table-cell">
                    {p.cadence}
                    {detail(p) && <p>{detail(p)}</p>}
                    {p.reason && p.status !== "ok" && <p className="text-loss">{p.reason}</p>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-[11px] text-muted">
        Trading cycles, last 24 h (from the bot's log):{" "}
        {cycles.data ? (
          <span className="text-white">
            {cycles.data.count} completed · p50 {ms(cycles.data.p50_ms)} · p95 {ms(cycles.data.p95_ms)}
            {cycles.data.failed ? (
              <span className="text-loss"> · {cycles.data.failed} failed</span>
            ) : null}
            {cycles.data.halted ? (
              <span className="text-warning"> · {cycles.data.halted} halted</span>
            ) : null}
          </span>
        ) : (
          "—"
        )}
      </p>
    </Panel>
  );
}

export function FreshnessPanel({ data }: { data: OperationsStatus }) {
  return (
    <Panel title="Data freshness" right="newest of each input">
      <ul className="divide-y divide-border">
        {data.freshness.map((f) => (
          <li key={f.name} className="flex items-start gap-3 py-2.5">
            <span className="pt-1.5">
              <Dot status={f.status} />
            </span>
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold text-white">{f.name}</p>
              <p className="text-xs text-muted">{f.detail}</p>
            </div>
            <p
              className={cn(
                "shrink-0 text-right text-xs tabular-nums",
                f.status === "stale" ? "text-loss" : "text-white",
              )}
            >
              {f.as_of
                ? f.as_of.length === 10
                  ? formatDay(f.as_of)
                  : ago(f.as_of, data.now).includes("ago")
                    ? ago(f.as_of, data.now)
                    : formatTime(f.as_of)
                : "none"}
            </p>
          </li>
        ))}
      </ul>
    </Panel>
  );
}
