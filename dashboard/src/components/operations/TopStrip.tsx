import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import type { OperationsStatus } from "../../api/types";
import { cn, formatBytes, formatDayTime, formatTime } from "../../lib/utils";
import { ago, until, usd } from "./format";

function Tile({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-0 border-border p-4 sm:border-l sm:first:border-l-0">
      <p className="text-[11px] font-semibold uppercase tracking-widest text-muted">{label}</p>
      <div className="mt-1.5 text-xs text-muted">{children}</div>
    </div>
  );
}

const Big = ({ children, className }: { children: ReactNode; className?: string }) => (
  <p className={cn("text-xl font-bold tabular-nums text-white sm:text-2xl", className)}>{children}</p>
);

const VERDICT: Record<string, [string, string]> = {
  healthy: ["Healthy", "text-accent"],
  degraded: ["Needs a look", "text-warning"],
  down: ["Bot down", "text-loss"],
};

export function OpsHeader({ data, paper }: { data: OperationsStatus; paper: boolean }) {
  const m = data.market;
  return (
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-2xl font-bold text-white">Operations</h1>
          <span
            className={cn(
              "rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider",
              paper ? "border-warning/40 bg-warning/10 text-warning" : "border-loss/40 bg-loss/10 text-loss",
            )}
          >
            {paper ? "Paper money" : "Live money"}
          </span>
        </div>
        <p className="mt-1 max-w-3xl text-xs text-muted">
          Is everything running, will tonight's jobs run, what is it costing, and is the data
          current. Every job is judged by the trading calendar, not by how long ago it ran.
        </p>
      </div>
      <div className="text-xs text-muted sm:text-right">
        <p>
          read <span className="text-white">{formatTime(data.now)}</span>
        </p>
        <p>
          {m.open ? (
            <>
              market open · closes <span className="text-white">{formatDayTime(m.closes!)} ET</span>
            </>
          ) : (
            <>
              market closed · next open{" "}
              <span className="text-white">{formatDayTime(m.next_open)} ET</span>
            </>
          )}
          {" · "}refreshes every 30 s
        </p>
      </div>
    </div>
  );
}

export function OpsStrip({ data }: { data: OperationsStatus }) {
  const f = data.fleet;
  const [verdict, tone] = VERDICT[f.verdict];
  const live = data.llm.pools.find((p) => p.pool === "live");
  const nightly = data.backups.nightly;
  const dumpMb = nightly.detail?.dump_mb as number | undefined;
  return (
    <div className="grid grid-cols-2 overflow-hidden rounded-xl border border-border bg-surface sm:grid-cols-3 xl:grid-cols-5">
      <Tile label="Fleet">
        <Big className={tone}>
          <span className="mr-2 inline-block h-2.5 w-2.5 rounded-full bg-current align-middle" />
          {verdict}
        </Big>
        <p>
          {f.beating} process{f.beating === 1 ? "" : "es"} beating · {f.jobs_ran} job
          {f.jobs_ran === 1 ? "" : "s"} ran today
        </p>
        {f.reason && <p className="text-loss">{f.reason}</p>}
        {f.problems.length > 0 && (
          <p className="text-warning">
            {f.problems.length} to look at: {f.problems.slice(0, 2).join("; ")}
            {f.problems.length > 2 ? "…" : ""}
          </p>
        )}
      </Tile>
      <Tile label="Kill-switch">
        <Big className={data.halt.enabled ? "text-loss" : undefined}>
          {data.halt.enabled ? "Halted" : "Off"}
        </Big>
        {data.halt.enabled ? (
          <p>
            {data.halt.reason ?? "no reason given"} · since {formatTime(data.halt.set_at)}
          </p>
        ) : (
          <p>trading allowed{data.halt.set_at ? ` · last change ${formatTime(data.halt.set_at)}` : ""}</p>
        )}
        <Link to="/risk" className="text-accent hover:underline">
          Risk &amp; Halt →
        </Link>
      </Tile>
      <Tile label="LLM spend today">
        <Big>{usd(data.llm.today)}</Big>
        <p>
          {data.llm.calls_today.toLocaleString()} calls · UTC day
          {data.llm.enforced ? " · caps enforced" : ""}
        </p>
        {live?.daily_cap ? (
          <>
            <div className="mt-1.5 h-1.5 rounded bg-border">
              <div
                className="h-1.5 rounded bg-accent"
                style={{ width: `${Math.min(live.today / live.daily_cap, 1) * 100}%` }}
              />
            </div>
            <p className="mt-1">
              live pool {usd(live.today)} of its {usd(live.daily_cap)} daily cap
            </p>
          </>
        ) : null}
        {data.llm.credits && data.llm.credits.available_usd != null && (
          <p className={cn("mt-1", data.llm.credits.low ? "font-semibold text-loss" : undefined)}>
            {usd(data.llm.credits.available_usd)} left on OpenRouter
            {data.llm.credits.days_left != null
              ? ` · ~${data.llm.credits.days_left.toFixed(0)} days at this pace`
              : ""}
          </p>
        )}
      </Tile>
      <Tile label="Last backup">
        {nightly.at ? (
          <>
            <Big className={nightly.status === "ok" ? undefined : "text-loss"}>
              {dumpMb != null ? formatBytes(dumpMb * 1024 * 1024) : "dumped"}
            </Big>
            <p>{ago(nightly.at, data.now)}</p>
          </>
        ) : (
          <Big className="text-loss">None</Big>
        )}
        <p className={data.backups.offsite.status === "ok" ? undefined : "text-loss"}>
          off-site:{" "}
          {data.backups.offsite.at ? ago(data.backups.offsite.at, data.now) : "never recorded"}
        </p>
        <p>
          next {formatDayTime(data.backups.next_nightly)} ET ({until(data.backups.next_nightly, data.now)})
        </p>
      </Tile>
      <Tile label="Deployed">
        <Big>{data.deploy.version}</Big>
        <p className={data.deploy.schema_ok ? undefined : "text-loss"}>
          schema {data.deploy.revision ?? "none"}
          {data.deploy.schema_ok ? " · at head" : ` · head is ${data.deploy.expected_revision}`}
        </p>
        {data.deploy.web_started && <p>web up since {formatTime(data.deploy.web_started)}</p>}
      </Tile>
    </div>
  );
}
