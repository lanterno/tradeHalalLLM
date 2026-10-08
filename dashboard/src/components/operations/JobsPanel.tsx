import type { OperationsStatus, OpsJob } from "../../api/types";
import { cn, formatDay } from "../../lib/utils";
import { Panel } from "../Panel";
import { etDay, etMinutes, until, when } from "./format";
import { StatusBadge } from "./StatusBadge";

// The timeline's axis, in New York minutes: 08:00 to 22:00.
const FROM = 8 * 60;
const TO = 22 * 60;
const at = (minutes: number) => `${((minutes - FROM) / (TO - FROM)) * 100}%`;
const inRange = (minutes: number) => minutes >= FROM && minutes <= TO;

// Each job's name on the timeline, where the table's would not fit.
const SHORT: Record<string, string> = {
  "recommendation.daily": "pick",
  "core.trade": "core",
  "stock.eod": "EOD",
  "stock.ledger": "ledger",
  "digest.weekly": "digest",
  "research.daily": "research",
};

function jobState(j: OpsJob): string {
  if (j.status === "stale" || j.status === "disabled") return j.status;
  if (j.today_at && !j.ran_today && !j.due_today) return "pending";
  if (j.status === "unknown") return "unknown";
  return "ok";
}

function Timeline({ data }: { data: OperationsStatus }) {
  const m = data.market;
  const now = etMinutes(data.now);
  const today = data.jobs.filter((j) => j.today_at);
  const session = m.opens && m.closes ? [etMinutes(m.opens), etMinutes(m.closes)] : null;
  const ticks = [8, 10, 12, 14, 16, 18, 20, 22];
  return (
    <div className="mb-4 overflow-x-auto">
      <div className="relative mx-2 h-20 min-w-[560px]">
        {/* the snapshot window, then the session */}
        {m.trading_day && (
          <div
            className="absolute top-10 h-px border-t border-dashed border-muted/60"
            style={{ left: at(9 * 60), width: `calc(${at(17 * 60)} - ${at(9 * 60)})` }}
          />
        )}
        {session && (
          <div
            className="absolute top-[38px] h-1.5 rounded bg-accent/25"
            style={{ left: at(session[0]), width: `calc(${at(session[1])} - ${at(session[0])})` }}
          />
        )}
        <div className="absolute inset-x-0 top-[41px] h-px bg-border" />
        {today.map((j, i) => {
          const minutes = etMinutes(j.today_at!);
          const state = jobState(j);
          return (
            <div
              key={j.component}
              className="absolute -translate-x-1/2"
              style={{ left: at(minutes), top: i % 2 ? 46 : 10 }}
              title={`${j.label} ${j.at} ET: ${j.ran_today ? "ran" : state}`}
            >
              {i % 2 === 0 && (
                <p className="mb-1 whitespace-nowrap text-center text-[10px] text-white">
                  {SHORT[j.component] ?? j.label}
                </p>
              )}
              <span
                className={cn(
                  "mx-auto block h-3 w-3 rounded-full border-2",
                  j.ran_today
                    ? "border-accent bg-accent"
                    : state === "stale"
                      ? "border-loss bg-loss"
                      : "border-sky-400 bg-bg",
                  i % 2 ? "" : "mt-[13px]",
                )}
              />
              {i % 2 === 1 && (
                <p className="mt-1 whitespace-nowrap text-center text-[10px] text-white">
                  {SHORT[j.component] ?? j.label}
                </p>
              )}
            </div>
          );
        })}
        {inRange(now) && etDay(data.now) === m.today && (
          <div className="absolute top-6 h-10 w-0.5 bg-white" style={{ left: at(now) }}>
            <span className="absolute -top-4 -translate-x-1/2 whitespace-nowrap text-[10px] font-semibold text-white">
              now
            </span>
          </div>
        )}
        <div className="absolute inset-x-0 bottom-0 flex justify-between text-[10px] text-muted">
          {ticks.map((h) => (
            <span key={h}>{String(h).padStart(2, "0")}:00</span>
          ))}
        </div>
      </div>
      <p className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-muted">
        <span className="flex items-center gap-1">
          <i className="inline-block h-1.5 w-4 rounded bg-accent/25" /> regular session
        </span>
        <span className="flex items-center gap-1">
          <i className="inline-block w-4 border-t border-dashed border-muted" /> per-minute snapshots
        </span>
        <span className="flex items-center gap-1">
          <i className="inline-block h-2 w-2 rounded-full bg-accent" /> ran today
        </span>
        <span className="flex items-center gap-1">
          <i className="inline-block h-2 w-2 rounded-full border border-sky-400" /> still to come
        </span>
        <span>all times ET</span>
      </p>
    </div>
  );
}

export function JobsPanel({ data }: { data: OperationsStatus }) {
  const m = data.market;
  return (
    <Panel
      title="Daily jobs"
      right={
        m.trading_day
          ? `${formatDay(m.today)} · trading day · closes ${m.closes!.slice(11, 16)}`
          : `${formatDay(m.today)} · market closed`
      }
    >
      {m.trading_day && <Timeline data={data} />}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-[11px] uppercase tracking-wider text-muted">
            <tr>
              <th className="hidden py-1.5 pr-3 sm:table-cell">ET</th>
              <th className="py-1.5 pr-3">Job</th>
              <th className="hidden py-1.5 pr-3 sm:table-cell">Last run</th>
              <th className="hidden py-1.5 pr-3 md:table-cell">What it did</th>
              <th className="py-1.5 pr-3">Next</th>
              <th className="py-1.5 text-right">Watched</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {data.jobs.map((j) => {
              const state = jobState(j);
              return (
                <tr key={j.component} className="align-top">
                  <td className="hidden py-2 pr-3 tabular-nums text-muted sm:table-cell">
                    {j.weekday ? `${j.weekday} ${j.at}` : j.at}
                  </td>
                  <td className="whitespace-normal py-2 pr-3">
                    <p className="font-semibold text-white">{j.label}</p>
                    <p className="font-mono text-[10px] text-muted">
                      <span className="sm:hidden">{j.weekday ? `${j.weekday} ${j.at}` : j.at} ET · </span>
                      {j.component}
                    </p>
                  </td>
                  <td className="hidden py-2 pr-3 tabular-nums sm:table-cell">
                    {j.last ? when(j.last, data.now) : <span className="text-muted">not yet</span>}
                  </td>
                  <td className="hidden whitespace-normal py-2 pr-3 text-xs text-muted md:table-cell">
                    {j.summary ?? (j.last ? "—" : "nothing recorded yet")}
                    {j.reason && j.status !== "ok" && <p className="text-loss">{j.reason}</p>}
                  </td>
                  <td className="py-2 pr-3 tabular-nums">
                    <p>{when(j.next, data.now)}</p>
                    <p className="text-[11px] text-sky-300">{until(j.next, data.now)}</p>
                  </td>
                  <td className="py-2 text-right">
                    <StatusBadge status={state} title={j.reason} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {data.fleet.problems.length > 0 && (
        <div className="mt-3 rounded-lg border border-warning/30 bg-warning/5 p-3 text-xs text-warning">
          <p className="font-semibold">To look at</p>
          <ul className="mt-1 list-disc space-y-0.5 pl-4">
            {data.fleet.problems.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        </div>
      )}
    </Panel>
  );
}
