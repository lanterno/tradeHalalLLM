import { CheckCircle2, Circle } from "lucide-react";
import type { CoreStatus } from "../../api/types";
import { cn, formatPct } from "../../lib/utils";
import { day } from "./format";
import { Panel } from "./shared";

const DAY_STYLE: Record<string, string> = {
  run: "bg-accent",
  missed: "bg-loss",
  pending: "bg-accent/30 ring-1 ring-accent",
  before: "bg-border",
};

function Check({
  ok,
  label,
  detail,
  progress,
  when,
}: {
  ok: boolean;
  label: string;
  detail: string;
  progress?: number; // 0..1
  when?: string;
}) {
  const Icon = ok ? CheckCircle2 : Circle;
  return (
    <li className="flex items-start gap-3 py-2.5">
      <Icon className={cn("mt-0.5 h-4 w-4 shrink-0", ok ? "text-accent" : "text-muted")} />
      <div className="min-w-0 flex-1">
        <p className="text-sm text-white">{label}</p>
        <p className="text-xs text-muted">{detail}</p>
      </div>
      {progress !== undefined && (
        <div className="hidden w-24 pt-1.5 sm:block">
          <div className="h-1.5 rounded bg-border">
            <div
              className={cn("h-1.5 rounded", ok ? "bg-accent" : "bg-warning")}
              style={{ width: `${Math.min(progress, 1) * 100}%` }}
            />
          </div>
        </div>
      )}
      <p className={cn("w-28 shrink-0 text-right text-xs", ok ? "text-accent" : "text-muted")}>
        {ok ? "passing" : when}
      </p>
    </li>
  );
}

export function GatePanel({ data }: { data: CoreStatus }) {
  const r = data.readiness;
  const te = r.tracking_error;
  const passing = [
    r.days >= r.min_days,
    r.monthly_runs >= 1,
    te !== null && te <= r.max_tracking_error,
    r.gap !== null && Math.abs(r.gap) <= r.max_gap,
    r.refused === 0 && r.unfilled === 0 && r.halted === 0,
  ];
  const count = passing.filter(Boolean).length;
  return (
    <Panel title="Live-money gate" right="checked every evening · alerts when it passes">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className={cn("text-lg font-bold", r.ready ? "text-accent" : "text-warning")}>
          {r.ready ? "Ready for live keys" : "Not ready"}{" "}
          <span className="text-sm font-normal text-muted">
            · {count} of {passing.length} checks pass
          </span>
        </p>
        {!r.ready && (
          <p className="text-xs text-muted">
            earliest pass <span className="text-white">{day(r.earliest)}</span>
          </p>
        )}
      </div>

      <div className="mt-3">
        <div className="flex gap-1">
          {r.window.map((d) => (
            <div
              key={d.day}
              title={`${day(d.day)}: ${d.status}`}
              className={cn("h-4 flex-1 rounded-sm", DAY_STYLE[d.status])}
            />
          ))}
        </div>
        <div className="mt-1 flex justify-between text-[10px] text-muted">
          <span>{r.window[0] && day(r.window[0].day)}</span>
          <span>today</span>
        </div>
        <p className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-muted">
          <span>the last {r.min_days} trading days</span>
          <span className="flex items-center gap-1">
            <i className="inline-block h-2 w-2 rounded-sm bg-accent" /> ran
          </span>
          <span className="flex items-center gap-1">
            <i className="inline-block h-2 w-2 rounded-sm bg-loss" /> missed
          </span>
          <span className="flex items-center gap-1">
            <i className="inline-block h-2 w-2 rounded-sm bg-border" /> before the account
          </span>
        </p>
      </div>

      <ul className="mt-2 divide-y divide-border">
        <Check
          ok={passing[0]}
          label="Long enough on paper"
          detail={`${r.days} of ${r.min_days} trading days with the account and its book both recorded`}
          progress={r.days / r.min_days}
          when={`clears ${day(r.earliest_days)}`}
        />
        <Check
          ok={passing[1]}
          label="A monthly rebalance that traded"
          detail={
            r.monthly_runs
              ? `${r.monthly_runs} in the window`
              : data.monthly_due
                ? "due at the next run"
                : "none in the window yet"
          }
          when={`next ${day(data.next_rebalance)}`}
        />
        <Check
          ok={passing[2]}
          label={`Tracking error ≤ ${formatPct(r.max_tracking_error, 0)} a year`}
          detail={
            te === null
              ? "needs a few days of the account beside its book"
              : `${formatPct(te, 2)} a year against the forward book`
          }
          progress={te === null ? undefined : 1 - te / r.max_tracking_error}
          when="once measurable"
        />
        <Check
          ok={passing[3]}
          label={`Cumulative gap to the book ≤ ±${formatPct(r.max_gap, 0)}`}
          detail={
            r.gap === null ? "not yet measurable" : `${formatPct(r.gap, 2, { signed: true })} since the start`
          }
          when="once measurable"
        />
        <Check
          ok={passing[4]}
          label="Clean"
          detail={`${r.refused} refused, ${r.unfilled} unfilled, ${r.partial} partly filled order(s), ${r.halted} halted run(s)`}
          when="—"
        />
      </ul>
    </Panel>
  );
}
