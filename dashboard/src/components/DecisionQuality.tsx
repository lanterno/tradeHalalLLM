import { useDecisionQuality } from "../hooks/useDecisions";
import type { EdgeStat } from "../api/types";
import { ErrorState } from "./ErrorState";
import { Panel } from "./Panel";
import { StatCard } from "./StatCard";
import {
  cn,
  formatDay,
  formatPct,
  formatUsd,
  pnlColor,
  relativeTime,
} from "../lib/utils";

const tone = (v: number | null | undefined) => pnlColor(v ?? 0);

/** A per-trip edge, coloured by sign. */
function Edge({ stat }: { stat: EdgeStat }) {
  return (
    <span className={tone(stat.mean)}>
      {formatPct(stat.mean, 2, { signed: true })}
    </span>
  );
}

/** "t −1.4 over 6 sessions, not yet significant": a |t| under 2 is not yet evidence. */
function evidence(stat: EdgeStat): string {
  if (stat.t == null) return `${stat.trips} trips`;
  const strength =
    Math.abs(stat.t) >= 2 ? "significant" : "not yet significant";
  return `t ${stat.t >= 0 ? "+" : "−"}${Math.abs(stat.t).toFixed(1)} over ${stat.sessions} sessions, ${strength}`;
}

/**
 * Whether the day-trader's decisions beat simply holding SPY: each closed
 * round trip of the last 30 days, scored by the evening run. Buys are judged
 * from the fill to the session's close, sells from the sale to the close.
 */
export function DecisionQuality() {
  const { data, isLoading, isError, error, refetch } = useDecisionQuality();

  if (isError) {
    return (
      <Panel title="Decision quality vs SPY">
        <ErrorState compact error={error} onRetry={refetch} />
      </Panel>
    );
  }
  if (isLoading) {
    return (
      <Panel title="Decision quality vs SPY">
        <p className="text-sm text-muted">Loading…</p>
      </Panel>
    );
  }
  if (!data || !data.trips) {
    return (
      <Panel title="Decision quality vs SPY">
        <p className="text-sm text-muted">
          No scored round trips yet: the evening run scores the last 30 days
          after the close.
        </p>
      </Panel>
    );
  }

  return (
    <Panel
      title="Decision quality vs SPY"
      right={`${data.trips} round trips since ${formatDay(data.start)} · scored ${relativeTime(data.computed_at)}`}
    >
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <StatCard
          label="Buys: fill to close"
          value={<Edge stat={data.entry_edge} />}
          sub={`vs SPY per trip; ${evidence(data.entry_edge)}`}
        />
        <StatCard
          label="Sells: saved by selling"
          value={<Edge stat={data.exit_edge} />}
          sub={`vs holding to the close; ${evidence(data.exit_edge)}`}
        />
        <StatCard
          label="Round trip vs SPY"
          value={<Edge stat={data.excess} />}
          sub={`${formatUsd(data.pnl_usd, { signed: true })} P&L; ${evidence(data.excess)}`}
        />
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="overflow-x-auto">
          <p className="mb-1 text-xs text-muted">
            Buys by time of entry (New York)
          </p>
          <table className="w-full text-sm tabular-nums">
            <tbody>
              {Object.entries(data.by_hour).map(([hour, s]) => (
                <tr key={hour} className="border-t border-border/50">
                  <td className="py-1 pr-3 text-muted">{hour}</td>
                  <td className="py-1 pr-3 text-right">{s.trips} trips</td>
                  <td className="py-1 text-right">
                    <Edge stat={s} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="overflow-x-auto">
          <p className="mb-1 text-xs text-muted">
            By day: buys vs SPY to the close, P&amp;L
          </p>
          <table className="w-full text-sm tabular-nums">
            <tbody>
              {data.days
                .slice(-10)
                .reverse()
                .map((d) => (
                  <tr key={d.day} className="border-t border-border/50">
                    <td className="py-1 pr-3 text-muted">{formatDay(d.day)}</td>
                    <td className="py-1 pr-3 text-right">{d.trips} trips</td>
                    <td
                      className={cn("py-1 pr-3 text-right", tone(d.entry_edge))}
                    >
                      {formatPct(d.entry_edge, 2, { signed: true })}
                    </td>
                    <td className={cn("py-1 text-right", tone(d.pnl_usd))}>
                      {formatUsd(d.pnl_usd, { signed: true })}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </Panel>
  );
}
