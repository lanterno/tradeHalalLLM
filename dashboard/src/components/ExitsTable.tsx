import type { ExitStats } from "../api/types";
import { cn, formatDuration, formatPct, formatUsd } from "../lib/utils";

const LABELS: Record<string, string> = {
  llm_sell: "LLM sell",
  llm_close_position: "LLM close",
  eod_close_all: "End-of-day close",
  stop_loss: "Stop-loss / trailing",
  take_profit: "Take-profit",
  trend_break: "Trend break",
  balance_exhausted: "Broker held none",
  reconcile_adjustment: "Reconcile",
};
const label = (reason: string) => LABELS[reason] ?? reason.replaceAll("_", " ");
const tone = (v: number | null) =>
  v == null || v === 0 ? "text-muted" : v > 0 ? "text-accent" : "text-loss";

/** Each way a round trip closed, and how those trades did: the evidence for an exit rule. */
export function ExitsTable({ exits }: { exits: ExitStats[] }) {
  if (!exits.length) {
    return <p className="py-12 text-center text-sm text-muted">No closed trades in the window.</p>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm tabular-nums">
        <thead className="text-left text-[11px] uppercase tracking-wider text-muted">
          <tr>
            <th className="py-1.5 pr-3">Exit</th>
            <th className="py-1.5 pr-3 text-right">Trades</th>
            <th className="py-1.5 pr-3 text-right">Avg</th>
            <th className="hidden py-1.5 pr-3 text-right sm:table-cell">Win rate</th>
            <th className="hidden py-1.5 pr-3 text-right sm:table-cell">P&amp;L</th>
            <th className="py-1.5 pr-3 text-right">Held</th>
            <th className="py-1.5 text-right">First hour</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {exits.map((e) => (
            <tr key={e.reason}>
              <td className="whitespace-normal py-2 pr-3 text-white">{label(e.reason)}</td>
              <td className="py-2 pr-3 text-right">{e.trades}</td>
              <td className={cn("py-2 pr-3 text-right", tone(e.avg_pct))}>
                {formatPct(e.avg_pct, 2, { signed: true })}
              </td>
              <td className="hidden py-2 pr-3 text-right sm:table-cell">{formatPct(e.win_rate, 0)}</td>
              <td className={cn("hidden py-2 pr-3 text-right sm:table-cell", tone(e.total_pnl))}>
                {formatUsd(e.total_pnl, { signed: true })}
              </td>
              <td className="py-2 pr-3 text-right text-muted">
                {formatDuration(e.avg_hold_minutes * 60_000)}
              </td>
              <td className="py-2 text-right">
                {e.first_hour_trades ? (
                  <>
                    <span className={tone(e.first_hour_avg_pct)}>
                      {formatPct(e.first_hour_avg_pct, 2, { signed: true })}
                    </span>
                    <span className="text-muted"> · {e.first_hour_trades}</span>
                  </>
                ) : (
                  <span className="text-muted">—</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-[11px] leading-relaxed text-muted">
        Average return per trade, entry to exit. "First hour": the trades that closed within an
        hour of their entry, the case a minimum-hold rule would change.
      </p>
    </div>
  );
}
