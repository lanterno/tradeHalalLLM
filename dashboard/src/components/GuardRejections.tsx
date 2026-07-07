import { useRejections } from "../hooks/useMetrics";
import { ErrorState } from "./ErrorState";
import { formatTime } from "../lib/utils";

// Guard category → short badge label + tailwind classes.
const CATEGORY: Record<string, { label: string; cls: string }> = {
  concentration_cap: { label: "20% cap", cls: "bg-warning/15 text-warning" },
  recent_close_cooldown: { label: "Cooldown", cls: "bg-surface-hover text-gray-300" },
  stop_loss_reentry: { label: "SL re-entry", cls: "bg-loss/15 text-loss" },
  halal_screen: { label: "Halal", cls: "bg-accent/15 text-accent" },
  min_notional: { label: "Min notional", cls: "bg-surface-hover text-gray-300" },
  insufficient_funds: { label: "Funds", cls: "bg-surface-hover text-gray-300" },
  other: { label: "Other", cls: "bg-surface-hover text-muted" },
};

/**
 * "Guard rejections" — trades the cycle proposed but a guard blocked
 * (concentration cap, recent-close cooldown, stop-loss re-entry gate, halal
 * screen, …). Answers "why didn't the bot trade?" without reading the logs.
 * Self-contained (fetches its own 24h window).
 */
export function GuardRejections() {
  const { data, isLoading, isError, error, refetch } = useRejections(86400);

  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <h3 className="mb-1 text-sm font-medium uppercase tracking-wider text-muted">
        Guard Rejections
      </h3>
      <p className="mb-4 text-xs text-muted">
        Trades the bot proposed but a guard blocked — last 24h.
      </p>
      {isError ? (
        <ErrorState compact error={error} onRetry={refetch} />
      ) : isLoading ? (
        <p className="text-sm text-muted">Loading…</p>
      ) : !data?.length ? (
        <p className="text-sm text-accent">
          No guard rejections — every proposed trade passed.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                <th className="px-3 py-2">Time</th>
                <th className="px-3 py-2">Symbol</th>
                <th className="px-3 py-2">Guard</th>
                <th className="px-3 py-2">Reason</th>
              </tr>
            </thead>
            <tbody>
              {data.map((r, i) => {
                const cat = CATEGORY[r.category] ?? CATEGORY.other;
                return (
                  <tr
                    key={`${r.timestamp}-${i}`}
                    className="border-b border-border/50 hover:bg-surface-hover/50 transition-colors"
                  >
                    <td className="whitespace-nowrap px-3 py-2 text-muted">
                      {formatTime(r.timestamp)}
                    </td>
                    <td className="px-3 py-2 font-medium">{r.symbol ?? "—"}</td>
                    <td className="px-3 py-2">
                      <span
                        className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase ${cat.cls}`}
                      >
                        {cat.label}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-xs text-muted">{r.reason}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
