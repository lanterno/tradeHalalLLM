import type { Trade } from "../api/types";
import { cn, entityOf, formatTime, formatUsd, formatQty, pnlColor } from "../lib/utils";
import { entityLabel, useMarket } from "../lib/market";

interface TradesTableProps {
  trades: Trade[];
  showReasoning?: boolean;
  compact?: boolean;
}

export function TradesTable({
  trades,
  showReasoning = false,
  compact = false,
}: TradesTableProps) {
  const { market } = useMarket();

  if (!trades.length) {
    return (
      <p className="py-8 text-center text-sm text-muted">No trades yet.</p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
            <th className="px-3 py-2">Time</th>
            <th className="px-3 py-2">{entityLabel(market)}</th>
            <th className="px-3 py-2">Side</th>
            <th className="px-3 py-2 text-right">Qty</th>
            <th className="px-3 py-2 text-right">Price</th>
            {!compact && (
              <>
                <th className="px-3 py-2 text-right">Exit</th>
                <th className="px-3 py-2 text-right">P&L</th>
                <th className="px-3 py-2">Reason</th>
              </>
            )}
            <th className="px-3 py-2">Status</th>
            {showReasoning && <th className="px-3 py-2">Reasoning</th>}
          </tr>
        </thead>
        <tbody>
          {trades.map((t) => {
            // Entry basis: crypto rows carry entry_price, stocks rows carry
            // filled_price. Use != null (not truthiness) so a legit 0 isn't
            // dropped. Mirrors the CSV export basis in Trades.tsx.
            const entry = t.entry_price ?? t.filled_price ?? t.price;
            const pnl =
              t.exit_price != null && entry != null
                ? (t.exit_price - entry) * t.quantity
                : null;

            return (
              <tr
                key={t.id}
                className="border-b border-border/50 hover:bg-surface-hover/50 transition-colors"
              >
                <td className="whitespace-nowrap px-3 py-2 text-muted">
                  {formatTime(t.timestamp)}
                </td>
                <td className="px-3 py-2 font-medium">{entityOf(t)}</td>
                <td className="px-3 py-2">
                  <span
                    className={cn(
                      "inline-block rounded px-1.5 py-0.5 text-xs font-semibold uppercase",
                      t.side === "buy"
                        ? "bg-accent/15 text-accent"
                        : "bg-loss/15 text-loss",
                    )}
                  >
                    {t.side}
                  </span>
                </td>
                <td className="px-3 py-2 text-right font-mono">
                  {formatQty(t.quantity)}
                </td>
                <td className="px-3 py-2 text-right font-mono">
                  {formatUsd(t.price)}
                </td>
                {!compact && (
                  <>
                    <td className="px-3 py-2 text-right font-mono">
                      {t.exit_price ? formatUsd(t.exit_price) : "—"}
                    </td>
                    <td
                      className={cn(
                        "px-3 py-2 text-right font-mono",
                        pnl !== null ? pnlColor(pnl) : "text-muted",
                      )}
                    >
                      {pnl !== null ? formatUsd(pnl) : "—"}
                    </td>
                    <td className="px-3 py-2 text-muted">
                      {t.exit_reason ?? "—"}
                    </td>
                  </>
                )}
                <td className="px-3 py-2 text-muted">{t.status}</td>
                {showReasoning && (
                  <td
                    className="max-w-xs truncate px-3 py-2 text-xs text-muted"
                    title={t.llm_reasoning}
                  >
                    {t.llm_reasoning?.slice(0, 80)}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
