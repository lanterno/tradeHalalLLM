import { useState } from "react";
import { usePositions } from "../hooks/usePositions";
import { StatCard } from "../components/StatCard";
import { ErrorState } from "../components/ErrorState";
import type { AccountPositions, Holding } from "../api/types";
import { cn, formatDate, formatDuration, formatPct, formatQty, formatTime, formatUsd, pnlColor } from "../lib/utils";

const SHOWN = 15;
// Snapshots stop outside the session; flag only marks older than a long weekend.
const STALE_SECONDS = 3 * 86_400;

const STATUS_STYLE: Record<string, string> = {
  active: "border-accent/35 bg-accent/5 text-accent",
  disabled: "border-border bg-bg/40 text-muted",
};

function MarkedAt({ account }: { account: AccountPositions }) {
  if (account.source === "ledger") {
    return (
      <span className="text-warning">
        No broker snapshot yet · ledger quantities at the{" "}
        {account.as_of ? `${formatDate(account.as_of)} close` : "last close"}
      </span>
    );
  }
  const age = account.age_seconds ?? 0;
  return (
    <span className={cn(age > STALE_SECONDS ? "text-warning" : "text-muted")}>
      Broker marks as of {formatTime(account.as_of)}
      {age > 60 ? ` · ${formatDuration(age * 1000)} old` : ""}
    </span>
  );
}

function HoldingsTable({ account }: { account: AccountPositions }) {
  const [all, setAll] = useState(false);
  const dayTrader = account.account === "paper";
  const rows: Holding[] = all ? account.positions : account.positions.slice(0, SHOWN);
  const hidden = account.positions.length - rows.length;

  if (!account.positions.length) {
    return (
      <p className="rounded-xl border border-border bg-surface p-4 text-sm text-muted">
        Nothing held.
      </p>
    );
  }
  return (
    <div className="rounded-xl border border-border bg-surface p-2 sm:p-4">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-sm tabular-nums">
          <thead>
            <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
              <th className="px-3 py-2">Symbol</th>
              <th className="px-3 py-2 text-right">Qty</th>
              <th className="px-3 py-2 text-right">Avg cost</th>
              <th className="px-3 py-2 text-right">Price</th>
              <th className="px-3 py-2 text-right">Value</th>
              <th className="px-3 py-2 text-right">Weight</th>
              <th className="px-3 py-2 text-right">Unrealized P&L</th>
              <th className="px-3 py-2 text-right">Today</th>
              {dayTrader && (
                <>
                  <th className="px-3 py-2 text-right">Stop</th>
                  <th className="px-3 py-2 text-right">Target</th>
                  <th className="px-3 py-2">Opened</th>
                </>
              )}
            </tr>
          </thead>
          <tbody>
            {rows.map((p) => (
              <tr key={p.symbol} className="border-b border-border/50 hover:bg-surface-hover/50 transition-colors">
                <td className="px-3 py-2 font-medium text-white">{p.symbol}</td>
                <td className="px-3 py-2 text-right font-mono">{p.qty == null ? "—" : formatQty(p.qty, 4)}</td>
                <td className="px-3 py-2 text-right font-mono text-muted">{formatUsd(p.avg_entry)}</td>
                <td className="px-3 py-2 text-right font-mono">{formatUsd(p.price)}</td>
                <td className="px-3 py-2 text-right font-mono">{formatUsd(p.market_value)}</td>
                <td className="px-3 py-2 text-right font-mono text-muted">
                  {p.weight == null ? "—" : `${(p.weight * 100).toFixed(1)}%`}
                </td>
                <td className={cn("px-3 py-2 text-right font-mono font-semibold", pnlColor(p.unrealized_pl ?? 0))}>
                  {formatUsd(p.unrealized_pl)}
                  <span className="ml-1 text-[10px] font-normal">({formatPct(p.unrealized_pl_pct, 2, { signed: true })})</span>
                </td>
                <td className={cn("px-3 py-2 text-right font-mono", pnlColor(p.change_today ?? 0))}>
                  {formatPct(p.change_today, 1, { signed: true })}
                </td>
                {dayTrader && (
                  <>
                    <td className="px-3 py-2 text-right font-mono text-loss">{formatUsd(p.stop_loss)}</td>
                    <td className="px-3 py-2 text-right font-mono text-accent">{formatUsd(p.target_price)}</td>
                    <td className="whitespace-nowrap px-3 py-2 text-muted">{formatTime(p.opened_at)}</td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {account.positions.length > SHOWN && (
        <button
          type="button"
          onClick={() => setAll(!all)}
          className="mt-3 px-3 text-xs text-muted hover:text-white"
        >
          {all ? "Show the largest only ▴" : `Show all ${account.positions.length} holdings (${hidden} more) ▾`}
        </button>
      )}
    </div>
  );
}

function AccountSection({ account }: { account: AccountPositions }) {
  const pl = account.unrealized_pl;
  const plPct = pl != null && account.invested - pl ? pl / (account.invested - pl) : null;
  return (
    <section className="space-y-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h2 className="flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-muted">
          {account.label}
          <span
            className={cn(
              "rounded-full border px-2 py-0.5 text-[10px] font-semibold tracking-wide",
              STATUS_STYLE[account.status] ?? STATUS_STYLE.disabled,
            )}
          >
            {account.status}
          </span>
        </h2>
        <p className="text-xs">
          <MarkedAt account={account} />
        </p>
      </div>
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatCard label="Market value" value={formatUsd(account.invested)} />
        <StatCard
          label="Unrealized P&L"
          value={<span className={pnlColor(pl ?? 0)}>{formatUsd(pl)}</span>}
          sub={plPct == null ? undefined : `${formatPct(plPct, 2, { signed: true })} on cost`}
        />
        <StatCard label="Positions" value={account.positions.length} />
        <StatCard
          label="Cash"
          value={formatUsd(account.cash)}
          sub={account.equity != null ? `equity ${formatUsd(account.equity)}` : undefined}
        />
      </div>
      <HoldingsTable account={account} />
    </section>
  );
}

export default function Positions() {
  const { data, isLoading, isError, error, refetch } = usePositions();

  return (
    <div className="space-y-8 p-4 sm:p-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Positions</h1>
        <p className="mt-1 text-xs text-muted">
          What each account holds, marked at the broker's prices from the bot's minute snapshot: the same
          figures as the home page.
        </p>
      </div>

      {isError ? (
        <ErrorState error={error} onRetry={refetch} />
      ) : isLoading || !data ? (
        <p className="py-8 text-center text-sm text-muted">Loading…</p>
      ) : !data.accounts.length ? (
        <p className="py-8 text-center text-sm text-muted">No account has reported any holdings yet.</p>
      ) : (
        data.accounts.map((a) => <AccountSection key={a.account} account={a} />)
      )}
    </div>
  );
}
