import { useState } from "react";
import {
  useBackups,
  useClearHalt,
  useHaltStatus,
  useReconcileRecent,
  useRiskState,
  useSetHalt,
} from "../hooks/useRisk";
import { StatCard } from "../components/StatCard";
import { ErrorState } from "../components/ErrorState";
import { cn } from "../lib/utils";

function formatPct(v: number | null | undefined): string {
  if (v == null) return "—";
  return `${(v * 100).toFixed(2)}%`;
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function formatTimestamp(ts: string | null | undefined): string {
  if (!ts) return "—";
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return ts;
  }
}

export default function RiskAndSystem() {
  const risk = useRiskState();
  const halt = useHaltStatus();
  const setHaltMut = useSetHalt();
  const clearHaltMut = useClearHalt();
  const reconcile = useReconcileRecent(25);
  const backups = useBackups();

  const [haltReason, setHaltReason] = useState("");

  const onEngageHalt = () => {
    const reason = haltReason.trim() || "manual via dashboard";
    if (
      !confirm(
        `Engage the kill-switch with reason: "${reason}"?\n\n` +
          `Bots will refuse new entries until you Resume.`,
      )
    )
      return;
    setHaltMut.mutate(reason);
    setHaltReason("");
  };

  const onClearHalt = () => {
    if (!confirm("Clear the kill-switch and resume trading?")) return;
    clearHaltMut.mutate();
  };

  return (
    <div className="space-y-6 p-6">
      <h1 className="text-2xl font-bold text-white">Risk & System</h1>

      {/* Halt control */}
      <section className="rounded-xl border border-border bg-surface p-4">
        <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-muted">
          Kill-Switch
        </h2>
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <span
              className={cn(
                "h-3 w-3 rounded-full",
                halt.data?.enabled ? "bg-loss animate-pulse" : "bg-accent",
              )}
            />
            <div>
              <p
                className={cn(
                  "text-lg font-bold",
                  halt.data?.enabled ? "text-loss" : "text-accent",
                )}
              >
                {halt.data?.enabled ? "HALTED" : "Running"}
              </p>
              {halt.data?.set_by && (
                <p className="text-xs text-muted">
                  {halt.data.enabled ? "Set" : "Last set"} by {halt.data.set_by}{" "}
                  at {formatTimestamp(halt.data.set_at)}
                  {halt.data.reason && ` — ${halt.data.reason}`}
                </p>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2">
            {halt.data?.enabled ? (
              <button
                onClick={onClearHalt}
                disabled={clearHaltMut.isPending}
                className="rounded-md bg-accent/10 px-4 py-2 text-sm font-medium text-accent hover:bg-accent/20 disabled:opacity-50"
              >
                {clearHaltMut.isPending ? "Resuming..." : "Resume"}
              </button>
            ) : (
              <>
                <input
                  type="text"
                  value={haltReason}
                  onChange={(e) => setHaltReason(e.target.value)}
                  placeholder="Reason (audit trail)"
                  className="w-56 rounded-md border border-border bg-surface-hover px-3 py-2 text-sm focus:border-accent focus:outline-none"
                />
                <button
                  onClick={onEngageHalt}
                  disabled={setHaltMut.isPending}
                  className="rounded-md bg-loss/20 px-4 py-2 text-sm font-medium text-loss hover:bg-loss/30 disabled:opacity-50"
                >
                  {setHaltMut.isPending ? "Engaging..." : "Engage Halt"}
                </button>
              </>
            )}
          </div>
        </div>
        <p className="mt-3 text-xs text-muted">
          Engaging the halt blocks NEW positions on every cycle. In-flight
          SL/TP exits still run. Use the CLI{" "}
          <code className="font-mono">halal-trader halt --close-all=both</code>{" "}
          for the full panic button (also liquidates positions).
        </p>
      </section>

      {/* Portfolio risk state */}
      <section>
        <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-muted">
          Portfolio Risk (last cycle{risk.data?.market ? ` · ${risk.data.market}` : ""})
          {(() => {
            // Stale-snapshot badge: surface "stale Nm" when the cycle
            // hasn't pushed in a while. Stocks cycle is 15 min, crypto
            // cycle is ~60 s — different thresholds keep the badge
            // honest for each.
            const pushedAt = risk.data?.pushed_at;
            if (!pushedAt) return null;
            const ageMs = Date.now() - new Date(pushedAt).getTime();
            if (Number.isNaN(ageMs)) return null;
            const market = risk.data?.market;
            const thresholdMs = market === "stocks" ? 20 * 60 * 1000 : 3 * 60 * 1000;
            if (ageMs < thresholdMs) return null;
            const ageMin = Math.floor(ageMs / 60000);
            return (
              <span className="ml-2 rounded bg-loss/20 px-2 py-0.5 text-xs text-loss">
                stale {ageMin}m
              </span>
            );
          })()}
        </h2>

        {risk.isError ? (
          <ErrorState compact error={risk.error} onRetry={risk.refetch} />
        ) : !risk.data?.available ? (
          <div className="rounded-xl border border-border bg-surface p-4 text-sm text-muted">
            No risk state cached yet — wait for the next cycle to populate it.
          </div>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <StatCard
                label="Status"
                value={
                  risk.data.is_halted ? (
                    <span className="text-loss">HALTED</span>
                  ) : (
                    <span className="text-accent">OK</span>
                  )
                }
                sub={risk.data.halt_reason || undefined}
              />
              <StatCard
                label="Heat (unrealized)"
                value={formatPct(risk.data.portfolio_heat_pct)}
              />
              <StatCard
                label="Drawdown from peak"
                value={formatPct(risk.data.drawdown_pct)}
              />
              <StatCard
                label="Avg correlation"
                value={
                  risk.data.avg_correlation != null
                    ? risk.data.avg_correlation.toFixed(2)
                    : "—"
                }
              />
            </div>

            {risk.data.summary && (
              <div className="mt-3 rounded-xl border border-border bg-surface p-4">
                <p className="whitespace-pre-line font-mono text-xs text-muted">
                  {risk.data.summary}
                </p>
              </div>
            )}
          </>
        )}
      </section>

      {/* Reconciliation log */}
      <section className="rounded-xl border border-border bg-surface p-4">
        <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-muted">
          Recent Reconciliation Drift
        </h2>
        {/* Cycle + monitor decide off broker truth, not the DB ledger, so a
            high stocks drift is usually a stale/phantom ledger row (broker is
            flat by EOD) rather than a real position mismatch — clearing it is
            an operator-gated fix-drift op. Say so, so the red % isn't alarming. */}
        {reconcile.data && reconcile.data.length > 0 && (
          <p className="mb-3 rounded-lg border border-border/60 bg-bg/40 px-3 py-2 text-xs text-muted">
            Drift is measured DB-ledger vs broker. Trading decisions use broker
            truth, so a large <span className="text-warning">stocks</span> drift
            is typically ledger-only (phantom rows; the broker is flat by EOD),
            not a live mismatch — clearing it is an operator-gated{" "}
            <span className="font-mono">reconcile fix-drift</span>.
          </p>
        )}
        {reconcile.isError ? (
          <ErrorState compact error={reconcile.error} onRetry={reconcile.refetch} />
        ) : reconcile.isLoading ? (
          <p className="text-sm text-muted">Loading…</p>
        ) : !reconcile.data || reconcile.data.length === 0 ? (
          <p className="text-sm text-accent">No drift recorded — DB and broker agree.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                  <th className="px-3 py-2">When</th>
                  <th className="px-3 py-2">Market</th>
                  <th className="px-3 py-2">Symbol</th>
                  <th className="px-3 py-2 text-right">DB Qty</th>
                  <th className="px-3 py-2 text-right">Broker Qty</th>
                  <th className="px-3 py-2 text-right">Drift %</th>
                  <th className="px-3 py-2 text-right">Drift $</th>
                  <th className="px-3 py-2">Notes</th>
                </tr>
              </thead>
              <tbody>
                {reconcile.data.map((row) => (
                  <tr
                    key={row.id}
                    className="border-b border-border/50 hover:bg-surface-hover/50 transition-colors"
                  >
                    <td className="px-3 py-2 text-xs text-muted">
                      {formatTimestamp(row.timestamp)}
                    </td>
                    <td className="px-3 py-2 capitalize">{row.market}</td>
                    <td className="px-3 py-2 font-mono">{row.symbol}</td>
                    <td className="px-3 py-2 text-right font-mono">
                      {row.db_quantity.toFixed(8)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {row.broker_quantity.toFixed(8)}
                    </td>
                    <td className="px-3 py-2 text-right text-loss">
                      {formatPct(row.drift_pct)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {row.drift_usd != null ? `$${row.drift_usd.toFixed(2)}` : "—"}
                    </td>
                    <td className="px-3 py-2 text-xs text-muted">{row.notes ?? ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Backups */}
      <section className="rounded-xl border border-border bg-surface p-4">
        <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-muted">
          Daily Backups
        </h2>
        {backups.isError ? (
          <ErrorState compact error={backups.error} onRetry={backups.refetch} />
        ) : backups.isLoading ? (
          <p className="text-sm text-muted">Loading…</p>
        ) : !backups.data || backups.data.length === 0 ? (
          <p className="text-sm text-warning">
            No backups found. The bot writes one every EOD; run{" "}
            <code className="font-mono">halal-trader backup</code> to create one now.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                  <th className="px-3 py-2">Date</th>
                  <th className="px-3 py-2">Path</th>
                  <th className="px-3 py-2 text-right">Size</th>
                </tr>
              </thead>
              <tbody>
                {backups.data.slice(0, 14).map((b) => (
                  <tr
                    key={b.path}
                    className="border-b border-border/50 hover:bg-surface-hover/50 transition-colors"
                  >
                    <td className="px-3 py-2">
                      {formatTimestamp(b.backed_up_at).split(",")[0]}
                    </td>
                    <td className="px-3 py-2 font-mono text-xs text-muted">
                      {b.path}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {formatBytes(b.size_bytes)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
