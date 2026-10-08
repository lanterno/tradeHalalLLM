import { useState } from "react";
import {
  useBackups,
  useClearHalt,
  useCoreRisk,
  useHaltStatus,
  useReconcileRecent,
  useRiskState,
  useSetHalt,
} from "../hooks/useRisk";
import { StatCard } from "../components/StatCard";
import { ErrorState } from "../components/ErrorState";
import type { CoreRisk, RiskState } from "../api/types";
import { cn, formatDate, formatDuration, formatTime, formatUsd } from "../lib/utils";

function formatPct(v: number | null | undefined, digits = 2): string {
  if (v == null) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

const SECTORS_SHOWN = 5;
const SECTOR_COLORS = ["#4ade80", "#60a5fa", "#c084fc", "#facc15", "#fb923c"];

function CoreRiskPanel({ risk }: { risk: CoreRisk }) {
  const sectors = risk.sectors ?? [];
  const shown = sectors.slice(0, SECTORS_SHOWN);
  const rest = sectors.slice(SECTORS_SHOWN).reduce((s, x) => s + (x.weight ?? 0), 0);
  const dd = risk.drawdown_pct ?? null;
  const failing = risk.failing_screen ?? [];
  return (
    <>
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatCard
          label="Drawdown from peak"
          value={<span className={dd != null && dd < 0 ? "text-loss" : "text-accent"}>{formatPct(dd)}</span>}
          sub={
            risk.peak_day
              ? `peak ${formatUsd(risk.peak_equity ?? 0)} on ${formatDate(risk.peak_day)}`
              : `at its peak · ${risk.history_days ?? 0} day${risk.history_days === 1 ? "" : "s"} of history${
                  risk.history_from ? ` since ${formatDate(risk.history_from)}` : ""
                }`
          }
        />
        <StatCard
          label="Top 10 holdings"
          value={formatPct(risk.top10_weight, 1)}
          sub={`of equity · ${risk.positions ?? 0} holdings`}
        />
        <StatCard
          label="Largest holding"
          value={risk.largest ? formatPct(risk.largest.weight, 1) : "—"}
          sub={risk.largest?.symbol}
        />
        <StatCard
          label="Cash"
          value={formatPct(risk.cash_pct, 1)}
          sub={risk.cash != null ? formatUsd(risk.cash) : undefined}
        />
      </div>
      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-[1.4fr_1fr]">
        <div className="rounded-xl border border-border bg-surface p-4">
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-muted">Sector concentration</p>
          <div className="mb-3 flex h-3 overflow-hidden rounded-full bg-border">
            {shown.map((s, i) => (
              <i key={s.sector} style={{ width: formatPct(s.weight), background: SECTOR_COLORS[i] }} />
            ))}
            {rest > 0 && <i className="bg-gray-700" style={{ width: formatPct(rest) }} />}
          </div>
          <div className="grid gap-1 text-xs tabular-nums">
            {shown.map((s, i) => (
              <div key={s.sector} className="flex items-center gap-2">
                <i className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: SECTOR_COLORS[i] }} />
                <span>{s.sector}</span>
                <span className="ml-auto text-white">{formatPct(s.weight, 1)}</span>
              </div>
            ))}
            {rest > 0 && (
              <div className="flex items-center gap-2 text-muted">
                <i className="h-2.5 w-2.5 shrink-0 rounded-sm bg-gray-700" />
                <span>{sectors.length - shown.length} more sectors</span>
                <span className="ml-auto">{formatPct(rest, 1)}</span>
              </div>
            )}
          </div>
        </div>
        <div
          className={cn(
            "rounded-xl border bg-surface p-4 text-xs",
            failing.length ? "border-warning/30" : "border-border",
          )}
        >
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-muted">Halal screen</p>
          {failing.length ? (
            <p className="text-warning">
              {failing.length} holding{failing.length === 1 ? "" : "s"} the newest screen does not pass:{" "}
              {failing.slice(0, 8).join(", ")}
              {failing.length > 8 ? "…" : ""}. The core sells them at its next run (15:40 ET).
            </p>
          ) : (
            <p className="text-accent">Every holding passes the newest screen.</p>
          )}
          <p className="mt-3 text-muted">
            {risk.source === "ledger"
              ? "No broker snapshot yet: ledger quantities at the last close."
              : `Broker marks as of ${formatTime(risk.as_of)}${
                  (risk.age_seconds ?? 0) > 60 ? ` · ${formatDuration((risk.age_seconds ?? 0) * 1000)} old` : ""
                }.`}{" "}
            The drawdown is against the account's own daily closes.
          </p>
        </div>
      </div>
    </>
  );
}

function DayTraderRisk({ data, fetchedAt }: { data: RiskState; fetchedAt: number }) {
  const ageMs = data.pushed_at ? fetchedAt - new Date(data.pushed_at).getTime() : NaN;
  // The day-trader cycles every 15 minutes; a gap past 20 is a fault.
  const stale = Number.isFinite(ageMs) && ageMs > 20 * 60 * 1000;
  return (
    <>
      <p className="mb-3 text-xs text-muted">
        Last cycle's read,{" "}
        taken {data.pushed_at ? formatTime(data.pushed_at) : "—"}
        {Number.isFinite(ageMs) ? ` (${formatDuration(ageMs)} ago)` : ""}.
        {stale && (
          <span className="ml-2 rounded bg-loss/20 px-2 py-0.5 text-loss">stale {formatDuration(ageMs)}</span>
        )}
      </p>
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatCard
          label="Risk engine"
          value={data.is_halted ? <span className="text-loss">HALTED</span> : <span className="text-muted">OK</span>}
          sub={data.halt_reason || undefined}
        />
        <StatCard label="Heat (unrealized)" value={formatPct(data.portfolio_heat_pct)} />
        <StatCard label="Drawdown from peak" value={formatPct(data.drawdown_pct)} />
        <StatCard
          label="Avg correlation"
          value={data.avg_correlation != null ? data.avg_correlation.toFixed(2) : "—"}
        />
      </div>
      {data.summary && (
        <div className="mt-3 rounded-xl border border-border bg-surface p-4">
          <p className="whitespace-pre-line font-mono text-xs text-muted">{data.summary}</p>
        </div>
      )}
    </>
  );
}

export default function RiskAndSystem() {
  const risk = useRiskState();
  const core = useCoreRisk();
  const halt = useHaltStatus();
  const setHaltMut = useSetHalt();
  const clearHaltMut = useClearHalt();
  const reconcile = useReconcileRecent(25);
  const backups = useBackups();

  const [haltReason, setHaltReason] = useState("");
  const engaged = halt.data?.enabled === true;

  const onEngageHalt = () => {
    const reason = haltReason.trim() || "manual via dashboard";
    if (
      !confirm(
        `Engage the kill-switch with reason: "${reason}"?\n\n` +
          `The core stops buying (it still sells holdings that fail the halal screen) until you Resume.`,
      )
    )
      return;
    setHaltMut.mutate(reason);
    setHaltReason("");
  };

  const onClearHalt = () => {
    if (!confirm("Clear the kill-switch and let the core buy again?")) return;
    clearHaltMut.mutate();
  };

  return (
    <div className="space-y-6 p-4 sm:p-6">
      <h1 className="text-2xl font-bold text-white">Risk & Halt</h1>

      {/* Halt control */}
      <section className="rounded-xl border border-border bg-surface p-4">
        <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-muted">Kill-Switch</h2>
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <span className={cn("h-3 w-3 shrink-0 rounded-full", engaged ? "bg-loss animate-pulse" : "bg-accent")} />
            <div>
              <p className={cn("text-lg font-bold", engaged ? "text-loss" : "text-accent")}>
                {engaged ? "HALTED" : "Not engaged"}
              </p>
              {engaged ? (
                <p className="text-xs text-muted">
                  Set by {halt.data?.set_by ?? "—"} at {formatTime(halt.data?.set_at)}
                  {halt.data?.reason ? ` — ${halt.data.reason}` : ""}
                </p>
              ) : halt.data?.set_at ? (
                <p className="text-xs text-muted">
                  Last engaged by {halt.data.set_by ?? "—"} at {formatTime(halt.data.set_at)}; since cleared.
                </p>
              ) : null}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {engaged ? (
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
                  className="w-full min-w-0 rounded-md border border-border bg-surface-hover px-3 py-2 text-sm focus:border-accent focus:outline-none sm:w-56"
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
          While engaged, the core makes no buys, monthly rebalance included; its runs still sell any holding
          that fails the halal screen. The day-trader opens nothing while halted either, and its stop exits
          keep running. <code className="font-mono">halal-trader halt --close-all stocks</code> also
          liquidates the day-trader's account (not the core's).
        </p>
      </section>

      {/* The core: the product */}
      <section>
        <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-muted">Core portfolio risk</h2>
        {core.isError ? (
          <ErrorState compact error={core.error} onRetry={core.refetch} />
        ) : core.isLoading ? (
          <p className="text-sm text-muted">Loading…</p>
        ) : !core.data?.available ? (
          <div className="rounded-xl border border-border bg-surface p-4 text-sm text-muted">
            The core holds nothing yet: it buys at its first monthly rebalance.
          </div>
        ) : (
          <CoreRiskPanel risk={core.data} />
        )}
      </section>

      {/* The day-trader */}
      <section className="rounded-xl border border-border bg-surface p-4">
        <details open>
          <summary className="cursor-pointer text-sm font-medium uppercase tracking-wider text-muted">
            Day-trader risk
          </summary>
          <div className="mt-3">
            {risk.isError ? (
              <ErrorState compact error={risk.error} onRetry={risk.refetch} />
            ) : !risk.data?.available ? (
              <p className="text-sm text-muted">No risk read recorded.</p>
            ) : (
              <DayTraderRisk data={risk.data} fetchedAt={risk.dataUpdatedAt} />
            )}
          </div>
        </details>
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
            <table className="w-full min-w-[720px] text-sm">
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
                    <td className="whitespace-nowrap px-3 py-2 text-xs text-muted">
                      {formatTime(row.timestamp)}
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
                    <td className="min-w-48 whitespace-normal px-3 py-2 text-xs text-muted">{row.notes ?? ""}</td>
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
            <table className="w-full min-w-[480px] text-sm">
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
                    <td className="whitespace-nowrap px-3 py-2">{formatTime(b.backed_up_at)}</td>
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
