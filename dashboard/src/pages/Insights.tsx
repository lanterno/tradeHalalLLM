import { StatCard } from "../components/StatCard";
import { ErrorState } from "../components/ErrorState";
import {
  useDrift,
  usePurification,
  useRegret,
  useShadow,
  useThesis,
  useVelocity,
  useWhale,
} from "../hooks/useInsights";
import { cn, formatPct, formatUsd, pnlColor } from "../lib/utils";

const SHADOW_LEVEL_TONE: Record<string, string> = {
  ok: "text-accent",
  watch: "text-amber-400",
  diverged: "text-loss",
};

const DRIFT_STATE_TONE: Record<string, string> = {
  stable: "text-accent",
  warming_up: "text-muted",
  drift: "text-loss",
};

function EmptyTile({ label, message }: { label: string; message: string }) {
  return (
    <StatCard
      label={label}
      value={<span className="text-sm font-normal text-muted">—</span>}
      sub={message}
    />
  );
}

export default function Insights() {
  const {
    data: drift,
    isError: driftIsError,
    error: driftError,
    refetch: driftRefetch,
  } = useDrift();
  const {
    data: shadow,
    isError: shadowIsError,
    error: shadowError,
    refetch: shadowRefetch,
  } = useShadow();
  const {
    data: regret,
    isError: regretIsError,
    error: regretError,
    refetch: regretRefetch,
  } = useRegret(200);
  const {
    data: thesis,
    isError: thesisIsError,
    error: thesisError,
    refetch: thesisRefetch,
  } = useThesis(200);
  const {
    data: whale,
    isError: whaleIsError,
    error: whaleError,
    refetch: whaleRefetch,
  } = useWhale();
  const {
    data: velocity,
    isError: velocityIsError,
    error: velocityError,
    refetch: velocityRefetch,
  } = useVelocity();
  const {
    data: purification,
    isError: purificationIsError,
    error: purificationError,
    refetch: purificationRefetch,
  } = usePurification();

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Insights</h1>
        <p className="text-xs text-muted">Auto-refresh every 30s</p>
      </div>

      {/* Top row: live model-health tiles */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-4">
        {driftIsError ? (
          <ErrorState compact error={driftError} onRetry={driftRefetch} />
        ) : drift ? (
          <StatCard
            label="Concept Drift"
            value={
              <span className={DRIFT_STATE_TONE[drift.state] ?? "text-white"}>
                {drift.state}
              </span>
            }
            sub={`n=${drift.n} · ${drift.drift_count} drift event(s)`}
          />
        ) : (
          <EmptyTile label="Concept Drift" message="No residuals yet" />
        )}

        {shadowIsError ? (
          <ErrorState compact error={shadowError} onRetry={shadowRefetch} />
        ) : shadow && shadow.metrics ? (
          <StatCard
            label="Shadow Bot"
            value={
              <span className={SHADOW_LEVEL_TONE[shadow.level] ?? "text-white"}>
                {shadow.level}
              </span>
            }
            sub={`Δ mean ${formatPct(shadow.metrics.mean_diff_pct)} · ${shadow.metrics.direction}`}
          />
        ) : (
          <EmptyTile label="Shadow Bot" message="Waiting for samples" />
        )}

        {regretIsError ? (
          <ErrorState compact error={regretError} onRetry={regretRefetch} />
        ) : regret && regret.n > 0 ? (
          <StatCard
            label="Regret (mean)"
            value={
              <span
                className={cn(
                  regret.mean_regret <= 0.3
                    ? "text-accent"
                    : regret.mean_regret <= 0.5
                    ? "text-amber-400"
                    : "text-loss",
                )}
              >
                {regret.mean_regret.toFixed(2)}
              </span>
            }
            sub={`n=${regret.n} · ${regret.missed_edge_count} missed · ${regret.tail_loss_count} tail`}
          />
        ) : (
          <EmptyTile label="Regret (mean)" message="No closed trades" />
        )}

        {purificationIsError ? (
          <ErrorState compact error={purificationError} onRetry={purificationRefetch} />
        ) : purification ? (
          <StatCard
            label="Purification Due"
            value={
              <span className="text-white">{formatUsd(purification.total_usd)}</span>
            }
            sub={`${purification.n_entries} entries · ${formatUsd(purification.disbursed_total_usd)} disbursed`}
          />
        ) : (
          <EmptyTile label="Purification Due" message="No closed wins yet" />
        )}
      </div>

      {/* Thesis attribution table */}
      <section className="rounded-xl border border-border bg-surface">
        <header className="flex items-center justify-between border-b border-border px-4 py-3">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-muted">
            Thesis Attribution
          </h2>
          {thesis && thesis.kill_candidates.length > 0 && (
            <span className="rounded-md bg-loss/10 px-2 py-1 text-xs font-medium text-loss">
              kill: {thesis.kill_candidates.join(", ")}
            </span>
          )}
        </header>
        {thesisIsError ? (
          <ErrorState compact error={thesisError} onRetry={thesisRefetch} />
        ) : !thesis || thesis.rows.length === 0 ? (
          <p className="px-4 py-6 text-sm text-muted">
            No closed trades yet — tags accrue as the post-close hook fires.
          </p>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-xs uppercase tracking-wider text-muted">
              <tr>
                <th className="px-4 py-2 text-left">Tag</th>
                <th className="px-4 py-2 text-right">N</th>
                <th className="px-4 py-2 text-right">Win%</th>
                <th className="px-4 py-2 text-right">Avg P&L</th>
              </tr>
            </thead>
            <tbody>
              {thesis.rows
                .slice()
                .sort((a, b) => b.n_trades - a.n_trades)
                .map((row) => (
                  <tr key={row.tag} className="border-t border-border">
                    <td className="px-4 py-2 font-medium">{row.tag}</td>
                    <td className="px-4 py-2 text-right tabular-nums">
                      {row.n_trades}
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums">
                      {formatPct(row.win_rate)}
                    </td>
                    <td
                      className={cn(
                        "px-4 py-2 text-right tabular-nums",
                        pnlColor(row.avg_pnl_pct),
                      )}
                    >
                      {formatPct(row.avg_pnl_pct)}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        )}
      </section>

      {/* Whale flows */}
      <section className="rounded-xl border border-border bg-surface">
        <header className="border-b border-border px-4 py-3">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-muted">
            On-chain Whale Flows
          </h2>
        </header>
        {whaleIsError ? (
          <ErrorState compact error={whaleError} onRetry={whaleRefetch} />
        ) : !whale || whale.flows.length === 0 ? (
          <p className="px-4 py-6 text-sm text-muted">
            No flows recorded — set ETHERSCAN_API_KEY to enable.
          </p>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-xs uppercase tracking-wider text-muted">
              <tr>
                <th className="px-4 py-2 text-left">Symbol</th>
                <th className="px-4 py-2 text-right">Pressure</th>
                <th className="px-4 py-2 text-right">In ($)</th>
                <th className="px-4 py-2 text-right">Out ($)</th>
                <th className="px-4 py-2 text-left">Label</th>
              </tr>
            </thead>
            <tbody>
              {whale.flows.map((f) => (
                <tr key={f.symbol} className="border-t border-border">
                  <td className="px-4 py-2 font-medium">{f.symbol}</td>
                  <td
                    className={cn(
                      "px-4 py-2 text-right tabular-nums",
                      f.inflow_pressure > 0.3
                        ? "text-loss"
                        : f.inflow_pressure < -0.3
                        ? "text-accent"
                        : "text-white",
                    )}
                  >
                    {f.inflow_pressure >= 0 ? "+" : ""}
                    {f.inflow_pressure.toFixed(2)}
                  </td>
                  <td className="px-4 py-2 text-right tabular-nums">
                    {formatUsd(f.inflow_to_exchange_usd)}
                  </td>
                  <td className="px-4 py-2 text-right tabular-nums">
                    {formatUsd(f.outflow_from_exchange_usd)}
                  </td>
                  <td className="px-4 py-2 text-muted">{f.label}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {/* Velocity */}
      <section className="rounded-xl border border-border bg-surface">
        <header className="border-b border-border px-4 py-3">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-muted">
            Reddit Mention Velocity
          </h2>
        </header>
        {velocityIsError ? (
          <ErrorState compact error={velocityError} onRetry={velocityRefetch} />
        ) : !velocity || velocity.results.length === 0 ? (
          <p className="px-4 py-6 text-sm text-muted">
            No velocity results — fetcher runs each crypto cycle.
          </p>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-xs uppercase tracking-wider text-muted">
              <tr>
                <th className="px-4 py-2 text-left">Symbol</th>
                <th className="px-4 py-2 text-right">Velocity</th>
                <th className="px-4 py-2 text-right">Novelty</th>
                <th className="px-4 py-2 text-right">Recent</th>
                <th className="px-4 py-2 text-right">Older</th>
                <th className="px-4 py-2 text-left">Label</th>
              </tr>
            </thead>
            <tbody>
              {velocity.results
                .slice()
                .sort((a, b) => b.velocity - a.velocity)
                .map((r) => (
                  <tr key={r.symbol} className="border-t border-border">
                    <td className="px-4 py-2 font-medium">{r.symbol}</td>
                    <td
                      className={cn(
                        "px-4 py-2 text-right tabular-nums",
                        r.label === "surge"
                          ? "text-accent"
                          : r.label === "decay"
                          ? "text-loss"
                          : "text-white",
                      )}
                    >
                      {r.velocity.toFixed(2)}×
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums">
                      {formatPct(r.novelty)}
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums">
                      {r.n_recent}
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums">
                      {r.n_older}
                    </td>
                    <td className="px-4 py-2 text-muted">{r.label}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
