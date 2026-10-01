import { StatCard } from "../components/StatCard";
import { ErrorState } from "../components/ErrorState";
import { useDrift, usePurification, useShadow } from "../hooks/useInsights";
import { formatPct, formatUsd } from "../lib/utils";

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

    </div>
  );
}
