import { useHealth, useSystemStatus, useConfig, useCoreConfig } from "../hooks/useSystem";
import { StatCard } from "../components/StatCard";
import { ErrorState } from "../components/ErrorState";
import { cn } from "../lib/utils";

type Value = string | number | boolean | null | undefined | (string | number)[];

/** What each core parameter means, so the table reads without the source. */
const CORE_LABELS: Record<string, [string, (v: Value) => string]> = {
  core_enabled: ["Core trading enabled", (v) => (v ? "yes" : "no")],
  core_paper: ["Account", (v) => (v ? "paper" : "LIVE money")],
  core_keys_set: ["Broker keys set", (v) => (v ? "yes" : "no")],
  core_top_n: ["Holdings targeted (largest that pass the screen)", (v) => String(v)],
  core_rebalance_band: [
    "Rebalance band (a holding trades only this far off target)",
    (v) => `±${(Number(v) * 100).toFixed(0)}% of its target weight`,
  ],
  core_band_floor: ["Band floor", (v) => `${(Number(v) * 100).toFixed(1)}% of equity`],
  core_min_trade_usd: ["Smallest trade", (v) => `$${v}`],
  core_cash_buffer: ["Cash left uninvested", (v) => `${(Number(v) * 100).toFixed(0)}% of equity`],
  core_max_screen_age_days: ["Oldest halal screen it will trade on", (v) => `${v} days`],
  core_trades_at_et: ["Trades at", (v) => `${v} ET, monthly rebalance + daily forced sales`],
  day_trader_enabled: ["Day-trader", (v) => (v ? "running" : "retired")],
};

function ConfigTable({ rows }: { rows: [string, string][] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
            <th className="px-3 py-2">Setting</th>
            <th className="px-3 py-2">Value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([key, value]) => (
            <tr key={key} className="border-b border-border/50 hover:bg-surface-hover/50 transition-colors">
              <td className="whitespace-normal px-3 py-2 text-xs text-muted">{key}</td>
              <td className="px-3 py-2 font-mono text-xs">{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function System() {
  const { data: health, isLoading: hLoading } = useHealth();
  const { data: status } = useSystemStatus();
  const {
    data: config,
    isError: configIsError,
    error: configError,
    refetch: refetchConfig,
  } = useConfig();
  const core = useCoreConfig();

  return (
    <div className="space-y-6 p-4 sm:p-6">
      <h1 className="text-2xl font-bold text-white">System</h1>

      {/* Health */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatCard
          label="API Status"
          value={
            hLoading ? (
              "..."
            ) : (
              <span
                className={cn(
                  health?.status === "running" ? "text-accent" : "text-loss",
                )}
              >
                {health?.status ?? "unknown"}
              </span>
            )
          }
        />
        <StatCard label="Version" value={health?.version ?? "—"} />
        <StatCard
          label="Bot Running"
          value={
            status?.bot_running ? (
              <span className="text-accent">Yes</span>
            ) : (
              <span className="text-muted">No</span>
            )
          }
        />
        <StatCard
          label="Uptime"
          value={
            status?.uptime_seconds != null
              ? status.uptime_seconds >= 3600
                ? `${(status.uptime_seconds / 3600).toFixed(1)}h`
                : `${Math.floor(status.uptime_seconds / 60)}m`
              : "—"
          }
        />
      </div>

      {/* The core: the product */}
      <div className="rounded-xl border border-border bg-surface p-4">
        <h3 className="mb-4 text-sm font-medium uppercase tracking-wider text-muted">
          Configuration · core portfolio
        </h3>
        {core.data ? (
          <ConfigTable
            rows={Object.entries(core.data).map(([k, v]) => {
              const [label, fmt] = CORE_LABELS[k] ?? [k, (x: Value) => String(x)];
              return [label, fmt(v)];
            })}
          />
        ) : core.isError ? (
          <ErrorState compact error={core.error} onRetry={core.refetch} />
        ) : (
          <p className="text-sm text-muted">Loading configuration…</p>
        )}
      </div>

      {/* The day-trader and the LLM */}
      <div className="rounded-xl border border-border bg-surface p-4">
        <h3 className="mb-4 text-sm font-medium uppercase tracking-wider text-muted">
          Configuration · day-trader (retired) &amp; LLM
        </h3>
        {config ? (
          <ConfigTable
            rows={Object.entries(config).map(([key, value]) => [
              key,
              Array.isArray(value) ? value.join(", ") : String(value),
            ])}
          />
        ) : configIsError ? (
          <ErrorState compact error={configError} onRetry={refetchConfig} />
        ) : (
          <p className="text-sm text-muted">Loading configuration…</p>
        )}
      </div>
    </div>
  );
}
