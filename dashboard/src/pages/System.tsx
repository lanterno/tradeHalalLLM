import { useHealth, useSystemStatus, useConfig } from "../hooks/useSystem";
import { StatCard } from "../components/StatCard";
import { cn } from "../lib/utils";

export default function System() {
  const { data: health, isLoading: hLoading } = useHealth();
  const { data: status } = useSystemStatus();
  const { data: config } = useConfig();

  return (
    <div className="space-y-6 p-6">
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

      {/* WebSocket health */}
      {status?.ws_health && Object.keys(status.ws_health).length > 0 && (
        <div className="rounded-xl border border-border bg-surface p-4">
          <h3 className="mb-4 text-sm font-medium uppercase tracking-wider text-muted">
            WebSocket Health
          </h3>
          <div className="grid gap-2 md:grid-cols-3 lg:grid-cols-4">
            {Object.entries(status.ws_health).map(([symbol, info]) => {
              const stale = typeof info === "object" && info !== null && "stale" in info
                ? (info as Record<string, unknown>).stale
                : false;
              return (
                <div
                  key={symbol}
                  className="flex items-center justify-between rounded-lg border border-border/50 px-3 py-2"
                >
                  <span className="text-sm font-medium">{symbol}</span>
                  <span
                    className={cn(
                      "h-2 w-2 rounded-full",
                      stale ? "bg-loss" : "bg-accent",
                    )}
                  />
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Configuration */}
      <div className="rounded-xl border border-border bg-surface p-4">
        <h3 className="mb-4 text-sm font-medium uppercase tracking-wider text-muted">
          Configuration
        </h3>
        {config ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                  <th className="px-3 py-2">Setting</th>
                  <th className="px-3 py-2">Value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(config).map(([key, value]) => (
                  <tr
                    key={key}
                    className="border-b border-border/50 hover:bg-surface-hover/50 transition-colors"
                  >
                    <td className="px-3 py-2 font-mono text-xs text-muted">
                      {key}
                    </td>
                    <td className="px-3 py-2 font-mono text-xs">
                      {Array.isArray(value) ? value.join(", ") : String(value)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-sm text-muted">Loading configuration...</p>
        )}
      </div>
    </div>
  );
}
