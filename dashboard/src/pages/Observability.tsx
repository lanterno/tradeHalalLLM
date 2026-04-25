import { useState } from "react";
import { useCycleMetrics, useLlmMetrics } from "../hooks/useMetrics";
import { StatCard } from "../components/StatCard";
import { cn } from "../lib/utils";

const CYCLE_WINDOWS = [
  { label: "1h", seconds: 3600 },
  { label: "6h", seconds: 21600 },
  { label: "24h", seconds: 86400 },
] as const;

const LLM_WINDOWS = [
  { label: "24h", seconds: 86400 },
  { label: "7d", seconds: 604800 },
  { label: "30d", seconds: 2592000 },
] as const;

function formatMs(value: number | null | undefined): string {
  if (value == null) return "—";
  if (value < 1000) return `${value.toFixed(0)} ms`;
  return `${(value / 1000).toFixed(2)} s`;
}

function formatTokens(value: number): string {
  if (value < 1_000) return value.toString();
  if (value < 1_000_000) return `${(value / 1_000).toFixed(1)}k`;
  return `${(value / 1_000_000).toFixed(2)}M`;
}

export default function Observability() {
  const [cycleWindow, setCycleWindow] = useState<number>(CYCLE_WINDOWS[0].seconds);
  const [llmWindow, setLlmWindow] = useState<number>(LLM_WINDOWS[0].seconds);

  const cycles = useCycleMetrics(cycleWindow);
  const llm = useLlmMetrics(llmWindow);

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Observability</h1>
        <p className="text-xs text-muted">
          Derived from <span className="font-mono">logs/halal_trader.log</span>; refreshes every 30s.
        </p>
      </div>

      {/* Cycle metrics */}
      <section>
        <header className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-medium uppercase tracking-wider text-muted">
            Cycle Latency
          </h2>
          <div className="flex gap-1">
            {CYCLE_WINDOWS.map(({ label, seconds }) => (
              <button
                key={seconds}
                onClick={() => setCycleWindow(seconds)}
                className={cn(
                  "rounded-md px-3 py-1 text-xs font-medium transition-colors",
                  cycleWindow === seconds
                    ? "bg-accent/10 text-accent"
                    : "text-muted hover:text-white",
                )}
              >
                {label}
              </button>
            ))}
          </div>
        </header>

        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <StatCard
            label="Completed cycles"
            value={cycles.data?.count ?? "—"}
            sub={cycles.isLoading ? "Loading..." : undefined}
          />
          <StatCard label="p50" value={formatMs(cycles.data?.p50_ms)} />
          <StatCard label="p95" value={formatMs(cycles.data?.p95_ms)} />
          <StatCard label="p99" value={formatMs(cycles.data?.p99_ms)} />
        </div>

        <div className="mt-3 grid grid-cols-2 gap-4">
          <StatCard
            label="Failed"
            value={
              <span className={cn(cycles.data?.failed ? "text-loss" : "text-accent")}>
                {cycles.data?.failed ?? 0}
              </span>
            }
            sub="cycle.failed"
          />
          <StatCard
            label="Halted"
            value={
              <span className={cn(cycles.data?.halted ? "text-warning" : "")}>
                {cycles.data?.halted ?? 0}
              </span>
            }
            sub="cycle.halted (kill-switch / loss limit)"
          />
        </div>
      </section>

      {/* LLM metrics */}
      <section>
        <header className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-medium uppercase tracking-wider text-muted">
            LLM Calls + Cost
          </h2>
          <div className="flex gap-1">
            {LLM_WINDOWS.map(({ label, seconds }) => (
              <button
                key={seconds}
                onClick={() => setLlmWindow(seconds)}
                className={cn(
                  "rounded-md px-3 py-1 text-xs font-medium transition-colors",
                  llmWindow === seconds
                    ? "bg-accent/10 text-accent"
                    : "text-muted hover:text-white",
                )}
              >
                {label}
              </button>
            ))}
          </div>
        </header>

        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <StatCard label="Total calls" value={llm.data?.calls ?? "—"} />
          <StatCard
            label="Total tokens"
            value={llm.data ? formatTokens(llm.data.total_tokens) : "—"}
            sub={llm.data?.total_tokens ? `${llm.data.total_tokens.toLocaleString()}` : undefined}
          />
          <StatCard label="p50 latency" value={formatMs(llm.data?.p50_ms)} />
          <StatCard label="p95 latency" value={formatMs(llm.data?.p95_ms)} />
        </div>

        {llm.data && Object.keys(llm.data.by_provider).length > 0 && (
          <div className="mt-4 rounded-xl border border-border bg-surface p-4">
            <h3 className="mb-3 text-xs font-medium uppercase tracking-wider text-muted">
              By Provider
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                    <th className="px-3 py-2">Provider</th>
                    <th className="px-3 py-2 text-right">Calls</th>
                    <th className="px-3 py-2 text-right">Tokens</th>
                    <th className="px-3 py-2 text-right">p50</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(llm.data.by_provider).map(([provider, stats]) => (
                    <tr
                      key={provider}
                      className="border-b border-border/50 hover:bg-surface-hover/50 transition-colors"
                    >
                      <td className="px-3 py-2 font-mono">{provider}</td>
                      <td className="px-3 py-2 text-right">{stats.calls}</td>
                      <td className="px-3 py-2 text-right">{formatTokens(stats.tokens)}</td>
                      <td className="px-3 py-2 text-right">{formatMs(stats.p50_ms)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
