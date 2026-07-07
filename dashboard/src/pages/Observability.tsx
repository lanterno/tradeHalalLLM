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

function formatCost(value: number): string {
  // Costs are tiny (~$0.05/day) — show enough precision to be meaningful.
  return `$${value.toFixed(value < 10 ? 4 : 2)}`;
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
          <StatCard label="Completed cycles" value={cycles.data?.count ?? "—"} />
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

        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-5">
          <StatCard label="Total calls" value={llm.data?.calls ?? "—"} />
          <StatCard
            label="Total tokens"
            value={llm.data ? formatTokens(llm.data.total_tokens) : "—"}
            sub={llm.data?.total_tokens ? `${llm.data.total_tokens.toLocaleString()}` : undefined}
          />
          <StatCard
            label="Total cost"
            value={
              <span className="text-accent">
                {llm.data ? formatCost(llm.data.total_cost_usd) : "—"}
              </span>
            }
            sub="GLM via OpenRouter"
          />
          <StatCard label="p50 latency" value={formatMs(llm.data?.p50_ms)} />
          <StatCard label="p95 latency" value={formatMs(llm.data?.p95_ms)} />
        </div>
      </section>
    </div>
  );
}
