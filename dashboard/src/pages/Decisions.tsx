import { useState } from "react";
import { useDecisions, useAdjustments } from "../hooks/useDecisions";
import { ErrorState } from "../components/ErrorState";
import { formatTime, relativeTime } from "../lib/utils";

export default function Decisions() {
  const {
    data: decisions,
    isLoading: dLoading,
    isError: dError,
    error: dErr,
    refetch: dRefetch,
  } = useDecisions(50);
  const {
    data: adjustments,
    isLoading: aLoading,
    isError: aError,
    error: aErr,
    refetch: aRefetch,
  } = useAdjustments();
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  const toggle = (id: number) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div className="space-y-6 p-6">
      <h1 className="text-2xl font-bold text-white">LLM Decisions</h1>

      {/* Strategy adjustments */}
      <div className="rounded-xl border border-border bg-surface p-4">
        <h3 className="mb-4 text-sm font-medium uppercase tracking-wider text-muted">
          Strategy Adjustments
        </h3>
        {aError ? (
          <ErrorState compact error={aErr} onRetry={aRefetch} />
        ) : aLoading ? (
          <p className="text-sm text-muted">Loading…</p>
        ) : !adjustments?.length ? (
          <p className="text-sm text-muted">No adjustments yet.</p>
        ) : (
          <div className="space-y-3">
            {adjustments.map((adj) => (
              <div
                key={adj.id}
                className="rounded-lg border border-border/50 p-3"
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <span className="text-sm font-medium text-accent">
                      {adj.parameter}
                    </span>
                    <span className="mx-2 text-muted">→</span>
                    <span className="font-mono text-sm text-loss line-through">
                      {adj.old_value}
                    </span>
                    <span className="mx-1 text-muted">→</span>
                    <span className="font-mono text-sm text-accent">
                      {adj.new_value}
                    </span>
                  </div>
                  <span className="shrink-0 text-xs text-muted">
                    {relativeTime(adj.timestamp)}
                  </span>
                </div>
                {adj.reasoning && (
                  <p className="mt-1 text-xs text-muted">{adj.reasoning}</p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Decision log */}
      <div className="rounded-xl border border-border bg-surface p-4">
        <h3 className="mb-4 text-sm font-medium uppercase tracking-wider text-muted">
          Decision Log
        </h3>
        {dError ? (
          <ErrorState compact error={dErr} onRetry={dRefetch} />
        ) : dLoading ? (
          <p className="text-sm text-muted">Loading…</p>
        ) : !decisions?.length ? (
          <p className="text-sm text-muted">No decisions logged.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                  <th className="px-3 py-2">Time</th>
                  <th className="px-3 py-2">Model</th>
                  <th className="px-3 py-2 text-right">Latency</th>
                  <th className="px-3 py-2">Symbols</th>
                  <th className="px-3 py-2">Actions</th>
                  <th className="px-3 py-2 w-8"></th>
                </tr>
              </thead>
              <tbody>
                {decisions.map((d) => {
                  let parsedAction: unknown = null;
                  if (d.parsed_action) {
                    try {
                      parsedAction = JSON.parse(d.parsed_action);
                    } catch {
                      parsedAction = d.parsed_action;
                    }
                  }
                  let symbols: string[] = [];
                  if (d.symbols) {
                    try {
                      const parsed = JSON.parse(d.symbols);
                      symbols = Array.isArray(parsed) ? parsed : [d.symbols];
                    } catch {
                      symbols = [d.symbols];
                    }
                  }

                  const isExpanded = expanded.has(d.id);

                  return (
                    <tr
                      key={d.id}
                      className="border-b border-border/50 hover:bg-surface-hover/50 transition-colors cursor-pointer"
                      onClick={() => toggle(d.id)}
                    >
                      <td className="whitespace-nowrap px-3 py-2 text-muted">
                        {formatTime(d.timestamp)}
                      </td>
                      <td className="px-3 py-2 font-mono text-xs">{d.model}</td>
                      <td className="px-3 py-2 text-right font-mono">
                        {d.execution_ms ? `${d.execution_ms}ms` : "—"}
                      </td>
                      <td className="px-3 py-2">
                        <div className="flex flex-wrap gap-1">
                          {symbols.map((s, i) => (
                            <span
                              key={i}
                              className="rounded bg-surface-hover px-1.5 py-0.5 text-[10px] font-medium"
                            >
                              {s}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td className="max-w-xs truncate px-3 py-2 text-xs text-muted">
                        {typeof parsedAction === "string"
                          ? parsedAction.slice(0, 60)
                          : JSON.stringify(parsedAction)?.slice(0, 60)}
                      </td>
                      <td className="px-3 py-2 text-muted text-xs">
                        {isExpanded ? "▲" : "▼"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {/* Expanded details rendered separately */}
            {decisions
              .filter((d) => expanded.has(d.id))
              .map((d) => (
                <div
                  key={`exp-${d.id}`}
                  className="border-b border-border/50 bg-surface-hover/30 p-4"
                >
                  <p className="mb-1 text-xs font-medium text-muted">
                    Prompt Summary
                  </p>
                  <p className="mb-3 text-xs text-gray-300 whitespace-pre-wrap">
                    {d.prompt_summary}
                  </p>
                  <p className="mb-1 text-xs font-medium text-muted">
                    Raw Response
                  </p>
                  <pre className="max-h-60 overflow-auto rounded-lg bg-bg p-3 text-[11px] text-gray-300">
                    {d.raw_response}
                  </pre>
                </div>
              ))}
          </div>
        )}
      </div>
    </div>
  );
}
