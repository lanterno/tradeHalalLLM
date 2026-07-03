import { BeliefCard } from "../components/BeliefCard";
import { useBeliefBoard, useShadowDecisions } from "../hooks/useBeliefs";
import { relativeTime } from "../lib/utils";

function DecisionRow({ d }: { d: import("../api/types").ShadowDecision }) {
  const p = d.payload as Record<string, unknown>;
  const side = String(p.side ?? "");
  const reason = String(p.reason ?? "");
  const delta = typeof p.weight_delta === "number" ? p.weight_delta : null;
  return (
    <tr className="border-t border-border">
      <td className="px-3 py-2 text-xs text-muted">{relativeTime(d.ts)}</td>
      <td className="px-3 py-2 text-sm font-medium text-white">{d.asset ?? "—"}</td>
      <td
        className={
          side === "buy"
            ? "px-3 py-2 text-sm text-accent"
            : "px-3 py-2 text-sm text-loss"
        }
      >
        {side}
        {p.forced_exit === true ? " (forced)" : ""}
      </td>
      <td className="px-3 py-2 text-sm text-gray-300">
        {delta != null ? `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)}%` : "—"}
      </td>
      <td className="max-w-md truncate px-3 py-2 text-xs text-muted" title={reason}>
        {reason}
      </td>
    </tr>
  );
}

export default function BeliefBoard() {
  const board = useBeliefBoard();
  const decisions = useShadowDecisions(30);

  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Belief Board</h1>
        <p className="text-sm text-muted">
          The shadow engine's live market understanding — advisory only, never
          trades.
        </p>
      </div>

      {board.isLoading && (
        <div className="rounded-xl border border-border bg-surface p-6 text-muted">
          Loading beliefs…
        </div>
      )}
      {board.data && !board.data.available && (
        <div className="rounded-xl border border-border bg-surface p-6 text-muted">
          No active beliefs yet — the shadow engine builds them while its
          daemon runs.
        </div>
      )}
      {board.data && board.data.available && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {board.data.beliefs.map((b) => (
            <BeliefCard key={b.asset} belief={b} />
          ))}
        </div>
      )}

      <div className="rounded-xl border border-border bg-surface p-4">
        <h2 className="mb-3 text-lg font-semibold text-white">
          Shadow decision stream
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="text-xs uppercase text-muted">
                <th className="px-3 py-2">when</th>
                <th className="px-3 py-2">asset</th>
                <th className="px-3 py-2">side</th>
                <th className="px-3 py-2">Δ weight</th>
                <th className="px-3 py-2">reason</th>
              </tr>
            </thead>
            <tbody>
              {(decisions.data ?? []).map((d) => (
                <DecisionRow key={d.id} d={d} />
              ))}
            </tbody>
          </table>
          {decisions.data?.length === 0 && (
            <div className="px-3 py-4 text-sm text-muted">
              No shadow proposals recorded yet.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
