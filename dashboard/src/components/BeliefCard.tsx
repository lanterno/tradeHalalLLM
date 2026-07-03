import { Calendar, ShieldCheck, ShieldQuestion } from "lucide-react";
import type { Belief } from "../api/types";
import { cn, relativeTime } from "../lib/utils";

function directionBadge(direction: string) {
  const label = direction.replace("_", " ");
  const style =
    direction === "long_bias"
      ? "bg-accent/15 text-accent"
      : direction === "short_bias"
        ? "bg-loss/15 text-loss"
        : "bg-muted/15 text-muted";
  return (
    <span className={cn("rounded-full px-2 py-0.5 text-xs font-medium", style)}>
      {label}
    </span>
  );
}

function level(label: string, value: number | null) {
  return (
    <div>
      <div className="text-xs text-muted">{label}</div>
      <div className="text-sm text-white">
        {value != null ? `$${value.toFixed(2)}` : "—"}
      </div>
    </div>
  );
}

export function BeliefCard({ belief }: { belief: Belief }) {
  const conviction = Math.max(0, Math.min(1, belief.conviction));
  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold text-white">{belief.asset}</span>
          {directionBadge(belief.direction)}
          {belief.halal === "halal" ? (
            <ShieldCheck className="h-4 w-4 text-accent" aria-label="halal" />
          ) : (
            <ShieldQuestion className="h-4 w-4 text-warning" aria-label="unverified" />
          )}
        </div>
        <span className="text-xs text-muted">
          {belief.regime.replace("_", " ")} · v{belief.version}
        </span>
      </div>

      <div className="mt-3">
        <div className="flex justify-between text-xs text-muted">
          <span>conviction</span>
          <span>{(conviction * 100).toFixed(0)}%</span>
        </div>
        <div className="mt-1 h-1.5 rounded-full bg-border">
          <div
            className={cn(
              "h-1.5 rounded-full",
              conviction >= 0.55 ? "bg-accent" : "bg-warning",
            )}
            style={{ width: `${conviction * 100}%` }}
          />
        </div>
      </div>

      {belief.thesis && (
        <p className="mt-3 text-sm text-gray-300">{belief.thesis}</p>
      )}

      <div className="mt-3 grid grid-cols-4 gap-2">
        {level("support", belief.support)}
        {level("resistance", belief.resistance)}
        {level("stop", belief.stop)}
        {level("invalidation", belief.invalidation)}
      </div>

      {belief.top_evidence.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {belief.top_evidence.map((e) => (
            <span
              key={`${e.source}-${e.direction}`}
              className={cn(
                "rounded px-1.5 py-0.5 text-xs",
                e.direction >= 0 ? "bg-accent/10 text-accent" : "bg-loss/10 text-loss",
              )}
              title={e.detail}
            >
              {e.source} {e.direction >= 0 ? "+" : ""}
              {e.direction.toFixed(2)}
            </span>
          ))}
        </div>
      )}

      {belief.catalysts_pending.length > 0 && (
        <div className="mt-3 space-y-1">
          {belief.catalysts_pending.map((c) => (
            <div
              key={`${c.kind}-${c.scheduled_for}`}
              className="flex items-center gap-2 text-xs text-warning"
            >
              <Calendar className="h-3.5 w-3.5" />
              <span className="font-medium">{c.kind}</span>
              <span className="text-muted">
                {new Date(c.scheduled_for).toLocaleString(undefined, {
                  month: "short",
                  day: "numeric",
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </span>
              <span className="text-muted">impact {(c.expected_impact * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      )}

      {belief.last_updated && (
        <div className="mt-3 text-right text-xs text-muted">
          updated {relativeTime(belief.last_updated)}
        </div>
      )}
    </div>
  );
}
