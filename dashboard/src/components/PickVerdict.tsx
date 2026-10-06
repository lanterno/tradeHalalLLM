import type { RecommendationScorecard } from "../api/types";
import { cn } from "../lib/utils";

const STYLE = {
  negative: "border-loss/35 bg-loss/5 text-loss",
  unproven: "border-warning/35 bg-warning/5 text-warning",
  positive: "border-accent/35 bg-accent/5 text-accent",
} as const;

const LABEL = {
  negative: "Negative record",
  unproven: "Unproven",
  positive: "Beating the benchmark",
} as const;

/** The stock-of-the-day record in one word (the scorecard's own verdict). */
export function VerdictBadge({ verdict }: { verdict: RecommendationScorecard["verdict"] }) {
  const v = verdict ?? "unproven";
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
        STYLE[v],
      )}
    >
      {LABEL[v]}
    </span>
  );
}
