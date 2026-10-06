import type { RecommendationScorecard } from "../api/types";

function signed(v: number, digits = 1): string {
  return `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;
}

/** The stock-of-the-day scorecard in one line, so the pick is never read without its record. */
export function verdictSummary(sc: RecommendationScorecard): string {
  if (!sc.available) return "No matured picks yet: nothing to judge it by.";
  const parts: string[] = [];
  if (sc.avg_excess_5d != null) {
    parts.push(`${signed(sc.avg_excess_5d)}% vs ${sc.benchmark ?? "SPUS"} per 5 days`);
  }
  if (sc.conviction_ic != null) parts.push(`conviction IC ${signed(sc.conviction_ic, 2)}`);
  parts.push(`${sc.n_scored} scored pick${sc.n_scored === 1 ? "" : "s"}`);
  if (sc.sufficient === false) parts.push(`under the ${sc.min_samples ?? 20} needed to trust it`);
  return parts.join(" · ");
}
