// Shared wording and number formats for the Belief Board.
import type { Belief, Stance, StrictVerdict } from "../../api/types";
import { MARKET_TZ } from "../../lib/utils";

/** A share as "62%" (d=0) or "10.7%". */
export const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;

/** A signed change, typographic minus: "+2.24%", "−0.38%". */
export function signedPct(v: number, d = 2): string {
  const s = (Math.abs(v) * 100).toFixed(d);
  if (Number(s) === 0) return `${(0).toFixed(d)}%`;
  return `${v >= 0 ? "+" : "−"}${s}%`;
}

/** The New York day of an instant: "6 Oct", or "Wed 14 Oct" with the weekday. */
export function etDay(iso: string, weekday = false): string {
  return new Date(iso).toLocaleDateString("en-GB", {
    weekday: weekday ? "short" : undefined,
    day: "numeric",
    month: "short",
    timeZone: MARKET_TZ,
  });
}

export function usd(v: number): string {
  return `$${v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export const STANCE: Record<Stance, { label: string; short: string; pill: string }> = {
  long: { label: "Long", short: "Long", pill: "bg-accent/10 text-accent" },
  leaning: { label: "Leaning long", short: "Leaning", pill: "bg-lime-400/10 text-lime-400" },
  none: { label: "No view", short: "No view", pill: "bg-muted/10 text-muted" },
  excluded: { label: "Excluded", short: "Excluded", pill: "bg-loss/10 text-loss" },
  benchmark: { label: "Benchmark", short: "Benchmark", pill: "bg-blue-400/10 text-blue-400" },
};

/** The filters, in the order the toolbar shows them (the benchmark is only in "All"). */
export const FILTERS: (Stance | "all")[] = ["all", "long", "leaning", "none", "excluded"];

export const REGIME: Record<string, { label: string; tone: string }> = {
  trending_up: { label: "Uptrend", tone: "text-accent" },
  ranging: { label: "Range", tone: "text-muted" },
  trending_down: { label: "Downtrend", tone: "text-loss" },
};

export function regimeOf(b: Belief) {
  return REGIME[b.regime] ?? { label: b.regime.replace(/_/g, " "), tone: "text-muted" };
}

export const STRICT_TAG: Record<StrictVerdict, string> = {
  halal: "halal",
  not_halal: "not halal",
  doubtful: "doubtful",
  unscreened: "unscreened",
};

/** Does the name fail the strict screen (the benchmark aside)? */
export function failsStrict(b: Pick<Belief, "stance" | "strict">): boolean {
  return b.stance === "excluded" && b.strict !== "halal";
}

/** A level's distance from the price, one decimal: "−1.0%", "+4.5%". */
export function pctFrom(level: number, price: number): string {
  const v = level / price - 1;
  return `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(1)}%`;
}

/** "−1.0%": how far the price may fall before the stop. */
export function stopDistance(b: Belief): string | null {
  if (b.stop == null || b.price == null) return null;
  return pctFrom(b.stop, b.price);
}

export const HORIZON: Record<string, string> = {
  intraday: "intraday (hours)",
  swing: "swing (days)",
  position: "position (weeks)",
};
