import { MARKET_TZ } from "../../lib/utils";

/** Sector colours, shared by the strip, the legend and the table's group dots. */
export const SECTOR_COLORS: Record<string, string> = {
  "Software & internet": "#4ade80",
  "Computer hardware": "#60a5fa",
  Semiconductors: "#facc15",
  Healthcare: "#c084fc",
  Industrials: "#fb923c",
  Autos: "#94a3b8",
  "Consumer goods": "#f472b6",
  "Materials & mining": "#2dd4bf",
  Retail: "#a3e635",
  Energy: "#f87171",
  "Real estate": "#818cf8",
  Transport: "#fbbf24",
  Utilities: "#38bdf8",
  "Communications & media": "#e879f9",
  Financials: "#fde68a",
};
export const sectorColor = (sector: string) => SECTOR_COLORS[sector] ?? "#4b5563";

/** A signed change in percentage points: "+0.03 pts". */
export function pts(v: number | null | undefined): string {
  if (v == null) return "—";
  const s = Math.abs(v * 100).toFixed(2);
  return Number(s) === 0 ? "0.00 pts" : `${v >= 0 ? "+" : "−"}${s} pts`;
}

/** Basis points, signed: a positive figure is a cost. */
export function bps(v: number | null | undefined): string {
  if (v == null) return "—";
  return `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(1)} bps`;
}

export function toneOf(v: number | null | undefined): string {
  if (v == null || v === 0) return "text-muted";
  return v > 0 ? "text-accent" : "text-loss";
}

/** A cost in bps: green under the book's assumption, amber over it. */
export function costTone(v: number | null | undefined, budget: number): string {
  if (v == null) return "text-muted";
  return v <= budget ? "text-accent" : "text-warning";
}


const DAY = new Intl.DateTimeFormat("en-GB", {
  timeZone: MARKET_TZ,
  weekday: "short",
  day: "numeric",
  month: "short",
});
const DAY_TIME = new Intl.DateTimeFormat("en-GB", {
  timeZone: MARKET_TZ,
  weekday: "short",
  day: "numeric",
  month: "short",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});

/** "Thu 8 Oct" for a calendar day (an ISO date, read as that day). */
export function day(iso: string): string {
  return DAY.format(new Date(iso.length === 10 ? `${iso}T12:00:00-04:00` : iso));
}

/** "Thu 8 Oct, 15:40" in New York. */
export function dayTime(iso: string): string {
  return DAY_TIME.format(new Date(iso));
}
