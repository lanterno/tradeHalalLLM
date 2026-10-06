import { clsx, type ClassValue } from "clsx";

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

export function formatUsd(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

export function formatPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatQty(value: number, decimals = 6): string {
  // Trim trailing zeros so whole share counts render as "44" (not
  // "44.000000") while fractional shares keep their precision.
  return parseFloat(value.toFixed(decimals)).toString();
}

/** Every time on the dashboard is the market's: New York, labelled "ET". */
export const MARKET_TZ = "America/New_York";

const ET_DATETIME = new Intl.DateTimeFormat("en-US", {
  timeZone: MARKET_TZ,
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});

const ET_DATETIME_SECONDS = new Intl.DateTimeFormat("en-US", {
  timeZone: MARKET_TZ,
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hourCycle: "h23",
});

/** An instant as "Oct 2, 15:46 ET": New York time whatever the browser's zone. */
export function formatTime(
  iso: string | null | undefined,
  opts: { seconds?: boolean } = {},
): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return `${(opts.seconds ? ET_DATETIME_SECONDS : ET_DATETIME).format(d)} ET`;
}

const ET_CLOCK = new Intl.DateTimeFormat("en-US", {
  timeZone: MARKET_TZ,
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});

/**
 * The time of day in New York, "12:16": for instants already known to be
 * today (a stream of today's decisions). ``withDay`` prefixes the date when
 * the instant is not today in New York ("Oct 5, 15:46"), and ``et`` appends
 * the zone label.
 */
export function formatClock(
  iso: string | null | undefined,
  opts: { et?: boolean; withDay?: boolean; now?: Date } = {},
): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const suffix = opts.et ? " ET" : "";
  if (opts.withDay && todayET(d) !== todayET(opts.now)) return `${ET_DATETIME.format(d)}${suffix}`;
  return `${ET_CLOCK.format(d)}${suffix}`;
}

/** Today's date in New York, "YYYY-MM-DD": the market's day, not the browser's. */
export function todayET(now: Date = new Date()): string {
  return now.toLocaleDateString("en-CA", { timeZone: MARKET_TZ });
}

/** A span in its two largest units: "3d 15h", "2h 5m", "40s". */
export function formatDuration(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  const d = Math.floor(s / 86_400);
  const h = Math.floor((s % 86_400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (d) return h ? `${d}d ${h}h` : `${d}d`;
  if (h) return m ? `${h}h ${m}m` : `${h}h`;
  if (m) return `${m}m`;
  return `${s}s`;
}

const DATE_ONLY = /^\d{4}-\d{2}-\d{2}$/;

/** A calendar date: "2026-10-04" stays the 4th in every timezone. */
export function parseDay(iso: string): Date {
  // new Date("2026-10-04") is UTC midnight, the 3rd west of Greenwich; a bare
  // date names a day, not an instant, so read it as local noon.
  return DATE_ONLY.test(iso) ? new Date(`${iso}T12:00:00`) : new Date(iso);
}

export function formatDate(iso: string): string {
  if (!iso) return "";
  return parseDay(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

export function pnlColor(value: number): string {
  if (value > 0) return "text-accent";
  if (value < 0) return "text-loss";
  return "text-muted";
}

export function relativeTime(iso: string): string {
  if (!iso) return "";
  const diff = Date.now() - new Date(iso).getTime();
  if (diff < 60_000) return "just now";
  return `${formatDuration(diff)} ago`;
}
