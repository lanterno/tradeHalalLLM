import { clsx, type ClassValue } from "clsx";

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

const USD = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

// One sign convention everywhere: "+", a typographic minus, no sign on zero.
function sign(v: number, shown: string): string {
  if (Number(shown.replace(/[^0-9.]/g, "")) === 0) return "";
  return v >= 0 ? "+" : "−";
}

/** Dollars to the cent, "$1,234.50"; "—" when missing. ``signed``: "+$12.00" / "−$3.10". */
export function formatUsd(v: number | null | undefined, opts: { signed?: boolean } = {}): string {
  if (v == null) return "—";
  if (!opts.signed) return USD.format(v);
  const shown = USD.format(Math.abs(v));
  return `${sign(v, shown)}${shown}`;
}

/** A fraction as a percent, "12.3%" (``digits`` decimals); "—" when missing.
 *  ``signed``: "+2.24%" / "−0.38%". */
export function formatPct(
  v: number | null | undefined,
  digits = 1,
  opts: { signed?: boolean } = {},
): string {
  if (v == null) return "—";
  const shown = `${(Math.abs(v) * 100).toFixed(digits)}%`;
  if (opts.signed) return `${sign(v, shown)}${shown}`;
  return v < 0 && Number(shown.slice(0, -1)) !== 0 ? `−${shown}` : shown;
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
export function formatDay(iso: string): string {
  return DAY.format(new Date(iso.length === 10 ? `${iso}T12:00:00-04:00` : iso));
}

/** "Thu 8 Oct, 15:40" in New York. */
export function formatDayTime(iso: string): string {
  return DAY_TIME.format(new Date(iso));
}

/** A size in bytes, in its largest unit: "9.6 GB", "434 MB", "12 kB". */
export function formatBytes(n: number | null | undefined): string {
  if (n == null) return "—";
  const units = ["B", "kB", "MB", "GB", "TB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v >= 100 || i === 0 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`;
}
