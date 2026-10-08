import { MARKET_TZ, formatDay, formatDuration } from "../../lib/utils";

/** How long before ``now``: "65 s ago", "23 min ago", "Mon 5 Oct" past a day. */
export function ago(iso: string | null | undefined, now: string): string {
  if (!iso) return "never";
  const ms = new Date(now).getTime() - new Date(iso).getTime();
  if (ms < 0) return "just now";
  if (ms < 90_000) return `${Math.round(ms / 1000)} s ago`;
  if (ms < 86_400_000) return `${formatDuration(ms)} ago`;
  return formatDay(iso);
}

/** How long until ``iso``: "in 3h 7m". */
export function until(iso: string | null | undefined, now: string): string {
  if (!iso) return "";
  const ms = new Date(iso).getTime() - new Date(now).getTime();
  return ms <= 0 ? "now" : `in ${formatDuration(ms)}`;
}

const ET_PARTS = new Intl.DateTimeFormat("en-US", {
  timeZone: MARKET_TZ,
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});

/** Minutes since midnight in New York. */
export function etMinutes(iso: string): number {
  const parts = ET_PARTS.formatToParts(new Date(iso));
  const h = Number(parts.find((p) => p.type === "hour")?.value ?? 0);
  const m = Number(parts.find((p) => p.type === "minute")?.value ?? 0);
  return h * 60 + m;
}

const ET_DAY = new Intl.DateTimeFormat("en-CA", { timeZone: MARKET_TZ });

/** The New York day of an instant, "2026-10-08". */
export function etDay(iso: string): string {
  return ET_DAY.format(new Date(iso));
}

/** "today 09:05", "Wed 09:05" or "Mon 5 Oct" relative to ``now``, in New York. */
export function when(iso: string | null | undefined, now: string): string {
  if (!iso) return "—";
  const clock = ET_PARTS.format(new Date(iso));
  const days = Math.round(
    (new Date(`${etDay(iso)}T12:00:00Z`).getTime() - new Date(`${etDay(now)}T12:00:00Z`).getTime()) /
      86_400_000,
  );
  if (days === 0) return `today ${clock}`;
  if (days === -1) return `yesterday ${clock}`;
  if (days === 1) return `tomorrow ${clock}`;
  if (Math.abs(days) < 7) {
    const wd = new Intl.DateTimeFormat("en-GB", { timeZone: MARKET_TZ, weekday: "short" });
    return `${wd.format(new Date(iso))} ${clock}`;
  }
  return formatDay(iso);
}

export const usd = (v: number | null | undefined, digits = 2) =>
  v == null ? "—" : `$${v.toFixed(digits)}`;
