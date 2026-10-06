// The US equity market's state at a given instant, computed in the browser so
// the countdown ticks every second without asking the server. The rules match
// market_hours.py: NYSE holidays, 13:00 early closes, 9:30-16:00 regular
// session, 4:00-9:30 pre-market and 16:00-20:00 after-hours (New York time).
// The calendar (holidays, early closes) comes from /api/home.

export type MarketState = "open" | "pre" | "post" | "closed" | "weekend" | "holiday";

export interface MarketCalendar {
  holidays: string[]; // ISO dates
  early_closes: string[];
}

export interface ClockReading {
  state: MarketState;
  /** Seconds until the next change worth counting down to (open or close). */
  secondsLeft: number;
  /** "open" when counting down to an open, "close" to a close. */
  towards: "open" | "close";
  /** The instant counted down to. */
  target: Date;
  /** Is today (New York) a trading day? */
  tradingDay: boolean;
  /** Minutes since New York midnight, for the session bar. */
  minutes: number;
  /** Today's close in New York minutes (780 on an early close). */
  closeMinutes: number;
}

const OPEN = 9 * 60 + 30;
const PRE = 4 * 60;
const POST_END = 20 * 60;
const CLOSE = 16 * 60;
const EARLY = 13 * 60;

interface NyParts {
  date: string;
  weekday: number; // 0 = Sunday
  minutes: number;
  seconds: number;
}

const FMT = new Intl.DateTimeFormat("en-CA", {
  timeZone: "America/New_York",
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hourCycle: "h23",
  weekday: "short",
});
const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

function newYork(at: Date): NyParts {
  const p = Object.fromEntries(FMT.formatToParts(at).map((x) => [x.type, x.value]));
  return {
    date: `${p.year}-${p.month}-${p.day}`,
    weekday: WEEKDAYS.indexOf(p.weekday),
    minutes: Number(p.hour) * 60 + Number(p.minute),
    seconds: Number(p.second),
  };
}

function isTradingDay(date: string, weekday: number, cal: MarketCalendar): boolean {
  return weekday !== 0 && weekday !== 6 && !cal.holidays.includes(date);
}

/** The instant of New York wall-clock ``minutes`` on the day ``daysAhead`` from ``at``'s. */
function nyInstant(at: Date, daysAhead: number, minutes: number): Date {
  // Walk from ``at`` to the wanted wall-clock time, then correct for a DST
  // change in between (the offset is re-read at the result).
  const now = newYork(at);
  const guess = new Date(
    at.getTime() + (daysAhead * 1440 + minutes - now.minutes) * 60_000 - now.seconds * 1000,
  );
  const drift = newYork(guess).minutes - minutes;
  return drift === 0 ? guess : new Date(guess.getTime() - drift * 60_000);
}

function nextOpen(at: Date, cal: MarketCalendar): Date {
  for (let i = 0; i < 14; i++) {
    const day = newYork(new Date(at.getTime() + i * 86_400_000));
    if (!isTradingDay(day.date, day.weekday, cal)) continue;
    const open = nyInstant(at, i, OPEN);
    if (open.getTime() > at.getTime()) return open;
  }
  return nyInstant(at, 1, OPEN);
}

export function readClock(at: Date, cal: MarketCalendar): ClockReading {
  const ny = newYork(at);
  const tradingDay = isTradingDay(ny.date, ny.weekday, cal);
  const closeMinutes = cal.early_closes.includes(ny.date) ? EARLY : CLOSE;
  let state: MarketState;
  if (!tradingDay) state = ny.weekday === 0 || ny.weekday === 6 ? "weekend" : "holiday";
  else if (ny.minutes >= OPEN && ny.minutes < closeMinutes) state = "open";
  else if (ny.minutes >= PRE && ny.minutes < OPEN) state = "pre";
  else if (ny.minutes >= closeMinutes && ny.minutes < POST_END) state = "post";
  else state = "closed";

  const towards = state === "open" ? "close" : "open";
  const target = state === "open" ? nyInstant(at, 0, closeMinutes) : nextOpen(at, cal);
  return {
    state,
    secondsLeft: Math.max(0, Math.round((target.getTime() - at.getTime()) / 1000)),
    towards,
    target,
    tradingDay,
    minutes: ny.minutes + ny.seconds / 60,
    closeMinutes,
  };
}

export function formatCountdown(seconds: number): string {
  const d = Math.floor(seconds / 86_400);
  const h = Math.floor((seconds % 86_400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return d ? `${d}d ${pad(h)}:${pad(m)}:${pad(s)}` : `${pad(h)}:${pad(m)}:${pad(s)}`;
}

/** "in 6h 28m", "in 3 days", from now to an ISO instant. */
export function formatIn(iso: string, now: Date): string {
  const minutes = Math.round((new Date(iso).getTime() - now.getTime()) / 60_000);
  if (minutes <= 0) return "now";
  if (minutes < 60) return `in ${minutes}m`;
  if (minutes < 24 * 60) return `in ${Math.floor(minutes / 60)}h ${minutes % 60}m`;
  const days = Math.round(minutes / 1440);
  if (days < 14) return `in ${days} day${days === 1 ? "" : "s"}`;
  return `in ${Math.round(days / 7)} weeks`;
}

export const SESSION_BAR = { start: PRE, end: POST_END, open: OPEN, close: CLOSE };
