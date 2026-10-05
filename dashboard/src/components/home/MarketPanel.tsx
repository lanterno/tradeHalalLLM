import { useState } from "react";
import type { HomeStatus } from "../../api/types";
import { useNow } from "../../hooks/useNow";
import { cn } from "../../lib/utils";
import { formatCountdown, readClock, SESSION_BAR, type MarketState } from "../../lib/marketClock";

const STATES: Record<MarketState, { label: string; dot: string; live: boolean }> = {
  open: { label: "Market open", dot: "bg-accent text-accent", live: true },
  pre: { label: "Pre-market", dot: "bg-blue-400 text-blue-400", live: true },
  post: { label: "After-hours", dot: "bg-blue-400 text-blue-400", live: true },
  closed: { label: "Market closed", dot: "bg-muted text-muted", live: false },
  weekend: { label: "Closed for the weekend", dot: "bg-muted text-muted", live: false },
  holiday: { label: "Closed: market holiday", dot: "bg-warning text-warning", live: false },
};

const span = SESSION_BAR.end - SESSION_BAR.start;
const at = (minutes: number) => `${((minutes - SESSION_BAR.start) / span) * 100}%`;
const hhmm = (m: number) => `${Math.floor(m / 60)}:${String(m % 60).padStart(2, "0")}`;

function localTime(target: Date): string {
  return target.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/** The reader's local time for a New York wall-clock minute today. */
function localOf(nyMinutes: number, now: Date, nowNyMinutes: number): string {
  return localTime(new Date(now.getTime() + (nyMinutes - nowNyMinutes) * 60_000));
}

function shortDay(iso: string): string {
  return new Date(`${iso}T12:00:00`).toLocaleDateString("en-US", {
    weekday: "short",
    day: "numeric",
    month: "short",
  });
}

function benchmarkNote(benchmarks: HomeStatus["market"]["benchmarks"], open: boolean): string {
  const first = benchmarks[0];
  if (!first) return "";
  if (open && first.live) return "Live, refreshed every minute · change vs the previous close";
  if (!first.as_of.includes("T")) return `Close of ${shortDay(first.as_of)}`;
  const at = new Date(first.as_of);
  const time = at.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: "America/New_York" });
  const day = at.toLocaleDateString("en-US", {
    weekday: "short",
    day: "numeric",
    month: "short",
    timeZone: "America/New_York",
  });
  return `Last price ${day}, ${time} ET · change vs the previous close`;
}

export function MarketPanel({ market }: { market: HomeStatus["market"] }) {
  const now = useNow(1000);
  const [tz, setTz] = useState<"et" | "local">("et");
  const clock = readClock(now, market);
  const s = STATES[clock.state];
  const minute = Math.floor(clock.minutes);
  const label = (m: number) => (tz === "et" ? hhmm(m) : localOf(m, now, minute));
  const early = clock.closeMinutes !== SESSION_BAR.close;

  const whenTarget =
    clock.towards === "close"
      ? `at ${hhmm(clock.closeMinutes)} ET · ${localTime(clock.target)} your time${early ? " · early close today" : ""}`
      : `${clock.target.toLocaleDateString([], { weekday: "long" })} 9:30 ET · ${localTime(clock.target)} your time`;

  return (
    <section className="rounded-xl border border-border bg-surface p-4">
      <div className="grid grid-cols-1 items-center gap-5 lg:grid-cols-[1.15fr_1fr]">
        <div>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className={cn("relative h-2.5 w-2.5 rounded-full", s.dot)}>
              {s.live && (
                <span className="absolute -inset-1 animate-ping rounded-full border-2 border-current opacity-50" />
              )}
            </span>
            <b className="whitespace-nowrap text-[15px] text-white">{s.label}</b>
            <span className="basis-full text-xs text-muted sm:basis-auto">
              NYSE &amp; Nasdaq · regular session 9:30–16:00 ET
            </span>
          </div>
          <p className="mt-2 font-bold tabular-nums text-white">
            <span className="mr-2 text-sm font-medium text-muted">
              {clock.towards === "close" ? "Closes in" : "Opens in"}
            </span>
            <span className="text-4xl">{formatCountdown(clock.secondsLeft)}</span>
          </p>
          <p className="text-xs text-muted">{whenTarget}</p>

          <div className={cn("relative mt-5 h-14", !clock.tradingDay && "opacity-40")}>
            <div className="absolute inset-x-0 top-[18px] h-2 overflow-hidden rounded bg-border">
              <div className="absolute inset-y-0 bg-blue-400/20" style={{ left: 0, width: at(SESSION_BAR.open) }} />
              <div
                className="absolute inset-y-0 bg-accent/35"
                style={{ left: at(SESSION_BAR.open), width: `calc(${at(clock.closeMinutes)} - ${at(SESSION_BAR.open)})` }}
              />
              <div className="absolute inset-y-0 right-0 bg-blue-400/20" style={{ left: at(clock.closeMinutes) }} />
            </div>
            <div className="absolute top-0 -translate-x-1/2 whitespace-nowrap text-[10px] text-blue-400" style={{ left: at(940) }}>
              <span className="hidden sm:inline">Core trades </span>
              {label(940)}
              <div className="mx-auto mt-0.5 h-2 w-px bg-blue-400" />
            </div>
            {clock.tradingDay && clock.minutes >= SESSION_BAR.start && clock.minutes <= SESSION_BAR.end && (
              <div
                className="absolute top-2.5 h-6 w-0.5 rounded bg-white shadow-[0_0_0_3px_rgba(255,255,255,0.13)]"
                style={{ left: at(clock.minutes) }}
                aria-label="now"
              />
            )}
            {[SESSION_BAR.start, SESSION_BAR.open, clock.closeMinutes, SESSION_BAR.end].map((m, i) => (
              <div
                key={m}
                className={cn(
                  "absolute top-[30px] whitespace-nowrap text-[10px] text-muted",
                  i === 0 ? "" : i === 3 ? "-translate-x-full text-right" : "-translate-x-1/2 text-center",
                )}
                style={{ left: at(m) }}
              >
                {label(m)}
                <br />
                <span className="hidden sm:inline">{["pre", "open", "close", "after-hours"][i]}</span>
              </div>
            ))}
          </div>
          <div className="mt-2 flex gap-1.5">
            {(["et", "local"] as const).map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => setTz(t)}
                className={cn(
                  "rounded-md border px-2 py-0.5 text-[11px]",
                  tz === t ? "border-white/20 text-white" : "border-border text-muted",
                )}
              >
                {t === "et" ? "ET" : "Your time"}
              </button>
            ))}
          </div>
        </div>

        <div>
          <div className="grid grid-cols-3 gap-2.5">
            {market.benchmarks.map((b) => (
              <div key={b.symbol} className="rounded-lg border border-border bg-bg p-2.5">
                <p className="text-[11px] text-muted">{b.symbol === "SPY" ? "S&P 500 (SPY)" : b.symbol}</p>
                <p className="text-[17px] font-bold tabular-nums text-white">${b.price.toFixed(2)}</p>
                {b.change_pct !== null && (
                  <p className={cn("text-xs tabular-nums", b.change_pct >= 0 ? "text-accent" : "text-loss")}>
                    {b.change_pct >= 0 ? "+" : ""}
                    {(b.change_pct * 100).toFixed(2)}%
                  </p>
                )}
              </div>
            ))}
          </div>
          <p className="mt-1.5 text-xs text-muted">
            {benchmarkNote(market.benchmarks, clock.state === "open")}
          </p>
          <div className="mt-3 grid gap-1.5 text-xs">
            {market.next_holiday && (
              <div className="flex justify-between gap-2">
                <span className="text-muted">Next holiday</span>
                <span>
                  {market.next_holiday.name}, {shortDay(market.next_holiday.date)}{" "}
                  <span className="text-muted">(closed)</span>
                </span>
              </div>
            )}
            {market.next_early_close && (
              <div className="flex justify-between gap-2">
                <span className="text-muted">Next early close</span>
                <span>{shortDay(market.next_early_close)}, 13:00 ET</span>
              </div>
            )}
            <div className="flex justify-between gap-2">
              <span className="text-muted">Trading days this month</span>
              <span className="tabular-nums">
                {market.trading_days_month}{" "}
                <span className="text-muted">· {market.trading_days_left} after today</span>
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
