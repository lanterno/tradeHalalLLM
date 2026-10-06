// The board's small marks: stance pill, strict tag, conviction bar, level ladder.
import type { Belief, Stance } from "../../api/types";
import { cn } from "../../lib/utils";
import { STANCE, STRICT_TAG, pctFrom, usd } from "./format";

export function StancePill({ stance, className }: { stance: Stance; className?: string }) {
  const s = STANCE[stance];
  return (
    <span
      className={cn(
        "inline-block whitespace-nowrap rounded-full px-2 py-0.5 text-[11px] font-semibold",
        s.pill,
        className,
      )}
    >
      {s.label}
    </span>
  );
}

/** "not halal · strict" under an excluded name, "index" under the benchmark. */
export function StrictTag({ b, compact }: { b: Belief; compact?: boolean }) {
  if (b.stance === "benchmark") {
    return compact ? null : (
      <span className="mt-0.5 inline-block rounded bg-muted/10 px-1.5 text-[10px] text-muted">index</span>
    );
  }
  if (b.stance !== "excluded") return null;
  const text = b.strict === "halal" ? "screen stale" : compact ? "strict ✕" : `${STRICT_TAG[b.strict]} · strict`;
  return (
    <span className="mt-0.5 inline-block whitespace-nowrap rounded bg-loss/10 px-1.5 text-[10px] text-loss">
      {text}
    </span>
  );
}

/** Conviction as a bar, with a tick at the entry band. */
export function ConvictionBar({
  value,
  entry,
  exit,
  className,
}: {
  value: number;
  entry: number;
  exit: number;
  className?: string;
}) {
  const c = Math.max(0, Math.min(1, value));
  return (
    <div className={cn("relative h-[5px] rounded-sm bg-surface-hover", className)}>
      <div
        className={cn(
          "h-full rounded-sm",
          c >= entry ? "bg-accent" : c >= exit ? "bg-lime-400/70" : "bg-gray-600",
        )}
        style={{ width: `${c * 100}%` }}
      />
      <div
        className="absolute -top-[3px] h-[11px] w-px bg-white/55"
        style={{ left: `${entry * 100}%` }}
        aria-hidden
      />
    </div>
  );
}

interface Level {
  key: "stop" | "support" | "resistance";
  value: number;
}

function levelsOf(b: Belief): Level[] {
  const out: Level[] = [];
  if (b.stop != null) out.push({ key: "stop", value: b.stop });
  // Support at the stop is one level, drawn (and named) once, as the stop.
  if (b.support != null && b.support !== b.stop) out.push({ key: "support", value: b.support });
  if (b.resistance != null) out.push({ key: "resistance", value: b.resistance });
  return out;
}

const LEVEL_TONE: Record<Level["key"], { mark: string; text: string }> = {
  stop: { mark: "bg-loss", text: "text-loss" },
  support: { mark: "bg-accent/80", text: "text-accent" },
  resistance: { mark: "bg-blue-400", text: "text-blue-400" },
};

/**
 * Where the price sits among its levels: the support-resistance band, the
 * stop, and the price as a dot. ``big`` labels every level with its price
 * and its distance from the price.
 */
export function Ladder({ b, big }: { b: Belief; big?: boolean }) {
  const levels = levelsOf(b);
  const price = b.price;
  const values = [...levels.map((l) => l.value), ...(price != null ? [price] : [])];
  if (!values.length) return null;
  const ref = price ?? values[0];
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  const span = Math.max(hi - lo, ref * 0.004);
  const mid = (lo + hi) / 2;
  lo = Math.min(lo, mid - span / 2) - span * 0.12;
  hi = Math.max(hi, mid + span / 2) + span * 0.12;
  const pos = (v: number) => ((v - lo) / (hi - lo)) * 100;
  const band =
    b.support != null && b.resistance != null
      ? { left: pos(b.support), width: pos(b.resistance) - pos(b.support) }
      : null;

  // Labels under the track, staggered a row down when two would collide.
  const placed: { at: number; row: number }[] = [];
  const rowFor = (at: number) => {
    const row = placed.some((p) => Math.abs(p.at - at) < 16 && p.row === 0) ? 1 : 0;
    placed.push({ at, row });
    return row;
  };

  return (
    <div
      className={cn("relative", big ? "mt-9 mb-1.5" : "my-1 h-3.5")}
      style={big ? { height: 96 } : undefined}
      aria-hidden={!big}
    >
      <div className={cn("absolute inset-x-0 bg-[#1f1f30]", big ? "top-5 h-[3px]" : "top-1.5 h-0.5")} />
      {band && (
        <div
          className={cn("absolute rounded-sm bg-blue-400/15", big ? "top-4 h-[11px]" : "top-1 h-1.5")}
          style={{ left: `${band.left}%`, width: `${band.width}%` }}
        />
      )}
      {levels.map((l) => (
        <div
          key={l.key}
          className={cn(
            "absolute",
            LEVEL_TONE[l.key].mark,
            l.key === "stop"
              ? big
                ? "top-2.5 h-[23px] w-[3px]"
                : "top-px h-3 w-0.5"
              : big
                ? "top-[13px] h-[17px] w-[3px]"
                : "top-[3px] h-2 w-0.5",
          )}
          style={{ left: `${pos(l.value)}%` }}
          title={`${l.key} ${usd(l.value)}`}
        />
      ))}
      {price != null && (
        <div
          className={cn(
            "absolute rounded-full bg-white shadow-[0_0_0_3px_rgb(255_255_255/0.13)]",
            big ? "top-3.5 -ml-[7px] h-3.5 w-3.5" : "top-0.5 -ml-[5px] h-2.5 w-2.5",
          )}
          style={{ left: `${pos(price)}%` }}
        />
      )}
      {big &&
        levels.map((l) => {
          const at = pos(l.value);
          const row = rowFor(at);
          return (
            <div
              key={`label-${l.key}`}
              className={cn(
                "absolute -translate-x-1/2 text-center text-[10px] leading-tight whitespace-nowrap",
                LEVEL_TONE[l.key].text,
              )}
              style={{ left: `${Math.min(92, Math.max(8, at))}%`, top: 38 + row * 30 }}
            >
              {l.key}
              <br />
              <b className="font-semibold tabular-nums text-white">{usd(l.value)}</b>
              {price != null && (
                <>
                  <br />
                  <span className="tabular-nums">{pctFrom(l.value, price)}</span>
                </>
              )}
            </div>
          );
        })}
      {big && price != null && (
        <div
          className="absolute -top-8 -translate-x-1/2 text-center text-[10px] leading-tight whitespace-nowrap text-white"
          style={{ left: `${Math.min(92, Math.max(8, pos(price)))}%` }}
        >
          now
          <br />
          <b className="font-semibold tabular-nums">{usd(price)}</b>
        </div>
      )}
    </div>
  );
}
