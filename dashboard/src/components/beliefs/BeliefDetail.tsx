// One name in full: the evidence balance, its price among its levels, and
// the shadow's position in it with its recent decisions. The desktop table
// opens it under a row; the phone opens it in a bottom sheet.
import type { Belief, ShadowDecision } from "../../api/types";
import { useAssetDecisions } from "../../hooks/useBeliefs";
import { cn, formatClock } from "../../lib/utils";
import { HORIZON, pct, pctFrom, signedPct, usd } from "./format";
import { Ladder } from "./Marks";

function Label({ children, note }: { children: React.ReactNode; note?: React.ReactNode }) {
  return (
    <p className="mb-2.5 text-[11px] font-semibold uppercase tracking-wider text-muted">
      {children}
      {note && <span className="font-normal normal-case tracking-normal"> · {note}</span>}
    </p>
  );
}

// A full bar (half the track) is a direction × weight of 0.6, the strongest
// evidence the engine usually holds.
const FULL = 0.6;

function Evidence({ b }: { b: Belief }) {
  const items = [...b.top_evidence].sort((x, y) => y.direction * y.weight - x.direction * x.weight);
  const title = b.direction === "long_bias" ? "Why it leans long" : "What the evidence says";
  return (
    <div className="min-w-0">
      <Label note={`${b.n_evidence} pieces of evidence, strongest ${items.length}`}>{title}</Label>
      {items.length === 0 ? (
        <p className="text-xs text-muted">No evidence yet.</p>
      ) : (
        <>
          <div className="grid grid-cols-[104px_64px] gap-2.5 text-[10px] text-muted sm:grid-cols-[110px_120px]">
            <span />
            <span className="flex justify-between">
              <span>against</span>
              <span>for</span>
            </span>
          </div>
          {items.map((e, i) => {
            const v = e.direction * e.weight;
            const w = Math.min(50, (Math.abs(v) / FULL) * 50);
            return (
              <div
                key={`${e.source}-${i}`}
                className="grid grid-cols-[104px_64px_minmax(0,1fr)] items-center gap-2.5 py-1 text-xs sm:grid-cols-[110px_120px_minmax(0,1fr)]"
              >
                <span className="truncate text-gray-300">{e.label}</span>
                <span className="relative h-2">
                  <span className="absolute -top-[3px] -bottom-[3px] left-1/2 w-px bg-gray-700" />
                  <span
                    className={cn("absolute top-0 h-2 rounded-sm", v >= 0 ? "left-1/2 bg-accent" : "right-1/2 bg-loss")}
                    style={{ width: `${w}%` }}
                  />
                </span>
                <span className="text-muted" title={e.detail}>
                  {e.plain}
                </span>
              </div>
            );
          })}
        </>
      )}
      {b.caution && (
        <p className="mt-2.5 rounded-lg border border-warning/20 bg-warning/5 px-2.5 py-1.5 text-[11px] text-warning">
          {b.caution}
        </p>
      )}
    </div>
  );
}

function Fact({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-3">
      <span className="text-muted">{k}</span>
      <span className="text-right tabular-nums">{v}</span>
    </div>
  );
}

function Levels({ b }: { b: Belief }) {
  const p = b.price;
  const away = (level: number | null) => (level == null ? "—" : p == null ? usd(level) : `${pctFrom(level, p)} away`);
  const sameStop = b.stop != null && b.stop === b.invalidation;
  return (
    <div className="min-w-0">
      <Label note={b.price_at ? `bar to ${formatClock(b.price_at, { et: true, withDay: true })}` : "no recent bar"}>
        Price and levels
      </Label>
      {p == null && b.stop == null && b.support == null && b.resistance == null ? (
        <p className="text-xs text-muted">No price or levels yet.</p>
      ) : (
        <div className="px-6">
          <Ladder b={b} big />
        </div>
      )}
      <div className="mt-1 grid gap-1.5 text-xs">
        {sameStop ? (
          <Fact k="Stop is also its invalidation" v={away(b.stop)} />
        ) : (
          <>
            <Fact k="Stop" v={away(b.stop)} />
            <Fact k="Invalidation" v={away(b.invalidation)} />
          </>
        )}
        <Fact
          k="Resistance"
          v={b.resistance == null ? "none found above" : `${usd(b.resistance)}${p != null ? ` (${pctFrom(b.resistance, p)})` : ""}`}
        />
        <Fact k="Horizon" v={HORIZON[b.horizon] ?? b.horizon} />
      </div>
    </div>
  );
}

function Tile({ k, v, tone }: { k: string; v: string; tone?: string }) {
  return (
    <div className="rounded-lg border border-border bg-bg px-2.5 py-1.5">
      <div className="text-[11px] text-muted">{k}</div>
      <b className={cn("text-sm tabular-nums", tone ?? "text-white")}>{v}</b>
    </div>
  );
}

function DecisionLine({ d }: { d: ShadowDecision }) {
  const p = d.payload;
  const side = String(p.side ?? "");
  const delta = typeof p.weight_delta === "number" ? p.weight_delta : null;
  const target = typeof p.target_weight === "number" ? p.target_weight : null;
  return (
    <li
      className={cn(
        "grid grid-cols-[40px_30px_112px_minmax(0,1fr)] gap-2 border-t border-border py-1 text-xs",
        d.outside_session && "bg-warning/[0.04]",
      )}
    >
      <span className="tabular-nums text-muted" title={formatClock(d.ts, { et: true, withDay: true })}>
        {formatClock(d.ts)}
      </span>
      <span className={side === "buy" ? "text-accent" : "text-loss"}>{side}</span>
      <span className="tabular-nums">
        {delta != null ? signedPct(delta, 1) : "—"} → {target != null ? (target > 0 ? pct(target) : "0") : "—"}
      </span>
      <span className={d.outside_session ? "text-warning" : "text-muted"}>
        {d.plain}
        {d.outside_session && " · market shut: no fill possible"}
      </span>
    </li>
  );
}

function ShadowAndDecisions({ b }: { b: Belief }) {
  const { data, isLoading, isError } = useAssetDecisions(b.asset, 5);
  const s = b.shadow;
  return (
    <div className="min-w-0">
      <Label>Shadow position and decisions</Label>
      <div className="grid gap-x-6 gap-y-3 lg:grid-cols-[360px_minmax(0,1fr)]">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:self-start">
          {s ? (
            <>
              <Tile k="weight" v={pct(s.weight)} />
              <Tile k="avg entry" v={usd(s.entry_price)} />
              <Tile k="since entry" v={signedPct(s.return_pct)} tone={s.return_pct >= 0 ? "text-accent" : "text-loss"} />
            </>
          ) : (
            <div className="col-span-2 rounded-lg border border-border bg-bg px-2.5 py-1.5 text-xs text-muted sm:col-span-3">
              The shadow holds none.
            </div>
          )}
          <Tile k="in core" v={b.core_weight ? pct(b.core_weight) : "—"} />
        </div>
        <div className="min-w-0">
          {isError ? (
            <p className="text-xs text-loss">Decisions failed to load.</p>
          ) : isLoading ? (
            <p className="text-xs text-muted">Loading decisions…</p>
          ) : !data?.length ? (
            <p className="text-xs text-muted">No decisions on {b.asset} yet.</p>
          ) : (
            <ul className="[&>li:first-child]:border-t-0">
              {data.map((d) => (
                <DecisionLine key={d.id} d={d} />
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

/** The detail of one belief; ``stacked`` puts every block in one column (the sheet). */
export function BeliefDetail({ b, stacked }: { b: Belief; stacked?: boolean }) {
  return (
    <div className={cn("grid gap-x-7 gap-y-6", !stacked && "lg:grid-cols-[1.4fr_1fr]")}>
      <Evidence b={b} />
      <Levels b={b} />
      <div className={cn(!stacked && "lg:col-span-2")}>
        <ShadowAndDecisions b={b} />
      </div>
    </div>
  );
}
