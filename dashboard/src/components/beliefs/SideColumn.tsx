// The side column: the shadow's book, the macro calendar once, and today's
// decisions with their reasons.
import { useState } from "react";
import type { BeliefOverview, ShadowDecision } from "../../api/types";
import { useShadowDecisions } from "../../hooks/useBeliefs";
import { cn, formatClock, todayET } from "../../lib/utils";
import { etDay, pct, signedPct } from "./format";

function Card({ title, note, children }: { title: string; note?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="min-w-0 rounded-xl border border-border bg-surface p-4">
      <h3 className="mb-2.5 flex items-baseline justify-between gap-2 text-[11px] font-semibold uppercase tracking-wider text-muted">
        {title}
        {note && <span className="text-xs font-normal normal-case tracking-normal">{note}</span>}
      </h3>
      {children}
    </section>
  );
}

export function ShadowBookCard({ book }: { book: BeliefOverview["book"] }) {
  const top = Math.max(...book.positions.map((p) => p.weight), 0.0001);
  const failing = book.positions.filter((p) => p.strict !== "halal");
  return (
    <Card title="Shadow book" note="since entry">
      {book.positions.length === 0 ? (
        <p className="text-xs text-muted">The shadow holds nothing in this cohort yet: all cash.</p>
      ) : (
        <table className="w-full text-xs tabular-nums">
          <tbody>
            {book.positions.map((p) => (
              <tr key={p.asset} className="border-b border-[#15151f]">
                <td className="py-1.5 pr-1 text-white">
                  {p.asset}
                  {p.strict !== "halal" && <span className="ml-1 text-[10px] text-loss">strict ✕</span>}
                </td>
                <td className="w-[72px] py-1.5">
                  <div className="h-1 rounded-sm bg-purple-400/70" style={{ width: `${(p.weight / top) * 64}px` }} />
                </td>
                <td className="py-1.5 text-right">{pct(p.weight)}</td>
                <td className={cn("py-1.5 pl-2 text-right", p.return_pct >= 0 ? "text-accent" : "text-loss")}>
                  {signedPct(p.return_pct)}
                </td>
              </tr>
            ))}
            <tr>
              <td className="py-1.5 text-muted">cash</td>
              <td />
              <td className="py-1.5 text-right">{pct(book.cash)}</td>
              <td />
            </tr>
          </tbody>
        </table>
      )}
      {failing.length > 0 && (
        <p className="mt-2.5 rounded-lg border border-warning/20 bg-warning/5 px-2.5 py-1.5 text-[11px] text-warning">
          {failing.length} holding{failing.length === 1 ? "" : "s"} fail{failing.length === 1 ? "s" : ""} the strict
          halal screen. The engine opens strict-halal names only.
        </p>
      )}
    </Card>
  );
}

export function MacroCalendarCard({ events }: { events: BeliefOverview["calendar"] }) {
  return (
    <Card title="Macro calendar" note="every name">
      {events.length === 0 ? (
        <p className="text-xs text-muted">No scheduled releases ahead.</p>
      ) : (
        <ul>
          {events.map((e, i) => (
            <li
              key={`${e.kind}-${e.scheduled_for}`}
              className={cn(
                "grid grid-cols-[78px_minmax(0,1fr)_auto] items-center gap-2 py-1.5 text-xs",
                i > 0 && "border-t border-border",
              )}
            >
              <span className="tabular-nums">
                {etDay(e.scheduled_for, true)}
                <br />
                <span className="text-muted">{formatClock(e.scheduled_for, { et: true })}</span>
              </span>
              <span className="min-w-0">
                <b className="text-white">{e.kind}</b>
                <br />
                <span className="text-[11px] text-muted">{e.plain}</span>
              </span>
              <span
                className="relative inline-block h-1 w-9 rounded-sm bg-surface-hover"
                title={`expected impact ${pct(e.expected_impact, 0)}`}
              >
                <i className="absolute inset-y-0 left-0 rounded-sm bg-warning" style={{ width: pct(e.expected_impact, 0) }} />
              </span>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

function StreamLine({ d, withDay }: { d: ShadowDecision; withDay: boolean }) {
  const p = d.payload;
  const side = String(p.side ?? "");
  const delta = typeof p.weight_delta === "number" ? p.weight_delta : null;
  const target = typeof p.target_weight === "number" ? p.target_weight : null;
  return (
    <li className="grid grid-cols-[40px_48px_32px_minmax(0,1fr)] gap-1.5 border-t border-border py-1.5 text-xs first:border-t-0">
      <span className="tabular-nums text-muted" title={formatClock(d.ts, { et: true, withDay: true })}>
        {withDay ? etDay(d.ts) : formatClock(d.ts)}
      </span>
      <b className="text-white">{d.asset ?? "—"}</b>
      <span className={side === "buy" ? "text-accent" : "text-loss"}>{side}</span>
      <span className={cn("tabular-nums", d.outside_session ? "text-warning" : "text-muted")}>
        {delta != null ? signedPct(delta, 1) : "—"} → {target != null ? (target > 0 ? pct(target) : "0") : "—"} ·{" "}
        {d.plain}
        {d.outside_session && " · market shut"}
      </span>
    </li>
  );
}

export function DecisionsCard() {
  const [all, setAll] = useState(false);
  const { data, isError, isLoading } = useShadowDecisions(50);
  const today = todayET();
  const todays = (data ?? []).filter((d) => todayET(new Date(d.ts)) === today);
  // Today's ten newest; "all" opens the last fifty, whatever their day.
  const shown = all ? (data ?? []) : todays.slice(0, 10);
  const more = all ? 0 : todays.length - shown.length;
  return (
    <Card
      title={all ? "Recent decisions" : "Decisions today"}
      note={
        <button type="button" onClick={() => setAll(!all)} className="text-blue-400 hover:text-white">
          {all ? "today" : "all"}
        </button>
      }
    >
      {isError ? (
        <p className="text-xs text-loss">Decisions failed to load.</p>
      ) : isLoading ? (
        <p className="text-xs text-muted">Loading…</p>
      ) : shown.length === 0 ? (
        <p className="text-xs text-muted">
          None today
          {data?.length ? `; the last was ${formatClock(data[0].ts, { et: true, withDay: true })}` : ""}.
        </p>
      ) : (
        <ul>
          {shown.map((d) => (
            <StreamLine key={d.id} d={d} withDay={all && todayET(new Date(d.ts)) !== today} />
          ))}
          {more > 0 && (
            <li className="border-t border-border pt-1.5 text-xs text-muted">
              {more} earlier today ·{" "}
              <button type="button" onClick={() => setAll(true)} className="text-blue-400 hover:text-white">
                show
              </button>
            </li>
          )}
        </ul>
      )}
    </Card>
  );
}
