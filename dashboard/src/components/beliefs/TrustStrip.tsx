// "Should I believe any of this?": the track record against random entry,
// whether conviction is calibrated, and what the shadow's book holds.
import type { BeliefOverview } from "../../api/types";
import { cn } from "../../lib/utils";
import { etDay, pct, signedPct } from "./format";

const VERDICT_TONE: Record<BeliefOverview["verdict"]["status"], string> = {
  unproven: "text-orange-400",
  beating: "text-accent",
  not_beating: "text-loss",
};

function Tile({ label, children, className }: { label: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={cn("min-w-0 border-border px-4 py-3.5", className)}>
      <p className="text-[11px] font-semibold uppercase tracking-wider text-muted">{label}</p>
      {children}
    </div>
  );
}

const Big = ({ children, className }: { children: React.ReactNode; className?: string }) => (
  <p className={cn("my-0.5 text-[22px] font-bold text-white tabular-nums", className)}>{children}</p>
);

const Sub = ({ children }: { children: React.ReactNode }) => (
  <span className="ml-1.5 text-[13px] font-normal text-muted">{children}</span>
);

function verdictText(o: BeliefOverview): React.ReactNode {
  const { cohort, verdict } = o;
  const started = cohort.started ? etDay(cohort.started) : null;
  const weeks = Math.round(verdict.min_days / 7);
  if (verdict.status === "unproven") {
    return (
      <>
        {started ? (
          <>
            A fresh results cohort started <b className="text-white">{started}</b>, after the replay and
            market-hours fixes.
          </>
        ) : (
          <>A fresh results cohort starts with the shadow's next trade, after the replay and market-hours fixes.</>
        )}{" "}
        Older results are discarded: most were artifacts. Judged after {verdict.min_closed} closed trades and{" "}
        {weeks} weeks of regular-hours trading.
      </>
    );
  }
  const c = cohort.current;
  return (
    <>
      {c.closed} closed trades since {started}: {pct(c.win_rate ?? 0, 0)} won, against{" "}
      {pct(o.baseline_win_rate, 0)} for random entry.
      {c.mean_return_pct != null && <> Mean trade {signedPct(c.mean_return_pct)}.</>}
    </>
  );
}

export function TrustStrip({ o }: { o: BeliefOverview }) {
  const cur = o.cohort.current;
  const old = o.cohort.earlier;
  const book = o.book;
  const cal = o.calibration;
  return (
    <div className="grid grid-cols-1 overflow-hidden rounded-xl border border-border bg-surface sm:grid-cols-2 xl:grid-cols-[1.5fr_1fr_1fr_1fr]">
      <Tile label="Track record verdict" className="border-b sm:border-r xl:border-b-0">
        <Big className={VERDICT_TONE[o.verdict.status]}>{o.verdict.label}</Big>
        <p className="text-xs text-muted">{verdictText(o)}</p>
      </Tile>

      <Tile label={`Win rate · cohort ${o.cohort.cohort}`} className="border-b xl:border-r xl:border-b-0">
        <Big>
          {cur.win_rate != null ? pct(cur.win_rate, 0) : "—"}
          <Sub>{cur.closed.toLocaleString()} closed</Sub>
        </Big>
        <div className="relative mt-2 mb-1 h-1.5 rounded-sm bg-surface-hover" title={o.baseline_note}>
          {old.win_rate != null && (
            <div className="absolute inset-y-0 left-0 rounded-sm bg-loss/40" style={{ width: pct(old.win_rate, 2) }} />
          )}
          {cur.win_rate != null && (
            <div
              className={cn(
                "absolute inset-y-0 left-0 rounded-sm",
                cur.win_rate > o.baseline_win_rate ? "bg-accent" : "bg-loss",
              )}
              style={{ width: pct(cur.win_rate, 2) }}
            />
          )}
          <div className="absolute -top-1 h-3.5 w-0.5 bg-white" style={{ left: pct(o.baseline_win_rate, 2) }} />
        </div>
        <p className="text-xs text-muted">
          bar = random entry wins <b className="text-white">{pct(o.baseline_win_rate, 0)}</b>
          {!o.baseline_measured && <span title={o.baseline_note}> (review figure)</span>}
          <br />
          {old.closed > 0 && old.win_rate != null
            ? `earlier outcomes: ${pct(old.win_rate, 0)} (${old.closed.toLocaleString()} trades, not comparable)`
            : "no earlier outcomes"}
        </p>
      </Tile>

      <Tile label="Conviction calibration" className="border-b sm:border-r sm:border-b-0">
        <Big>{cal.status === "fitted" ? "On" : cal.status === "identity" ? "Off" : "Unknown"}</Big>
        <p className="text-xs text-muted">
          {cal.status === "fitted"
            ? "Fitted on closed trades: conviction reads as a chance of a win."
            : cal.status === "identity"
              ? `Too few clean outcomes to fit (${cal.samples} of ${cal.min_samples}); conviction is the engine's raw score, not a probability.`
              : "Nothing scored in the last 24 hours to tell from."}
        </p>
      </Tile>

      <Tile label="Shadow book">
        <Big>
          {pct(book.invested)}
          <Sub>invested</Sub>
        </Big>
        <p className="text-xs text-muted tabular-nums">
          {book.positions.length} position{book.positions.length === 1 ? "" : "s"}
          {book.return_pct != null && (
            <>
              {" · "}
              <span className={book.return_pct >= 0 ? "text-accent" : "text-loss"}>{signedPct(book.return_pct)}</span>{" "}
              since entries
            </>
          )}
          {o.llm.per_day_usd != null && ` · cost $${o.llm.per_day_usd.toFixed(2)}/day LLM`}
        </p>
      </Tile>
    </div>
  );
}

/** The phone's pinned one-liner: the verdict and what it is judged against. */
export function VerdictBanner({ o }: { o: BeliefOverview }) {
  const started = o.cohort.started ? `fresh cohort from ${etDay(o.cohort.started)}` : "fresh cohort";
  return (
    <div
      className={cn(
        "flex items-center gap-2.5 rounded-xl border px-3 py-2 text-xs",
        o.verdict.status === "unproven"
          ? "border-orange-400/25 bg-orange-400/5"
          : o.verdict.status === "beating"
            ? "border-accent/25 bg-accent/5"
            : "border-loss/25 bg-loss/5",
      )}
    >
      <b className={cn("shrink-0 font-semibold", VERDICT_TONE[o.verdict.status])}>{o.verdict.label}</b>
      <span className="text-muted">
        {started} · {o.cohort.current.closed} closed · bar: random entry {pct(o.baseline_win_rate, 0)}
      </span>
    </div>
  );
}
