import {
  useStockOfTheDay,
  useRecommendationHistory,
  useRecommendationScorecard,
} from "../hooks/useRecommendation";
import { useMemo, useState } from "react";
import { RecommendationCard } from "../components/RecommendationCard";
import { StatCard } from "../components/StatCard";
import { ErrorState } from "../components/ErrorState";
import { VerdictBadge } from "../components/PickVerdict";
import { verdictSummary } from "../lib/scorecard";
import { cn, formatPct, formatUsd } from "../lib/utils";

/** A history row's thesis: two lines, the rest on a click. */
function Thesis({ text }: { text?: string }) {
  const [open, setOpen] = useState(false);
  if (!text) return <>—</>;
  return (
    <button
      type="button"
      onClick={() => setOpen(!open)}
      aria-expanded={open}
      title={open ? "Collapse" : "Show the whole thesis"}
      className={cn("min-w-48 text-left hover:text-white", !open && "line-clamp-2")}
    >
      {text}
    </button>
  );
}

/** The API reports these moves in percent already (2.5 is 2.5%). */
function pct(v?: number | null) {
  return formatPct(v == null ? v : v / 100, 2, { signed: true });
}

function pctColor(v?: number | null) {
  if (typeof v !== "number") return "text-white";
  return v >= 0 ? "text-accent" : "text-loss";
}

export default function Recommendation() {
  const {
    data: pick,
    isLoading,
    isError: pickIsError,
    error: pickError,
    refetch: pickRefetch,
  } = useStockOfTheDay();
  const {
    data: history,
    isError: historyIsError,
    error: historyError,
    refetch: historyRefetch,
  } = useRecommendationHistory(30);
  const {
    data: sc,
    isError: scIsError,
    error: scError,
    refetch: scRefetch,
  } = useRecommendationScorecard();

  // A day generated more than once: the newest row is the day's pick (what the
  // scorecard scores); the older ones are shown dimmed.
  const superseded = useMemo(() => {
    const newest = new Map<string, number>();
    for (const r of history ?? []) {
      if (r.date && r.id != null && r.id > (newest.get(r.date) ?? -1)) newest.set(r.date, r.id);
    }
    return new Set(
      (history ?? [])
        .filter((r) => r.date && r.id != null && newest.get(r.date) !== r.id)
        .map((r) => r.id as number),
    );
  }, [history]);

  return (
    <div className="space-y-6 p-4 sm:p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Stock of the Day</h1>
        <p className="text-xs text-muted">Advisory halal recommendation</p>
      </div>

      {pickIsError ? (
        <div className="rounded-xl border border-border bg-surface p-4">
          <ErrorState compact error={pickError} onRetry={pickRefetch} />
        </div>
      ) : (
        <RecommendationCard pick={pick} isLoading={isLoading} />
      )}

      {scIsError ? (
        <div className="rounded-xl border border-border bg-surface p-3">
          <ErrorState compact error={scError} onRetry={scRefetch} />
        </div>
      ) : null}

      {/* The verdict first: the record decides whether the pick is worth reading. */}
      {sc ? (
        <div
          className={cn(
            "flex flex-wrap items-center gap-x-3 gap-y-1.5 rounded-xl border bg-surface p-3 text-xs text-muted",
            sc.verdict === "negative" ? "border-loss/30" : "border-border",
          )}
        >
          <VerdictBadge verdict={sc.verdict} />
          <span>{verdictSummary(sc)}</span>
          {sc.verdict === "negative" && (
            <span className="text-loss">The record argues against acting on the pick.</span>
          )}
          {sc.conviction_mode && sc.conviction_mode.n > 1 && (
            <span className="basis-full">
              Conviction was {sc.conviction_mode.value.toFixed(2)} on {sc.conviction_mode.n} of{" "}
              {sc.conviction_mode.of} days, so it carries little information.
            </span>
          )}
          {sc.n_duplicates ? (
            <span className="basis-full">
              One pick per day: {sc.n_duplicates} re-generated row{sc.n_duplicates === 1 ? "" : "s"} below
              {sc.n_duplicates === 1 ? " is" : " are"} left out of every figure (the newest of the day counts).
            </span>
          ) : null}
        </div>
      ) : null}
      {sc?.available ? (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-5">
          <StatCard
            label="5d Hit Rate"
            value={
              <span className={pctColor((sc.hit_rate_5d ?? 0) - 0.5)}>
                {typeof sc.hit_rate_5d === "number"
                  ? formatPct(sc.hit_rate_5d)
                  : "—"}
              </span>
            }
            sub={`${sc.n_scored} scored picks`}
          />
          <StatCard
            label="Avg 5d Return"
            value={<span className={pctColor(sc.avg_fwd_5d)}>{pct(sc.avg_fwd_5d)}</span>}
          />
          <StatCard
            label={`Excess vs ${sc.benchmark ?? "bench"}`}
            value={
              <span className={pctColor(sc.avg_excess_5d)}>
                {pct(sc.avg_excess_5d)}
              </span>
            }
          />
          <StatCard
            label="Avg 1d / 20d"
            value={
              <span className="text-sm">
                <span className={pctColor(sc.avg_fwd_1d)}>{pct(sc.avg_fwd_1d)}</span>
                {" / "}
                <span className={pctColor(sc.avg_fwd_20d)}>{pct(sc.avg_fwd_20d)}</span>
              </span>
            }
          />
          <StatCard
            label="Best / Worst (5d)"
            value={
              <span className="text-sm">
                <span className="text-accent">{sc.best?.symbol ?? "—"}</span>
                {" / "}
                <span className="text-loss">{sc.worst?.symbol ?? "—"}</span>
              </span>
            }
            sub={
              sc.best
                ? `${pct(sc.best.fwd_5d)} / ${pct(sc.worst?.fwd_5d)}`
                : undefined
            }
          />
        </div>
      ) : null}

      {/* Quant grounding: plan quality (LLM levels vs realized path) +
          band coverage + the counterfactual pick percentile. Populates as
          picks mature; tiles render only where their data exists. */}
      {sc?.available &&
      (sc.n_with_levels || sc.band_n || sc.candidate_band_n || sc.pick_percentile_n) ? (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-5">
          {typeof sc.avg_plan_return_5d === "number" ? (
            <StatCard
              label="Plan Return (5d)"
              value={
                <span className={pctColor(sc.avg_plan_return_5d)}>
                  {pct(sc.avg_plan_return_5d)}
                </span>
              }
              sub="entry@open → bracket"
            />
          ) : null}
          {typeof sc.target_hit_rate === "number" ? (
            <StatCard
              label="Target / Stop Hit"
              value={
                <span className="text-sm">
                  <span className="text-accent">{formatPct(sc.target_hit_rate)}</span>
                  {" / "}
                  <span className="text-loss">
                    {typeof sc.stop_hit_rate === "number"
                      ? formatPct(sc.stop_hit_rate)
                      : "—"}
                  </span>
                </span>
              }
              sub={`${sc.n_with_levels ?? 0} picks w/ levels`}
            />
          ) : null}
          {typeof sc.band_coverage_5d === "number" ? (
            <StatCard
              label="Band Coverage (5d)"
              value={formatPct(sc.band_coverage_5d)}
              sub={`pick n=${sc.band_n ?? 0}`}
            />
          ) : null}
          {typeof sc.candidate_band_coverage_5d === "number" ? (
            <StatCard
              label="Band Cov · all cand."
              value={formatPct(sc.candidate_band_coverage_5d)}
              sub={`n=${sc.candidate_band_n ?? 0}`}
            />
          ) : null}
          {typeof sc.avg_pick_percentile_5d === "number" ? (
            <StatCard
              label="Pick Percentile"
              value={
                <span className={pctColor(sc.avg_pick_percentile_5d - 0.5)}>
                  {formatPct(sc.avg_pick_percentile_5d)}
                </span>
              }
              sub="0.5 = random"
            />
          ) : null}
        </div>
      ) : null}

      <div className="rounded-xl border border-border bg-surface p-4">
        <h2 className="mb-4 text-sm font-medium uppercase tracking-wider text-muted">
          History
        </h2>
        {historyIsError ? (
          <ErrorState compact error={historyError} onRetry={historyRefetch} />
        ) : history && history.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs uppercase tracking-wider text-muted">
                  <th className="pb-2 pr-4">Date</th>
                  <th className="pb-2 pr-4">Symbol</th>
                  <th className="pb-2 pr-4">Conviction</th>
                  <th className="pb-2 pr-4">Entry</th>
                  <th className="pb-2 pr-4">Target</th>
                  <th className="pb-2 pr-4">Stop</th>
                  <th className="pb-2 pr-4">5d</th>
                  <th className="pb-2">Thesis</th>
                </tr>
              </thead>
              <tbody>
                {history.map((r) => (
                  <tr
                    key={r.id}
                    className={cn("border-t border-border align-top", superseded.has(r.id ?? -1) && "opacity-45")}
                  >
                    <td className="py-2 pr-4 text-muted whitespace-nowrap">
                      {r.date}
                      {superseded.has(r.id ?? -1) && (
                        <span className="block text-[10px] uppercase tracking-wide">re-run, not scored</span>
                      )}
                    </td>
                    <td className="py-2 pr-4 font-semibold text-accent">
                      {r.symbol}
                    </td>
                    <td className="py-2 pr-4">{formatPct(r.conviction ?? 0)}</td>
                    <td className="py-2 pr-4">{formatUsd(r.suggested_entry)}</td>
                    <td className="py-2 pr-4 text-accent">
                      {formatUsd(r.suggested_target)}
                    </td>
                    <td className="py-2 pr-4 text-loss">{formatUsd(r.suggested_stop)}</td>
                    <td className={`py-2 pr-4 ${pctColor(r.fwd_return_5d)}`}>
                      {pct(r.fwd_return_5d)}
                    </td>
                    <td className="min-w-64 max-w-md whitespace-normal py-2 text-muted">
                      <Thesis text={r.thesis} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-sm text-muted">No history yet.</p>
        )}
      </div>
    </div>
  );
}
