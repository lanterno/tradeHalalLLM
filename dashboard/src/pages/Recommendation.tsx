import {
  useStockOfTheDay,
  useRecommendationHistory,
  useRecommendationScorecard,
} from "../hooks/useRecommendation";
import { RecommendationCard } from "../components/RecommendationCard";
import { StatCard } from "../components/StatCard";
import { ErrorState } from "../components/ErrorState";
import { formatUsd, formatPct } from "../lib/utils";

function usd(v?: number | null) {
  return typeof v === "number" ? formatUsd(v) : "—";
}

function pct(v?: number | null) {
  return typeof v === "number" ? `${v >= 0 ? "+" : ""}${v.toFixed(2)}%` : "—";
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

  return (
    <div className="space-y-6 p-6">
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

      {/* Honest track record (forward returns vs halal benchmark) */}
      {sc?.available && sc.sufficient === false ? (
        <div className="rounded-xl border border-border bg-surface p-3 text-xs text-muted">
          ⚠ Track record is a thin sample ({sc.n_scored} scored picks) — needs
          ≥{sc.min_samples ?? 20} for the rates below to be trustworthy.
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
                  <tr key={r.id} className="border-t border-border align-top">
                    <td className="py-2 pr-4 text-muted whitespace-nowrap">
                      {r.date}
                    </td>
                    <td className="py-2 pr-4 font-semibold text-accent">
                      {r.symbol}
                    </td>
                    <td className="py-2 pr-4">{formatPct(r.conviction ?? 0)}</td>
                    <td className="py-2 pr-4">{usd(r.suggested_entry)}</td>
                    <td className="py-2 pr-4 text-accent">
                      {usd(r.suggested_target)}
                    </td>
                    <td className="py-2 pr-4 text-loss">{usd(r.suggested_stop)}</td>
                    <td className={`py-2 pr-4 ${pctColor(r.fwd_return_5d)}`}>
                      {pct(r.fwd_return_5d)}
                    </td>
                    <td className="max-w-md py-2 text-muted">{r.thesis}</td>
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
