import {
  useStockOfTheDay,
  useRecommendationHistory,
} from "../hooks/useRecommendation";
import { RecommendationCard } from "../components/RecommendationCard";
import { formatUsd, formatPct } from "../lib/utils";

function usd(v?: number | null) {
  return typeof v === "number" ? formatUsd(v) : "—";
}

export default function Recommendation() {
  const { data: pick, isLoading } = useStockOfTheDay();
  const { data: history } = useRecommendationHistory(30);

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Stock of the Day</h1>
        <p className="text-xs text-muted">Advisory halal recommendation</p>
      </div>

      <RecommendationCard pick={pick} isLoading={isLoading} />

      <div className="rounded-xl border border-border bg-surface p-4">
        <h2 className="mb-4 text-sm font-medium uppercase tracking-wider text-muted">
          History
        </h2>
        {history && history.length > 0 ? (
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
