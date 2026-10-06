import { StatCard } from "../components/StatCard";
import { ErrorState } from "../components/ErrorState";
import { usePurification } from "../hooks/useInsights";
import { formatUsd } from "../lib/utils";

const ACCOUNT: Record<string, string> = { core: "Core", paper: "Day-trader" };

export default function Insights() {
  const {
    data: purification,
    isError: purificationIsError,
    error: purificationError,
    refetch: purificationRefetch,
  } = usePurification();

  const accounts = purification ? Object.entries(purification.by_account ?? {}) : [];

  return (
    <div className="space-y-6 p-4 sm:p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Insights</h1>
        <p className="text-xs text-muted">Auto-refresh every 5 min</p>
      </div>

      {/* Purification still owed: the dividend ledger, both accounts */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-4">
        {purificationIsError ? (
          <ErrorState compact error={purificationError} onRetry={purificationRefetch} />
        ) : purification ? (
          <StatCard
            label="Purification due"
            value={
              <span className={purification.total_usd > 0 ? "text-warning" : "text-accent"}>
                {formatUsd(purification.total_usd)}
              </span>
            }
            sub={
              [
                ...accounts.map(([a, v]) => `${ACCOUNT[a] ?? a} ${formatUsd(v)}`),
                `${formatUsd(purification.disbursed_total_usd)} given so far`,
              ].join(" · ")
            }
          />
        ) : (
          <StatCard
            label="Purification due"
            value={<span className="text-accent">{formatUsd(0)}</span>}
            sub="No dividend has accrued purification yet"
          />
        )}
      </div>
    </div>
  );
}
