import { StatCard } from "../components/StatCard";
import { ErrorState } from "../components/ErrorState";
import { usePurification } from "../hooks/useInsights";
import { formatUsd } from "../lib/utils";

function EmptyTile({ label, message }: { label: string; message: string }) {
  return (
    <StatCard
      label={label}
      value={<span className="text-sm font-normal text-muted">—</span>}
      sub={message}
    />
  );
}

export default function Insights() {
  const {
    data: purification,
    isError: purificationIsError,
    error: purificationError,
    refetch: purificationRefetch,
  } = usePurification();

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Insights</h1>
        <p className="text-xs text-muted">Auto-refresh every 5 min</p>
      </div>

      {/* Purification owed on closed winning trades */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-4">
        {purificationIsError ? (
          <ErrorState compact error={purificationError} onRetry={purificationRefetch} />
        ) : purification ? (
          <StatCard
            label="Purification Due"
            value={
              <span className="text-white">{formatUsd(purification.total_usd)}</span>
            }
            sub={`${purification.n_entries} entries · ${formatUsd(purification.disbursed_total_usd)} disbursed`}
          />
        ) : (
          <EmptyTile label="Purification Due" message="No closed wins yet" />
        )}
      </div>

    </div>
  );
}
