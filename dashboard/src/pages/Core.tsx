import { useCore } from "../hooks/useCore";
import { ErrorState } from "../components/ErrorState";
import { PageSkeleton } from "../components/PageSkeleton";
import { CoreHeader, TopStrip } from "../components/core/TopStrip";
import { GatePanel } from "../components/core/GatePanel";
import { PerformancePanel } from "../components/core/PerformancePanel";
import { HoldingsPanel } from "../components/core/HoldingsPanel";
import { ExecutionPanel, RunsPanel } from "../components/core/RunsPanel";

export default function Core() {
  const { data, isLoading, isError, error, refetch } = useCore();

  if (isError) {
    return (
      <div className="p-4 sm:p-6">
        <ErrorState error={error} onRetry={refetch} />
      </div>
    );
  }
  if (isLoading || !data) return <PageSkeleton />;

  return (
    <div className="space-y-4 p-4 sm:p-6">
      <CoreHeader data={data} />
      <TopStrip data={data} />
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <GatePanel data={data} />
        <PerformancePanel data={data} />
      </div>
      <HoldingsPanel data={data} />
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr_340px]">
        <RunsPanel data={data} />
        {data.execution ? (
          <ExecutionPanel e={data.execution} />
        ) : (
          <div className="rounded-xl border border-border bg-surface p-4 text-sm text-muted">
            Execution quality appears after the core's first orders fill.
          </div>
        )}
      </div>
    </div>
  );
}
