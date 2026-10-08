import { useOperations } from "../hooks/useOperations";
import { ErrorState } from "../components/ErrorState";
import { PageSkeleton } from "../components/PageSkeleton";
import { OpsHeader, OpsStrip } from "../components/operations/TopStrip";
import { JobsPanel } from "../components/operations/JobsPanel";
import { FreshnessPanel, ProcessesPanel } from "../components/operations/ProcessesPanel";
import { LlmPanel } from "../components/operations/LlmPanel";
import { ConfigPanel, DatabasePanel } from "../components/operations/DatabasePanel";

export default function Operations() {
  const { data, isLoading, isError, error, refetch } = useOperations();

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
      <OpsHeader data={data} paper={data.paper} />
      <OpsStrip data={data} />
      <JobsPanel data={data} />
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <ProcessesPanel data={data} />
        <FreshnessPanel data={data} />
      </div>
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <LlmPanel data={data} />
        <DatabasePanel data={data} />
      </div>
      <ConfigPanel data={data} />
    </div>
  );
}
