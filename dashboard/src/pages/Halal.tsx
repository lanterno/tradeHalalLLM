import { useHalalCompliance } from "../hooks/useHalal";
import { StatCard } from "../components/StatCard";
import { ErrorState } from "../components/ErrorState";
import { formatUsd } from "../lib/utils";
import { CheckCircle2, AlertTriangle, XCircle } from "lucide-react";

const STATUS: Record<
  string,
  { label: string; cls: string; Icon: typeof CheckCircle2 }
> = {
  compliant: { label: "Compliant", cls: "text-accent", Icon: CheckCircle2 },
  attention: { label: "Needs attention", cls: "text-warning", Icon: AlertTriangle },
  violation: { label: "Violation", cls: "text-loss", Icon: XCircle },
};

export default function Halal() {
  const { data, isLoading, isError, error, refetch } = useHalalCompliance();

  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Halal Compliance</h1>
        <p className="mt-1 text-xs text-muted">
          AAOIFI screening + purification summary. Long-only, no
          interest/leverage/derivatives — non-negotiable.
        </p>
      </div>

      {isError ? (
        <ErrorState error={error} onRetry={refetch} />
      ) : isLoading ? (
        <p className="py-8 text-center text-sm text-muted">Loading…</p>
      ) : !data ? null : (
        <>
          {/* Status banner */}
          {(() => {
            const s = STATUS[data.status] ?? {
              label: data.status,
              cls: "text-muted",
              Icon: AlertTriangle,
            };
            return (
              <div className="flex items-center gap-3 rounded-xl border border-border bg-surface p-4">
                <s.Icon className={`h-7 w-7 ${s.cls}`} aria-hidden />
                <div>
                  <p className={`text-lg font-bold ${s.cls}`}>{s.label}</p>
                  <p className="text-xs text-muted">
                    Quarter to date · {data.trades_this_quarter} trades screened
                  </p>
                </div>
              </div>
            );
          })()}

          {/* Trade volume */}
          <section>
            <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-muted">
              Trade Volume
            </h2>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
              <StatCard label="Today" value={data.trades_today} />
              <StatCard label="This month" value={data.trades_this_month} />
              <StatCard label="This quarter" value={data.trades_this_quarter} />
            </div>
          </section>

          {/* Screening breakdown */}
          <section>
            <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-muted">
              Screenings (quarter)
            </h2>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <StatCard
                label="Halal"
                value={<span className="text-accent">{data.halal_screenings_quarter}</span>}
              />
              <StatCard
                label="Doubtful"
                value={<span className="text-warning">{data.doubtful_screenings_quarter}</span>}
              />
              <StatCard
                label="Not halal"
                value={<span className="text-loss">{data.not_halal_screenings_quarter}</span>}
              />
              <StatCard
                label="Non-halal fills"
                value={
                  <span className={data.non_halal_fills_quarter ? "text-loss" : "text-accent"}>
                    {data.non_halal_fills_quarter}
                  </span>
                }
                sub="must be 0"
              />
            </div>
            {data.halal_screenings_quarter === 0 &&
              data.not_halal_screenings_quarter === 0 && (
                <p className="mt-3 rounded-lg border border-border/60 bg-bg/40 px-3 py-2 text-xs text-muted">
                  No per-symbol screenings recorded this quarter. Screening runs
                  in Zoya <span className="text-warning">sandbox</span> (randomized
                  verdicts) or off the seeded AAOIFI-20 list — a real per-symbol
                  screen needs a paid Zoya production key.
                </p>
              )}
          </section>

          {/* Purification ledger */}
          <section>
            <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-muted">
              Purification
            </h2>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
              <StatCard label="Accrued" value={formatUsd(data.purification_accrued_usd)} />
              <StatCard label="Disbursed" value={formatUsd(data.purification_disbursed_usd)} />
              <StatCard
                label="Outstanding"
                value={
                  <span className={data.purification_outstanding_usd > 0 ? "text-warning" : "text-accent"}>
                    {formatUsd(data.purification_outstanding_usd)}
                  </span>
                }
              />
            </div>
          </section>
        </>
      )}
    </div>
  );
}
