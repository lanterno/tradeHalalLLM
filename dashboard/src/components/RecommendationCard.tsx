import type { StockOfTheDay } from "../api/types";
import { formatUsd, formatPct } from "../lib/utils";

interface Props {
  pick?: StockOfTheDay;
  isLoading?: boolean;
}

function Level({ label, value, tone }: { label: string; value?: number | null; tone?: string }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-wider text-muted">{label}</p>
      <p className={`font-semibold ${tone ?? "text-white"}`}>
        {typeof value === "number" ? formatUsd(value) : "—"}
      </p>
    </div>
  );
}

export function RecommendationCard({ pick, isLoading }: Props) {
  const header = (
    <h2 className="text-sm font-medium uppercase tracking-wider text-muted">
      📈 Halal Stock of the Day
    </h2>
  );

  if (isLoading) {
    return (
      <div className="h-40 animate-pulse rounded-xl border border-border bg-surface" />
    );
  }

  if (!pick || !pick.available) {
    return (
      <div className="rounded-xl border border-border bg-surface p-5">
        {header}
        <p className="mt-2 text-sm text-muted">
          No recommendation yet — it generates pre-market each trading day.
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border bg-surface p-5">
      <div className="flex items-center justify-between">
        {header}
        <span className="text-xs text-muted">
          {pick.date} · {pick.universe_size ?? 0} candidates
        </span>
      </div>

      <div className="mt-2 flex items-baseline gap-3">
        <span className="text-3xl font-bold text-accent">{pick.symbol}</span>
        <span className="text-sm text-muted">
          {formatPct(pick.conviction ?? 0)} conviction
        </span>
      </div>

      <p className="mt-3 text-sm text-white">{pick.thesis}</p>

      <div className="mt-4 grid grid-cols-3 gap-3 text-sm">
        <Level label="Entry" value={pick.suggested_entry} />
        <Level label="Target" value={pick.suggested_target} tone="text-accent" />
        <Level label="Stop" value={pick.suggested_stop} tone="text-loss" />
      </div>

      <details className="mt-4 text-sm">
        <summary className="cursor-pointer text-xs uppercase tracking-wider text-muted">
          Halal note &amp; details
        </summary>
        <p className="mt-2 text-white">
          <span className="text-muted">Halal: </span>
          {pick.halal_note}
        </p>
        {pick.catalysts && (
          <p className="mt-1 text-white">
            <span className="text-muted">Catalysts: </span>
            {pick.catalysts}
          </p>
        )}
        {pick.risks && (
          <p className="mt-1 text-white">
            <span className="text-muted">Risks: </span>
            {pick.risks}
          </p>
        )}
      </details>

      <p className="mt-3 text-xs text-muted">
        Advisory only — not auto-traded.{pick.model ? ` · ${pick.model}` : ""}
      </p>
    </div>
  );
}
