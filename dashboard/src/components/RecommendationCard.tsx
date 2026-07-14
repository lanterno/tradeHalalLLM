import type { CandidateQuant, StockOfTheDay } from "../api/types";
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

// Quantitative "how high / how low" grounding the LLM saw when it picked
// its levels: the calibrated statistical 5-day band and the options-implied
// move. Reads the pick's own entry in the candidates JSONB; renders nothing
// for older recs that predate the quant fields.
function QuantRange({ pick }: { pick: StockOfTheDay }) {
  const c: CandidateQuant | undefined =
    pick.symbol && pick.candidates ? pick.candidates[pick.symbol] : undefined;
  if (!c) return null;
  const hasBand = typeof c.band5d_lo === "number" && typeof c.band5d_hi === "number";
  const hasImpl = typeof c.impl_move_pct === "number";
  if (!hasBand && !hasImpl) return null;
  const calibrated = c.quant_bands?.calibrated;
  return (
    <div className="mt-4 rounded-lg border border-border bg-bg/40 p-3 text-sm">
      <p className="text-xs uppercase tracking-wider text-muted">Expected range</p>
      {hasBand && (
        <p className="mt-1 text-white">
          <span className="text-muted">5d band </span>
          <span className="text-cyan-400">
            {formatUsd(c.band5d_lo!)} .. {formatUsd(c.band5d_hi!)}
          </span>
          <span className="text-muted">
            {" "}
            ({calibrated ? "calibrated" : "uncalibrated"}
            {typeof c.vol_pctl === "number"
              ? ` · vol pctl ${formatPct(c.vol_pctl)}`
              : ""}
            )
          </span>
        </p>
      )}
      {hasImpl && (
        <p className="mt-1 text-white">
          <span className="text-muted">Options-implied </span>
          <span>
            ±{c.impl_move_pct!.toFixed(1)}%
            {typeof c.impl_dte === "number" ? `/${c.impl_dte}d` : ""}
          </span>
          {typeof c.impl_low === "number" && typeof c.impl_high === "number" ? (
            <span className="text-muted">
              {" "}
              ({formatUsd(c.impl_low)} .. {formatUsd(c.impl_high)})
            </span>
          ) : null}
        </p>
      )}
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

      <QuantRange pick={pick} />

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
