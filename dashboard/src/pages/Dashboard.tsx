import { Link } from "react-router-dom";
import { useHome } from "../hooks/useHome";
import { useHealth, useSystemStatus } from "../hooks/useSystem";
import { useAnalytics } from "../hooks/useAnalytics";
import { useRecommendationScorecard, useStockOfTheDay } from "../hooks/useRecommendation";
import { ErrorState } from "../components/ErrorState";
import { VerdictBadge } from "../components/PickVerdict";
import { verdictSummary } from "../lib/scorecard";
import { MarketPanel } from "../components/home/MarketPanel";
import { MoneyPanel } from "../components/home/MoneyPanel";
import { PortfolioPanel } from "../components/home/PortfolioPanel";
import { GatePanel, UpcomingPanel } from "../components/home/PlanPanels";
import { cn, formatDate, formatPct, formatUsd, pnlColor, todayET } from "../lib/utils";
import { staleComponents, type HealthStatus } from "../api/types";

const HIJRI_MONTHS = [
  "Muharram", "Safar", "Rabiʿ al-Awwal", "Rabiʿ al-Thani", "Jumada al-Ula", "Jumada al-Akhirah",
  "Rajab", "Shaʿban", "Ramadan", "Shawwal", "Dhu al-Qaʿdah", "Dhu al-Hijjah",
];

function hijri(label: string): string {
  const [y, m, d] = label.replace(" AH", "").split("-").map(Number);
  return `${d} ${HIJRI_MONTHS[m - 1]} ${y} AH`;
}

function Badge({ ok, children }: { ok: boolean | "paper"; children: React.ReactNode }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide",
        ok === "paper"
          ? "border-warning/35 bg-warning/5 text-warning"
          : ok
            ? "border-accent/35 bg-accent/5 text-accent"
            : "border-loss/35 bg-loss/5 text-loss",
      )}
    >
      ● {children}
    </span>
  );
}

function HealthBadge({ health, dayTraderEnabled }: { health: HealthStatus; dayTraderEnabled?: boolean }) {
  const stale = staleComponents(health, { dayTraderEnabled });
  if (!health.bot_alive) return <Badge ok={false}>Bot not reporting</Badge>;
  if (stale.length) return <Badge ok={false}>Stale: {stale.join(", ")}</Badge>;
  return <Badge ok>All systems healthy</Badge>;
}

/**
 * The advisory pick, always beside its own record. Kept on the page rather than
 * hidden while the record is negative: hiding it would also hide that the
 * experiment is still running (and spending) with nothing to show for it; the
 * verdict says plainly not to act on it.
 */
function StockOfTheDay() {
  const { data: pick } = useStockOfTheDay();
  const { data: sc } = useRecommendationScorecard();
  const stale = pick?.date ? pick.date !== todayET() : false;
  const negative = sc?.verdict === "negative";
  const mode = sc?.conviction_mode;
  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <h2 className="mb-3 flex items-baseline justify-between gap-2 text-xs font-semibold uppercase tracking-widest text-muted">
        Stock of the day
        <Link to="/recommendation" className="text-xs font-normal normal-case tracking-normal hover:text-white">
          advisory · never traded →
        </Link>
      </h2>
      {pick?.available && pick.symbol ? (
        <>
          <div className="flex items-center gap-3.5">
            <span className={cn("text-[22px] font-bold", negative ? "text-muted line-through decoration-loss/60" : "text-white")}>
              {pick.symbol}
            </span>
            <div className="min-w-0">
              <p className="flex flex-wrap items-center gap-x-2 text-sm text-white">
                <span>{pick.date ? `Picked ${formatDate(pick.date)}` : "Undated pick"}</span>
                {stale && (
                  <span className="rounded border border-warning/35 px-1.5 text-[10px] font-semibold uppercase tracking-wide text-warning">
                    stale
                  </span>
                )}
              </p>
              <p className="text-xs text-muted">
                Conviction {pick.conviction !== undefined ? pick.conviction.toFixed(2) : "—"}
                {mode && mode.n > 1 && mode.value === Math.round((pick.conviction ?? -1) * 100) / 100
                  ? ` (the same ${mode.value.toFixed(2)} on ${mode.n} of ${mode.of} days)`
                  : ""}
              </p>
              {!negative && <p className="line-clamp-2 text-xs text-muted">{pick.thesis}</p>}
            </div>
          </div>
          {sc && (
            <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-border pt-3 text-xs text-muted">
              <VerdictBadge verdict={sc.verdict} />
              <span>{verdictSummary(sc)}</span>
              {negative && <span className="text-loss">Do not act on it.</span>}
            </div>
          )}
        </>
      ) : (
        <p className="text-sm text-muted">No pick yet.</p>
      )}
    </div>
  );
}

function DayTraderRecord() {
  const { data: stats } = useAnalytics(30);
  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <details>
        <summary className="flex cursor-pointer list-none items-baseline justify-between text-xs font-semibold uppercase tracking-widest text-muted">
          Day-trader record (retired)
          <span className="text-xs font-normal normal-case tracking-normal">closed in the last 30 days ▾</span>
        </summary>
        {stats ? (
          <div className="mt-3 grid gap-0.5 text-xs text-muted">
            <div className="flex justify-between">
              <span>Closed-trade P&amp;L</span>
              <b className={cn("font-medium tabular-nums", pnlColor(stats.total_pnl))}>{formatUsd(stats.total_pnl)}</b>
            </div>
            <div className="flex justify-between">
              <span>Win rate</span>
              <b className="font-medium tabular-nums text-[#e0e0e0]">
                {formatPct(stats.win_rate)} ({stats.wins}W / {stats.losses}L)
              </b>
            </div>
            <div className="flex justify-between">
              <span>Trades</span>
              <b className="font-medium tabular-nums text-[#e0e0e0]">{stats.total_trades}</b>
            </div>
          </div>
        ) : (
          <p className="mt-3 text-xs text-muted">Loading…</p>
        )}
      </details>
    </div>
  );
}

export default function Dashboard() {
  const { data, isLoading, isError, error, refetch } = useHome();
  const { data: health } = useHealth();
  const { data: status, isError: statusError } = useSystemStatus();
  // Whether the day-trader runs: the backend says so directly; the home
  // payload's account status is the fallback for an older backend.
  const paper = data?.accounts.find((a) => a.account === "paper");
  const dayTraderEnabled = status?.day_trader_enabled ?? (paper ? paper.status === "active" : undefined);
  // Wait for that answer before judging staleness, so a retired day-trader's
  // old cycle beat never flashes the badge red on load.
  const healthReady = health && (status || statusError || data);
  const todayLong = new Date().toLocaleDateString("en-GB", {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
  });

  return (
    <div className="space-y-5 p-4 sm:p-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-white">Home</h1>
          <p className="text-xs text-muted">
            {todayLong}
            {data ? ` · ${hijri(data.today_hijri)}` : ""}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {data && data.accounts.every((a) => a.paper) && <Badge ok="paper">Paper money</Badge>}
          {healthReady && <HealthBadge health={health} dayTraderEnabled={dayTraderEnabled} />}
        </div>
      </div>

      {isError ? (
        <div className="rounded-xl border border-border bg-surface p-4">
          <ErrorState error={error} onRetry={refetch} />
        </div>
      ) : isLoading || !data ? (
        <div className="space-y-5">
          {[180, 160, 320].map((h) => (
            <div key={h} className="animate-pulse rounded-xl border border-border bg-surface" style={{ height: h }} />
          ))}
        </div>
      ) : (
        <>
          <MarketPanel market={data.market} />
          <MoneyPanel data={data} />
          <PortfolioPanel portfolio={data.portfolio} />
          <section className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <UpcomingPanel items={data.upcoming} />
            <GatePanel gate={data.gate} />
          </section>
        </>
      )}

      <section className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <StockOfTheDay />
        <DayTraderRecord />
      </section>
    </div>
  );
}
