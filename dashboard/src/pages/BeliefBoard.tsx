import { useState } from "react";
import type { BeliefOverview, Stance } from "../api/types";
import { ErrorState } from "../components/ErrorState";
import { BoardTable } from "../components/beliefs/BoardTable";
import { FILTERS, STANCE, pct } from "../components/beliefs/format";
import { BeliefSheet, PhoneList } from "../components/beliefs/PhoneList";
import { DecisionsCard, MacroCalendarCard, ShadowBookCard } from "../components/beliefs/SideColumn";
import { TrustStrip, VerdictBanner } from "../components/beliefs/TrustStrip";
import { useBeliefBoard, useBeliefOverview } from "../hooks/useBeliefs";
import { useNow } from "../hooks/useNow";
import { cn, formatClock, relativeTime } from "../lib/utils";

type Filter = Stance | "all";

function EngineStatus({ engine }: { engine: BeliefOverview["engine"] }) {
  useNow(30_000); // keep "refreshed … ago" moving between polls
  const age = engine.heartbeat_age_s;
  const badge =
    engine.status === "live"
      ? { text: "engine live", tone: "border-accent/35 bg-accent/5 text-accent" }
      : engine.status === "stale"
        ? {
            text: `engine stale${age != null ? ` · ${Math.round(age / 60)} min` : ""}`,
            tone: "border-loss/35 bg-loss/5 text-loss",
          }
        : { text: "no engine heartbeat", tone: "border-muted/35 bg-muted/5 text-muted" };
  return (
    <div className="text-xs text-muted sm:text-right">
      <span
        className={cn(
          "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide",
          badge.tone,
        )}
      >
        ● {badge.text}
      </span>
      <p className="mt-1">
        {engine.last_bar_at && (
          <>
            bars to <b className="font-medium text-white">{formatClock(engine.last_bar_at, { et: true, withDay: true })}</b>
            {" · "}
          </>
        )}
        <b className="font-medium text-white">{engine.names}</b> names
        {engine.refreshed_at && (
          <>
            {" · "}beliefs refreshed <b className="font-medium text-white">{relativeTime(engine.refreshed_at)}</b>
          </>
        )}
      </p>
    </div>
  );
}

function Switch({ on, onChange, children }: { on: boolean; onChange: (v: boolean) => void; children: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      onClick={() => onChange(!on)}
      className={cn("inline-flex items-center gap-1.5 text-xs", on ? "text-white" : "text-muted hover:text-white")}
    >
      <i
        className={cn(
          "relative inline-block h-3.5 w-[26px] rounded-full transition-colors",
          on ? "bg-accent/30" : "bg-surface-hover",
        )}
      >
        <i
          className={cn(
            "absolute top-0.5 h-2.5 w-2.5 rounded-full transition-all",
            on ? "left-[14px] bg-accent" : "left-0.5 bg-muted",
          )}
        />
      </i>
      {children}
    </button>
  );
}

function Skeleton() {
  return (
    <div className="space-y-4">
      {[110, 40, 520].map((h) => (
        <div key={h} className="animate-pulse rounded-xl border border-border bg-surface" style={{ height: h }} />
      ))}
    </div>
  );
}

export default function BeliefBoard() {
  const board = useBeliefBoard();
  const overview = useBeliefOverview();
  const [filter, setFilter] = useState<Filter>("all");
  const [onlyCore, setOnlyCore] = useState(false);
  const [onlyShadow, setOnlyShadow] = useState(false);
  const [openAsset, setOpenAsset] = useState<string | null>(null);
  const [sheetAsset, setSheetAsset] = useState<string | null>(null);

  const data = board.data;
  const beliefs = data?.beliefs ?? [];
  const counts: Record<Filter, number> = { all: beliefs.length, long: 0, leaning: 0, none: 0, excluded: 0, benchmark: 0 };
  for (const b of beliefs) counts[b.stance] += 1;
  const rows = beliefs.filter(
    (b) =>
      (filter === "all" || b.stance === filter) &&
      (!onlyCore || (b.core_weight ?? 0) > 0) &&
      (!onlyShadow || b.shadow != null),
  );
  const sheet = sheetAsset ? beliefs.find((b) => b.asset === sheetAsset) : undefined;
  const o = overview.data;

  return (
    <div className="space-y-4 p-4 sm:p-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="min-w-0">
          <h1 className="flex flex-wrap items-center gap-2.5 text-xl font-bold text-white sm:text-2xl">
            Belief Board
            <span className="inline-flex items-center rounded-full border border-purple-400/35 bg-purple-400/5 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-purple-400">
              <span className="sm:hidden">Research</span>
              <span className="hidden sm:inline">Research · advisory</span>
            </span>
          </h1>
          <p className="mt-1 hidden max-w-3xl text-[13px] text-muted sm:block">
            What the shadow engine currently thinks about each stock, and why. It paper-trades a book of its own to earn
            a track record; nothing here touches the core portfolio.
          </p>
        </div>
        {o && (
          <div className="ml-auto hidden sm:block">
            <EngineStatus engine={o.engine} />
          </div>
        )}
      </div>

      {/* The verdict comes first everywhere: a strip of four on a wide
          screen, one pinned line on a phone. */}
      {overview.isError ? (
        <div className="rounded-xl border border-border bg-surface">
          <ErrorState compact error={overview.error} onRetry={overview.refetch} />
        </div>
      ) : o ? (
        <>
          <div className="hidden md:block">
            <TrustStrip o={o} />
          </div>
          <div className="sticky -top-4 z-20 -mx-4 bg-bg px-4 py-2 md:hidden">
            <VerdictBanner o={o} />
          </div>
        </>
      ) : (
        overview.isLoading && <div className="h-[110px] animate-pulse rounded-xl border border-border bg-surface" />
      )}

      {board.isError ? (
        <div className="rounded-xl border border-border bg-surface">
          <ErrorState error={board.error} onRetry={board.refetch} />
        </div>
      ) : board.isLoading || !data ? (
        <Skeleton />
      ) : !data.available ? (
        <div className="rounded-xl border border-border bg-surface p-6 text-sm text-muted">
          No active beliefs yet. The shadow engine builds them while its daemon runs.
        </div>
      ) : (
        <>
          <div className="flex flex-wrap items-center justify-between gap-2.5">
            <div className="flex min-w-0 flex-wrap items-center gap-x-4 gap-y-2">
              {/* Phone: chips that scroll inside their own row. */}
              <div className="-mx-4 flex max-w-[100vw] gap-1.5 overflow-x-auto px-4 md:hidden">
                {FILTERS.map((f) => (
                  <button
                    key={f}
                    type="button"
                    onClick={() => setFilter(f)}
                    className={cn(
                      "shrink-0 rounded-full border border-border px-2.5 py-1 text-[11px] whitespace-nowrap",
                      filter === f ? "bg-surface-hover text-white" : "text-muted",
                    )}
                  >
                    {f === "all" ? "All" : STANCE[f].short} {counts[f]}
                  </button>
                ))}
              </div>
              <div className="hidden overflow-hidden rounded-lg border border-border md:inline-flex">
                {FILTERS.map((f) => (
                  <button
                    key={f}
                    type="button"
                    aria-pressed={filter === f}
                    onClick={() => setFilter(f)}
                    className={cn(
                      "border-r border-border px-3 py-1.5 text-xs last:border-r-0",
                      filter === f ? "bg-surface-hover text-white" : "text-muted hover:text-white",
                    )}
                  >
                    {f === "all" ? "All" : STANCE[f].short}
                    <b className="ml-1 font-semibold text-white">{counts[f]}</b>
                  </button>
                ))}
              </div>
              <span className="hidden gap-4 md:inline-flex">
                <Switch on={onlyCore} onChange={setOnlyCore}>
                  only names in the core
                </Switch>
                <Switch on={onlyShadow} onChange={setOnlyShadow}>
                  only the shadow's holdings
                </Switch>
              </span>
            </div>
            <p className="hidden text-[11px] text-muted lg:block">
              sorted by conviction · │ on each bar = the {pct(data.entry_band, 0)} entry line
            </p>
          </div>

          <div className="grid items-start gap-4 min-[1400px]:grid-cols-[minmax(0,1fr)_300px]">
            <div className="min-w-0">
              <div className="hidden lg:block">
                <BoardTable
                  board={data}
                  rows={rows}
                  openAsset={openAsset}
                  onToggle={(a) => setOpenAsset(openAsset === a ? null : a)}
                />
              </div>
              <div className="lg:hidden">
                <PhoneList board={data} rows={rows} onOpen={setSheetAsset} />
              </div>
            </div>
            {o && (
              <div className="grid min-w-0 gap-4 md:max-[1399px]:grid-cols-2 lg:max-[1399px]:grid-cols-3">
                <ShadowBookCard book={o.book} />
                <MacroCalendarCard events={o.calendar} />
                <DecisionsCard />
              </div>
            )}
          </div>
          {o && (
            <div className="sm:hidden">
              <EngineStatus engine={o.engine} />
            </div>
          )}
        </>
      )}

      {sheet && (
        <div className="lg:hidden">
          <BeliefSheet b={sheet} onClose={() => setSheetAsset(null)} />
        </div>
      )}
    </div>
  );
}
