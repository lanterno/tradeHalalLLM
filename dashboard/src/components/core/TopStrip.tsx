import type { CoreStatus } from "../../api/types";
import { cn, formatPct, formatTime, formatUsd } from "../../lib/utils";
import { day, dayTime, pts, toneOf } from "./format";
import { SignedPct } from "./shared";

function Tile({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0 border-border p-4 sm:border-l sm:first:border-l-0">
      <p className="text-[11px] font-semibold uppercase tracking-widest text-muted">{label}</p>
      <div className="mt-1.5 text-xs text-muted">{children}</div>
    </div>
  );
}

const Big = ({ children, className }: { children: React.ReactNode; className?: string }) => (
  <p className={cn("text-xl font-bold tabular-nums text-white sm:text-2xl", className)}>{children}</p>
);

export function CoreHeader({ data }: { data: CoreStatus }) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-2xl font-bold text-white">Core portfolio</h1>
          <span className="rounded-full border border-accent/40 bg-accent/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-accent">
            The product
          </span>
          <span
            className={cn(
              "rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider",
              data.paper
                ? "border-warning/40 bg-warning/10 text-warning"
                : "border-loss/40 bg-loss/10 text-loss",
            )}
          >
            {data.paper ? "Paper" : "Live money"}
          </span>
        </div>
        <p className="mt-1 max-w-3xl text-xs text-muted">
          The largest 100 names the strict screen passes, cap-weighted, rebalanced monthly within
          bands, on its own Alpaca account. A holding that stops passing is sold at the next daily
          check. Real money only after the gate below passes.
        </p>
      </div>
      <div className="text-xs text-muted sm:text-right">
        {data.equity_day && (
          <p>
            {data.equity_source === "live" ? "account snapshot" : "ledger close"}{" "}
            <span className="text-white">
              {data.equity_day.length === 10 ? day(data.equity_day) : formatTime(data.equity_day)}
            </span>
          </p>
        )}
        <p>
          next daily check <span className="text-white">{dayTime(data.next_check)} ET</span>
        </p>
        <p>
          next rebalance <span className="text-white">{dayTime(data.next_rebalance)} ET</span>
        </p>
      </div>
    </div>
  );
}

export function TopStrip({ data }: { data: CoreStatus }) {
  const spus = data.today_vs.find((b) => b.symbol === "SPUS");
  const r = data.readiness;
  return (
    <div className="grid grid-cols-2 overflow-hidden rounded-xl border border-border bg-surface sm:grid-cols-3 xl:grid-cols-6">
      <Tile label="Equity">
        <Big>{formatUsd(data.equity)}</Big>
        {data.change !== null && (
          <p className={toneOf(data.change)}>
            {formatUsd(data.change, { signed: true })} ({formatPct(data.change_pct, 2, { signed: true })})
            today
          </p>
        )}
      </Tile>
      <Tile label="Today vs halal ETFs">
        <Big className={toneOf(spus?.diff_pts)}>{pts(spus?.diff_pts)}</Big>
        <p>vs SPUS</p>
        {data.today_vs
          .filter((b) => b.symbol !== "SPUS")
          .map((b) => (
            <p key={b.symbol}>
              <span className={toneOf(b.diff_pts)}>{pts(b.diff_pts)}</span> vs {b.symbol}
            </p>
          ))}
        <p className="mt-1">
          {data.today_vs.map((b, i) => (
            <span key={b.symbol}>
              {i > 0 && " · "}
              {b.symbol} <SignedPct v={b.change_pct} />
            </span>
          ))}
        </p>
      </Tile>
      <Tile label="Since the start">
        {data.since ? (
          <>
            <Big className={toneOf(data.since.change_pct)}>
              {formatPct(data.since.change_pct, 2, { signed: true })}
            </Big>
            <p>
              {formatUsd(data.since.change, { signed: true })} from{" "}
              {formatUsd(data.since.start_equity)} on {day(data.since.since)}
            </p>
            <p>
              SPUS <SignedPct v={data.since.spus_pct} /> · book{" "}
              {data.since.book_pct == null ? "—" : <SignedPct v={data.since.book_pct} />}
            </p>
          </>
        ) : (
          <Big>—</Big>
        )}
      </Tile>
      <Tile label="Cash">
        <Big>{formatUsd(data.cash)}</Big>
        {data.equity && data.cash !== null && (
          <p>
            {formatPct(data.cash / data.equity, 2)} · invested {formatUsd(data.invested)}
          </p>
        )}
      </Tile>
      <Tile label="Positions">
        <Big>{data.positions}</Big>
        {data.to_sell.length > 0 && (
          <p className="text-loss">
            {data.to_sell.length} to be sold: {data.to_sell.map((s) => s.symbol).join(", ")}
          </p>
        )}
        <p>top 10 = {formatPct(data.top10_weight)}</p>
      </Tile>
      <Tile label="Mode">
        <Big className={data.paper ? "text-warning" : "text-loss"}>{data.paper ? "Paper" : "Live"}</Big>
        {data.paper && (
          <p>
            live after the gate passes, earliest{" "}
            <span className="text-white">{day(r.earliest)}</span>
          </p>
        )}
        <p>{data.monthly_due ? "monthly rebalance due" : `rebalance ${day(data.next_rebalance)}`}</p>
      </Tile>
    </div>
  );
}
