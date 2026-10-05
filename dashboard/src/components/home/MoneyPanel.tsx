import { Area, AreaChart, ResponsiveContainer, YAxis } from "recharts";
import type { HomeAccount, HomeStatus } from "../../api/types";
import { CHART } from "../../lib/charts";
import { cn, formatUsd } from "../../lib/utils";

const signed = (v: number) => `${v >= 0 ? "+" : "−"}${formatUsd(Math.abs(v))}`;
const pct = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(2)}%`;

function Change({ change, changePct }: { change: number | null; changePct: number | null }) {
  if (change === null) return <p className="text-xs text-muted">No change yet today</p>;
  return (
    <p className="text-xs tabular-nums">
      <span className={change >= 0 ? "text-accent" : "text-loss"}>
        {signed(change)}
        {changePct !== null && ` (${pct(changePct)})`}
      </span>{" "}
      <span className="text-muted">today</span>
    </p>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between text-xs text-muted">
      <span>{label}</span>
      <b className="font-medium tabular-nums text-[#e0e0e0]">{value}</b>
    </div>
  );
}

const BADGE: Record<HomeAccount["status"], string> = {
  active: "border-accent/35 bg-accent/5 text-accent",
  paused: "border-muted/35 text-muted",
  disabled: "border-muted/35 text-muted",
};

function Account({ a }: { a: HomeAccount }) {
  const investedShare = a.invested !== null && a.equity ? a.invested / a.equity : null;
  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <div className="flex items-center justify-between gap-2">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-muted">{a.label}</p>
        <span className={cn("rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase", BADGE[a.status])}>
          {a.status}
        </span>
      </div>
      <p className="mt-1.5 text-[22px] font-bold tabular-nums text-white">{formatUsd(a.equity)}</p>
      <Change change={a.change} changePct={a.change_pct} />
      {investedShare !== null && (
        <div className="my-2.5 flex h-1.5 overflow-hidden rounded bg-border">
          <i className="bg-accent" style={{ width: `${investedShare * 100}%` }} />
          <i className="bg-blue-400" style={{ width: `${(1 - investedShare) * 100}%` }} />
        </div>
      )}
      <div className="mt-2 grid gap-0.5">
        {a.invested !== null && <Row label="Invested" value={formatUsd(a.invested)} />}
        {a.cash !== null && <Row label="Cash" value={formatUsd(a.cash)} />}
        <Row
          label="Positions"
          value={
            a.positions
              ? `${a.positions}${a.positions <= 3 ? ` (${a.symbols.join(", ")})` : ""}`
              : "none"
          }
        />
        {a.source !== "live" && (
          <p className="mt-1 text-[11px] text-muted">
            {a.source === "ledger" ? `As of the close, ${a.as_of}` : `As of the ${a.as_of} run`}
          </p>
        )}
      </div>
    </div>
  );
}

export function MoneyPanel({ data }: { data: HomeStatus }) {
  const { total, accounts, set_aside } = data;
  const series = total.series;
  return (
    <section>
      <h2 className="mb-3 flex flex-wrap items-baseline justify-between gap-x-2 text-xs font-semibold uppercase tracking-widest text-muted">
        Money
        <span className="basis-full text-xs font-normal normal-case tracking-normal sm:basis-auto">
          {accounts.every((a) => a.paper)
            ? "All accounts are paper until the live-money gate passes"
            : "Includes real money"}
        </span>
      </h2>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-[1.3fr_1fr_1fr_1fr]">
        <div className="rounded-xl border border-border bg-surface p-4">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-muted">Total across accounts</p>
          <p className="my-1 text-[34px] font-bold leading-tight tabular-nums text-white">
            {formatUsd(total.equity)}
          </p>
          <Change change={total.change} changePct={total.change_pct} />
          {series.length > 1 && (
            <div className="mt-2 h-14">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={series} margin={{ top: 2, bottom: 0, left: 0, right: 0 }}>
                  <defs>
                    <linearGradient id="homeTotal" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={CHART.accent} stopOpacity={0.35} />
                      <stop offset="100%" stopColor={CHART.accent} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <YAxis hide domain={["dataMin", "dataMax"]} />
                  <Area
                    type="monotone"
                    dataKey="equity"
                    stroke={CHART.accent}
                    strokeWidth={2}
                    fill="url(#homeTotal)"
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
          {total.change_30d_pct !== null && (
            <Row label="30 days" value={pct(total.change_30d_pct)} />
          )}
        </div>
        {accounts.map((a) => (
          <Account key={a.account} a={a} />
        ))}
        <div className="rounded-xl border border-border bg-surface p-4">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-muted">Set aside</p>
          <p className="mt-1.5 text-[22px] font-bold tabular-nums text-white">
            {formatUsd(set_aside.purification_unpaid)}
          </p>
          <p className="text-xs text-muted">purification owed, held as cash</p>
          {set_aside.zakat && (
            <div className="mt-3.5 grid gap-0.5">
              <Row label="Zakat if due today" value={formatUsd(set_aside.zakat.amount)} />
              <Row label="Method" value={`${set_aside.zakat.chosen} (higher)`} />
              <Row
                label="Next hawl"
                value={new Date(`${set_aside.zakat.next_hawl}T12:00:00`).toLocaleDateString("en-GB", {
                  day: "numeric",
                  month: "short",
                  year: "numeric",
                })}
              />
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
