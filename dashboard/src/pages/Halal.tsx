import { useHalalCompliance } from "../hooks/useHalal";
import { StatCard } from "../components/StatCard";
import { ZakatPanel } from "../components/ZakatPanel";
import { ErrorState } from "../components/ErrorState";
import type { AccountCompliance } from "../api/types";
import { cn, formatDate, formatUsd } from "../lib/utils";
import { CheckCircle2, AlertTriangle, XCircle } from "lucide-react";

const STATUS: Record<string, { label: string; cls: string; Icon: typeof CheckCircle2 }> = {
  compliant: { label: "Compliant", cls: "text-accent", Icon: CheckCircle2 },
  attention: { label: "Needs attention", cls: "text-warning", Icon: AlertTriangle },
  violation: { label: "Violation", cls: "text-loss", Icon: XCircle },
};

const VERDICT_LABEL: Record<string, string> = {
  halal: "Halal",
  doubtful: "Doubtful",
  not_halal: "Not halal",
  unscreened: "Not screened",
};

const VERDICT_CLS: Record<string, string> = {
  halal: "bg-accent",
  doubtful: "bg-warning",
  not_halal: "bg-loss",
  unscreened: "bg-muted",
};

const LABEL: Record<string, string> = { core: "Core portfolio", paper: "Day-trader (retired)" };

function AccountCard({ a }: { a: AccountCompliance }) {
  const s = STATUS[a.status] ?? STATUS.compliant;
  const buys = a.buys_this_quarter;
  const order = ["halal", "doubtful", "not_halal", "unscreened"] as const;
  return (
    <div className={cn("rounded-xl border bg-surface p-4", a.status === "violation" ? "border-loss/30" : "border-border")}>
      <p className="text-[11px] font-semibold uppercase tracking-wider text-muted">{LABEL[a.account] ?? a.label}</p>
      <div className="mt-2 flex items-center gap-3">
        <s.Icon className={`h-7 w-7 shrink-0 ${s.cls}`} aria-hidden />
        <div>
          <p className={`text-lg font-bold ${s.cls}`}>{s.label}</p>
          <p className="text-xs text-muted">
            Quarter to date · {a.trades_this_quarter} trades · {buys} buys
            {a.status === "violation"
              ? ` · ${a.non_halal_buys_quarter} not held halal by the screen that day`
              : buys
                ? " · every buy halal on its day"
                : ""}
          </p>
        </div>
      </div>

      {buys > 0 && (
        <>
          <div className="mt-3 flex h-2 overflow-hidden rounded-full bg-border">
            {order.map((v) =>
              a.buy_verdicts[v] ? (
                <i key={v} className={VERDICT_CLS[v]} style={{ width: `${(a.buy_verdicts[v] / buys) * 100}%` }} />
              ) : null,
            )}
          </div>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
            {order.map((v) => (
              <span key={v} className="flex items-center gap-1.5 tabular-nums">
                <i className={cn("h-2 w-2 rounded-sm", VERDICT_CLS[v])} />
                {VERDICT_LABEL[v]} <b className="font-medium text-white">{a.buy_verdicts[v]}</b>
              </span>
            ))}
          </div>
        </>
      )}

      {a.non_halal_buys.length > 0 && (
        <details className="mt-3 text-xs">
          <summary className="cursor-pointer text-muted hover:text-white">
            The {a.non_halal_buys_quarter} buy{a.non_halal_buys_quarter === 1 ? "" : "s"} ▾
          </summary>
          <div className="mt-2 overflow-x-auto">
            <table className="w-full min-w-[320px] tabular-nums">
              <thead>
                <tr className="text-left text-[10px] uppercase tracking-wider text-muted">
                  <th className="py-1 pr-3">Day</th>
                  <th className="py-1 pr-3">Symbol</th>
                  <th className="py-1 pr-3">Screen said</th>
                  <th className="py-1">Screen of</th>
                </tr>
              </thead>
              <tbody>
                {a.non_halal_buys.map((b, i) => (
                  <tr key={`${b.symbol}-${b.day}-${i}`} className="border-t border-border">
                    <td className="py-1 pr-3 text-muted">{formatDate(b.day)}</td>
                    <td className="py-1 pr-3 font-medium text-white">{b.symbol}</td>
                    <td className={cn("py-1 pr-3", b.verdict === "not_halal" ? "text-loss" : "text-warning")}>
                      {VERDICT_LABEL[b.verdict] ?? b.verdict}
                    </td>
                    <td className="py-1 text-muted">{b.screen_as_of ? formatDate(b.screen_as_of) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
      {a.account === "paper" && a.buys_this_quarter > 0 && (
        <p className="mt-3 text-[11px] text-muted">
          Judged by the in-house AAOIFI screen; the day-trader's own order gate used its older screener at the
          time, so a violation here is the two disagreeing. It buys nothing now.
        </p>
      )}
    </div>
  );
}

export default function Halal() {
  const { data, isLoading, isError, error, refetch } = useHalalCompliance();
  const unpaid = data ? Object.entries(data.purification_unpaid_by_account) : [];

  return (
    <div className="space-y-6 p-4 sm:p-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Halal Compliance</h1>
        <p className="mt-1 text-xs text-muted">
          Every buy checked against the in-house AAOIFI screen in force on its day; purification and zakat.
          Long-only, no interest/leverage/derivatives — non-negotiable.
        </p>
      </div>

      {isError ? (
        <ErrorState error={error} onRetry={refetch} />
      ) : isLoading ? (
        <p className="py-8 text-center text-sm text-muted">Loading…</p>
      ) : !data ? null : (
        <>
          <section>
            <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-muted">
              This quarter, by account <span className="normal-case tracking-normal">· since {formatDate(data.quarter_start)}</span>
            </h2>
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              {data.accounts.map((a) => (
                <AccountCard key={a.account} a={a} />
              ))}
            </div>
          </section>

          {/* Trade volume */}
          <section>
            <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-muted">Trades</h2>
            <div className="overflow-x-auto rounded-xl border border-border bg-surface p-2 sm:p-4">
              <table className="w-full min-w-[360px] text-sm tabular-nums">
                <thead>
                  <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                    <th className="px-3 py-2">Account</th>
                    <th className="px-3 py-2 text-right">Today</th>
                    <th className="px-3 py-2 text-right">This month</th>
                    <th className="px-3 py-2 text-right">This quarter</th>
                  </tr>
                </thead>
                <tbody>
                  {data.accounts.map((a) => (
                    <tr key={a.account} className="border-b border-border/50">
                      <td className="px-3 py-2">{LABEL[a.account] ?? a.label}</td>
                      <td className="px-3 py-2 text-right">{a.trades_today}</td>
                      <td className="px-3 py-2 text-right">{a.trades_this_month}</td>
                      <td className="px-3 py-2 text-right">{a.trades_this_quarter}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {/* Purification ledger */}
          <section>
            <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-muted">Purification</h2>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
              <StatCard label="Accrued · this quarter" value={formatUsd(data.purification_accrued_usd)} />
              <StatCard label="Disbursed · this quarter" value={formatUsd(data.purification_disbursed_usd)} />
              <StatCard
                label="Outstanding · all time"
                value={
                  <span className={data.purification_outstanding_usd > 0 ? "text-warning" : "text-accent"}>
                    {formatUsd(data.purification_outstanding_usd)}
                  </span>
                }
                sub={
                  unpaid.length
                    ? unpaid.map(([acct, v]) => `${LABEL[acct] ?? acct} ${formatUsd(v)}`).join(" · ")
                    : "nothing owed"
                }
              />
            </div>
          </section>

          <ZakatPanel />
        </>
      )}
    </div>
  );
}
