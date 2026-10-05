import { Link } from "react-router-dom";
import type { HomeStatus } from "../../api/types";
import { cn } from "../../lib/utils";

const COLORS = ["#4ade80", "#60a5fa", "#c084fc", "#facc15", "#fb923c", "#f87171", "#2dd4bf"];
const OTHER = "#374151";
const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;

function shortDate(iso: string): string {
  return new Date(`${iso}T12:00:00`).toLocaleDateString("en-GB", { day: "numeric", month: "short" });
}

export function PortfolioPanel({ portfolio }: { portfolio: HomeStatus["portfolio"] }) {
  const { holdings, sectors, screen } = portfolio;
  const top = holdings[0]?.weight || 1;
  const shown = sectors.slice(0, COLORS.length - 1);
  const rest = sectors.slice(shown.length).reduce((s, x) => s + x.weight, 0);
  const legend = [
    ...shown.map((s, i) => ({ ...s, color: COLORS[i] })),
    ...(rest > 0.0005 ? [{ sector: "Everything else", weight: rest, color: OTHER }] : []),
  ];

  return (
    <section>
      <h2 className="mb-3 flex flex-wrap items-baseline justify-between gap-x-2 text-xs font-semibold uppercase tracking-widest text-muted">
        Core portfolio
        <Link to="/core" className="basis-full text-xs font-normal normal-case tracking-normal hover:text-white sm:basis-auto">
          {portfolio.count} holdings · open the Core page →
        </Link>
      </h2>
      {portfolio.count === 0 ? (
        <p className="rounded-xl border border-border bg-surface p-4 text-sm text-muted">
          Nothing held yet. The core buys at its first monthly rebalance.
        </p>
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1.4fr_1fr]">
          <div className="rounded-xl border border-border bg-surface p-4">
            <table className="w-full tabular-nums">
              <tbody>
                {holdings.map((h, i) => (
                  <tr key={h.symbol} className={cn(i > 0 && "border-t border-border")}>
                    <td className="w-16 py-1.5 text-sm font-semibold text-white">{h.symbol}</td>
                    <td className="hidden max-w-40 truncate py-1.5 text-xs text-muted sm:table-cell">{h.name}</td>
                    <td className="w-1/3 py-1.5 pl-2.5">
                      <div className="h-1.5 rounded bg-accent/85" style={{ width: `${(h.weight / top) * 100}%` }} />
                    </td>
                    <td className="py-1.5 pl-2.5 text-right text-sm">{pct(h.weight)}</td>
                    <td
                      className={cn(
                        "py-1.5 pl-2.5 text-right text-sm",
                        h.change_today === null ? "text-muted" : h.change_today >= 0 ? "text-accent" : "text-loss",
                      )}
                    >
                      {h.change_today === null
                        ? "—"
                        : `${h.change_today >= 0 ? "+" : "−"}${Math.abs(h.change_today * 100).toFixed(1)}%`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-2 text-xs text-muted">
              Top {holdings.length} = {portfolio.top_weight !== null ? pct(portfolio.top_weight) : "—"} of the
              portfolio{portfolio.count > holdings.length ? ` · ${portfolio.count - holdings.length} more holdings` : ""}
            </p>
          </div>

          <div className="rounded-xl border border-border bg-surface p-4">
            <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-muted">By sector</p>
            <div className="mb-3.5 flex h-3.5 overflow-hidden rounded-full">
              {legend.map((s) => (
                <i key={s.sector} style={{ width: pct(s.weight, 2), background: s.color }} />
              ))}
            </div>
            <div className="grid gap-1.5 text-xs tabular-nums">
              {legend.map((s) => (
                <div key={s.sector} className="flex items-center gap-2">
                  <i className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: s.color }} />
                  <span>{s.sector}</span>
                  <span className="ml-auto text-white">{pct(s.weight)}</span>
                </div>
              ))}
            </div>
            <div
              className={cn(
                "mt-4 flex items-start gap-2.5 rounded-lg border px-3 py-2.5 text-xs",
                screen.failing.length ? "border-warning/30 bg-warning/5" : "border-accent/20 bg-accent/5",
              )}
            >
              <span className={screen.failing.length ? "text-warning" : "text-accent"}>
                {screen.failing.length ? "!" : "✓"}
              </span>
              <div>
                <b className="text-white">
                  {screen.failing.length
                    ? `${screen.failing.length} holding(s) no longer pass: ${screen.failing.slice(0, 5).join(", ")}`
                    : `All ${portfolio.count} pass the strict screen`}
                </b>
                <br />
                <span className="text-muted">
                  {screen.failing.length ? "Sold at the next core run (15:40 ET). " : ""}
                  {screen.as_of
                    ? `Screened ${shortDate(screen.as_of)} (${screen.halal} of ${screen.screened.toLocaleString()} names halal)`
                    : ""}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
