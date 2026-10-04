import { useZakat, usePurification } from "../hooks/useHalal";
import { formatUsd, cn } from "../lib/utils";

function Method({
  name,
  basis,
  amount,
  higher,
}: {
  name: string;
  basis: string;
  amount: number;
  higher: boolean;
}) {
  return (
    <div
      className={cn(
        "rounded-lg border p-3",
        higher ? "border-accent/50 bg-accent/5" : "border-border bg-bg/40",
      )}
    >
      <div className="flex items-baseline justify-between gap-2">
        <p className="text-sm font-medium text-white">{name}</p>
        {higher && (
          <span className="rounded bg-accent/15 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-accent">
            higher
          </span>
        )}
      </div>
      <p className="mt-1 text-xl font-bold text-white">{formatUsd(amount)}</p>
      <p className="mt-0.5 text-xs text-muted">{basis}</p>
    </div>
  );
}

/** Zakat by both of Dar al-Ifta's methods, the higher taken, and the year's
 *  purification per holding. */
export function ZakatPanel() {
  const zakat = useZakat();
  const purification = usePurification();
  const z = zakat.data;
  const now = z?.if_due_today;

  return (
    <>
      <section>
        <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-muted">
          Zakat
        </h2>
        {!z ? (
          <p className="text-sm text-muted">{zakat.isError ? "Unavailable." : "Loading…"}</p>
        ) : !z.configured ? (
          <p className="text-sm text-muted">Set ZAKAT_HAWL_HIJRI to enable.</p>
        ) : (
          <div className="space-y-4 rounded-xl border border-border bg-surface p-4">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <p className="text-sm text-white">
                Next hawl: <span className="font-semibold">{z.next_hawl}</span>{" "}
                <span className="text-muted">({z.next_hawl_hijri})</span>
              </p>
              <p className="text-xs text-muted">in {z.days_to_next} days</p>
            </div>
            {now && (
              <>
                <p className="text-xs uppercase tracking-wider text-muted">
                  If it were due today (since {z.last_hawl})
                </p>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <Method
                    name="Trade goods"
                    basis={`2.5% of market value ${formatUsd(now.market_value)}`}
                    amount={now.trade_goods_zakat}
                    higher={now.chosen === "trade goods"}
                  />
                  <Method
                    name="Income"
                    basis={`2.5% of dividends ${formatUsd(now.dividends)} less purified ${formatUsd(now.purified)}`}
                    amount={now.income_zakat}
                    higher={now.chosen === "income"}
                  />
                </div>
              </>
            )}
            {z.last_recorded && (
              <p className="text-xs text-muted">
                Last recorded: {z.last_recorded.hawl_date} ({z.last_recorded.hawl_hijri}) —{" "}
                <span className="text-white">{formatUsd(z.last_recorded.amount)}</span> by{" "}
                {z.last_recorded.chosen}
              </p>
            )}
            <p className="text-[11px] leading-relaxed text-muted">
              {z.source}. Paper account. The nisab test and cash held depend on your
              whole wealth and are not included.
            </p>
          </div>
        )}
      </section>

      <section>
        <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-muted">
          Purification by holding {purification.data ? `(${purification.data.year})` : ""}
        </h2>
        {!purification.data ? (
          <p className="text-sm text-muted">
            {purification.isError ? "Unavailable." : "Loading…"}
          </p>
        ) : purification.data.lines.length === 0 ? (
          <p className="text-sm text-muted">No dividends this year.</p>
        ) : (
          <div className="rounded-xl border border-border bg-surface p-4">
            <ul className="divide-y divide-border">
              {purification.data.lines.map((line) => (
                <li key={line.symbol} className="flex items-center justify-between gap-3 py-2 text-sm">
                  <span className="font-medium text-white">{line.symbol}</span>
                  <span className="text-right text-muted">
                    {formatUsd(line.dividends)} dividends ·{" "}
                    <span className="text-warning">{formatUsd(line.amount)}</span> to purify
                    {line.assumed > 0 && " (5% assumed)"}
                  </span>
                </li>
              ))}
            </ul>
            <p className="mt-3 text-xs text-muted">
              Total {formatUsd(purification.data.dividends)} dividends,{" "}
              <span className="text-white">{formatUsd(purification.data.amount)}</span> to purify.
              Interest income only: treat this as the minimum.
            </p>
          </div>
        )}
      </section>
    </>
  );
}
