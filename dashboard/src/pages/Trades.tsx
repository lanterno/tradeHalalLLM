import { useState, useMemo, useCallback } from "react";
import { useTrades } from "../hooks/useTrades";
import { TradesTable } from "../components/TradesTable";
import { entityOf } from "../lib/utils";
import { entityLabel, useMarket } from "../lib/market";

const PAGE_SIZE = 25;

export default function Trades() {
  const { market } = useMarket();
  const [page, setPage] = useState(0);
  const [pair, setPair] = useState("");
  const [side, setSide] = useState("");

  // Filters are per-market — a symbol picked in stocks would otherwise leak
  // into crypto (and vice versa) and silently empty the table. Reset them
  // during render when the market changes (React's blessed alternative to a
  // setState-in-effect: https://react.dev/learn/you-might-not-need-an-effect).
  const [prevMarket, setPrevMarket] = useState(market);
  if (market !== prevMarket) {
    setPrevMarket(market);
    setPair("");
    setSide("");
    setPage(0);
  }

  const filters = useMemo(
    () => ({
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
      pair: pair || undefined,
      side: side || undefined,
    }),
    [page, pair, side],
  );

  const { data: trades, isLoading } = useTrades(filters);
  const { data: allTrades } = useTrades({ limit: 500 });

  const entities = useMemo(() => {
    if (!allTrades) return [];
    return [...new Set(allTrades.map(entityOf))].filter(Boolean).sort();
  }, [allTrades]);

  const exportCsv = useCallback(() => {
    if (!allTrades?.length) return;
    const headers = [
      "timestamp",
      "symbol",
      "side",
      "quantity",
      "price",
      "filled_price",
      "exit_price",
      "exit_reason",
      "status",
      "llm_reasoning",
    ];
    const cell = (v: unknown) => {
      const s = v == null ? "" : String(v);
      // Quote if the value contains a comma, quote, or newline (llm_reasoning
      // can span lines); double up embedded quotes per RFC 4180.
      return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const rows = allTrades.map((t) =>
      [
        t.timestamp,
        entityOf(t),
        t.side,
        t.quantity,
        t.price,
        t.filled_price ?? t.entry_price ?? "",
        t.exit_price ?? "",
        t.exit_reason ?? "",
        t.status,
        t.llm_reasoning,
      ]
        .map(cell)
        .join(","),
    );
    const csv = [headers.join(","), ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trades-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [allTrades]);

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Trade History</h1>
        <button
          onClick={exportCsv}
          className="rounded-lg border border-border bg-surface px-3 py-1.5 text-xs font-medium text-muted hover:text-white hover:bg-surface-hover transition-colors"
        >
          Export CSV
        </button>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3">
        <select
          value={pair}
          onChange={(e) => {
            setPair(e.target.value);
            setPage(0);
          }}
          className="rounded-lg border border-border bg-surface px-3 py-1.5 text-sm text-gray-200 outline-none focus:border-accent"
        >
          <option value="">All {entityLabel(market)}s</option>
          {entities.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
        <select
          value={side}
          onChange={(e) => {
            setSide(e.target.value);
            setPage(0);
          }}
          className="rounded-lg border border-border bg-surface px-3 py-1.5 text-sm text-gray-200 outline-none focus:border-accent"
        >
          <option value="">All Sides</option>
          <option value="buy">Buy</option>
          <option value="sell">Sell</option>
        </select>
      </div>

      {/* Table */}
      <div className="rounded-xl border border-border bg-surface p-4">
        {isLoading ? (
          <p className="py-8 text-center text-sm text-muted">Loading...</p>
        ) : (
          <TradesTable trades={trades ?? []} showReasoning />
        )}
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between text-sm text-muted">
        <button
          disabled={page === 0}
          onClick={() => setPage((p) => p - 1)}
          className="rounded-lg border border-border px-3 py-1.5 hover:bg-surface-hover disabled:opacity-30 transition-colors"
        >
          Previous
        </button>
        <span>Page {page + 1}</span>
        <button
          disabled={!trades || trades.length < PAGE_SIZE}
          onClick={() => setPage((p) => p + 1)}
          className="rounded-lg border border-border px-3 py-1.5 hover:bg-surface-hover disabled:opacity-30 transition-colors"
        >
          Next
        </button>
      </div>
    </div>
  );
}
