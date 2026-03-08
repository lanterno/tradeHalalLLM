import { useState, useMemo, useCallback } from "react";
import { useTrades } from "../hooks/useTrades";
import { TradesTable } from "../components/TradesTable";

const PAGE_SIZE = 25;

export default function Trades() {
  const [page, setPage] = useState(0);
  const [pair, setPair] = useState("");
  const [side, setSide] = useState("");

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

  const pairs = useMemo(() => {
    if (!allTrades) return [];
    return [...new Set(allTrades.map((t) => t.pair))].sort();
  }, [allTrades]);

  const exportCsv = useCallback(() => {
    if (!allTrades?.length) return;
    const headers = [
      "timestamp",
      "pair",
      "side",
      "quantity",
      "price",
      "entry_price",
      "exit_price",
      "exit_reason",
      "status",
      "llm_reasoning",
    ];
    const rows = allTrades.map((t) =>
      headers
        .map((h) => {
          const v = t[h as keyof typeof t];
          const s = v == null ? "" : String(v);
          return s.includes(",") ? `"${s}"` : s;
        })
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
          <option value="">All Pairs</option>
          {pairs.map((p) => (
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
