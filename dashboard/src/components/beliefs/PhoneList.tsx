// The board on a phone: compact rows, and a bottom sheet with the same
// detail the desktop row expands into.
import { useEffect } from "react";
import { X } from "lucide-react";
import type { Belief, BeliefBoard } from "../../api/types";
import { cn } from "../../lib/utils";
import { BeliefDetail } from "./BeliefDetail";
import { pct, regimeOf, usd } from "./format";
import { ConvictionBar, StancePill, StrictTag } from "./Marks";

export function PhoneList({
  board,
  rows,
  onOpen,
}: {
  board: BeliefBoard;
  rows: Belief[];
  onOpen: (asset: string) => void;
}) {
  if (rows.length === 0) return <p className="py-6 text-center text-sm text-muted">No names match these filters.</p>;
  return (
    <div>
      {rows.map((b) => {
        const dim = b.stance === "excluded" && "opacity-50";
        return (
          <button
            key={b.asset}
            type="button"
            onClick={() => onOpen(b.asset)}
            className="flex w-full items-center justify-between gap-3 border-b border-[#15151f] px-0.5 py-2.5 text-left"
          >
            <span className="min-w-0">
              <span className="flex items-center gap-1.5">
                <span className={cn("font-bold text-white", dim)}>{b.asset}</span>
                <StrictTag b={b} compact />
              </span>
              <span className={cn("block truncate text-[11px] text-muted", dim)}>
                {b.main_reason ?? "nothing in favour"}
              </span>
            </span>
            <span className="shrink-0 text-right">
              <StancePill stance={b.stance} />
              <span className={cn("mt-0.5 block text-[11px] tabular-nums", dim)}>
                <b className="text-white">{pct(b.conviction, 0)}</b>
                {b.price != null && ` · ${usd(b.price)}`}
              </span>
              <ConvictionBar
                value={b.conviction}
                entry={board.entry_band}
                exit={board.exit_band}
                className="mt-1 ml-auto w-[90px]"
              />
            </span>
          </button>
        );
      })}
      <p className="pt-2.5 text-center text-[11px] text-muted">tap a row for reasons, levels and decisions</p>
    </div>
  );
}

export function BeliefSheet({ b, onClose }: { b: Belief; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const regime = regimeOf(b);
  return (
    <>
      <div className="fixed inset-0 z-40 bg-black/60" onClick={onClose} aria-hidden="true" />
      <div
        role="dialog"
        aria-modal="true"
        aria-label={`${b.asset} detail`}
        className="fixed inset-x-0 bottom-0 z-50 max-h-[85dvh] overflow-y-auto rounded-t-2xl border-t border-border bg-surface px-4 pb-6"
      >
        <div className="sticky top-0 -mx-4 mb-3 flex items-center justify-between gap-3 border-b border-border bg-surface px-4 py-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-lg font-bold text-white">{b.asset}</span>
              <StancePill stance={b.stance} />
            </div>
            <p className="text-xs text-muted tabular-nums">
              {pct(b.conviction, 0)} conviction · {b.price != null ? usd(b.price) : "no price"} ·{" "}
              <span className={regime.tone}>{regime.label}</span> · core {b.core_weight ? pct(b.core_weight) : "—"}
            </p>
            <StrictTag b={b} />
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded-lg p-2 text-muted hover:bg-surface-hover hover:text-white"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        <BeliefDetail b={b} stacked />
      </div>
    </>
  );
}
