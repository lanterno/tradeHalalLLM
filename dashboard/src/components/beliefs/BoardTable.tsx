// The ranked board (desktop): one row per name, its detail opening beneath it.
// Fixed column widths, so a long headline ellipsises instead of widening the
// table past its card.
import { Fragment } from "react";
import type { Belief, BeliefBoard } from "../../api/types";
import { cn } from "../../lib/utils";
import { BeliefDetail } from "./BeliefDetail";
import { pct, regimeOf, signedPct, stopDistance, usd } from "./format";
import { ConvictionBar, Ladder, StancePill, StrictTag } from "./Marks";

// ``short`` stands in below 2xl, where the full header would not fit.
const COLS: { label: string; width?: number; short?: string }[] = [
  { label: "Stock", width: 96 },
  { label: "Stance · regime", width: 124 },
  { label: "Conviction", width: 92 },
  { label: "Price · stop · levels", width: 152 },
  { label: "Main reason · strongest counter", short: "Main reason · counter" },
  { label: "Shadow", width: 70 },
  { label: "Core", width: 54 },
  { label: "", width: 22 },
];

function Row({
  b,
  board,
  open,
  onToggle,
}: {
  b: Belief;
  board: BeliefBoard;
  open: boolean;
  onToggle: () => void;
}) {
  const regime = regimeOf(b);
  const stop = stopDistance(b);
  const dim = b.stance === "excluded" ? "opacity-45" : undefined;
  return (
    <tr
      className={cn(
        "cursor-pointer border-b border-[#15151f] hover:bg-[#14141d]",
        open && "border-b-transparent bg-[#14141d]",
      )}
      onClick={onToggle}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onToggle();
        }
      }}
      tabIndex={0}
      aria-expanded={open}
    >
      <td className="px-2.5 py-2.5">
        <div className="text-sm font-bold text-white">{b.asset}</div>
        <StrictTag b={b} />
      </td>
      <td className="px-2.5 py-2.5">
        <StancePill stance={b.stance} />
        <div className={cn("mt-1 text-[11px]", regime.tone, dim)}>
          {regime.label} <span className="tabular-nums text-muted">{pct(b.regime_confidence, 0)}</span>
        </div>
      </td>
      <td className={cn("px-2.5 py-2.5", dim)}>
        <div className="text-[13px] font-semibold tabular-nums text-white">{pct(b.conviction, 0)}</div>
        <ConvictionBar value={b.conviction} entry={board.entry_band} exit={board.exit_band} className="mt-1" />
      </td>
      <td className={cn("px-2.5 py-2.5", dim)}>
        <div className="text-[13px] tabular-nums text-white">{b.price != null ? usd(b.price) : "—"}</div>
        <Ladder b={b} />
        {stop && <div className="text-[11px] tabular-nums text-muted">stop {stop}</div>}
      </td>
      <td className={cn("px-2.5 py-2.5 text-xs", dim)}>
        {b.main_reason ? (
          <div className="truncate text-gray-300" title={b.main_reason}>
            <span className="text-accent">▲</span> {b.main_reason}
          </div>
        ) : (
          <div className="text-muted">nothing in favour</div>
        )}
        {b.counter_reason && (
          <div className="mt-0.5 truncate text-muted" title={b.counter_reason}>
            <span className="text-loss">▼</span> {b.counter_reason}
          </div>
        )}
      </td>
      <td className={cn("px-2.5 py-2.5 text-[13px] tabular-nums", dim)}>
        {b.shadow ? (
          <>
            {pct(b.shadow.weight)}
            <div className={cn("text-[11px]", b.shadow.return_pct >= 0 ? "text-accent" : "text-loss")}>
              {signedPct(b.shadow.return_pct)}
            </div>
          </>
        ) : (
          <span className="text-muted">—</span>
        )}
      </td>
      <td className={cn("px-2.5 py-2.5 text-[13px] tabular-nums", dim)}>
        {b.core_weight ? pct(b.core_weight) : <span className="text-muted">—</span>}
      </td>
      <td className="py-2.5 pr-2 text-muted">{open ? "▾" : "▸"}</td>
    </tr>
  );
}

export function BoardTable({
  board,
  rows,
  openAsset,
  onToggle,
}: {
  board: BeliefBoard;
  rows: Belief[];
  openAsset: string | null;
  onToggle: (asset: string) => void;
}) {
  return (
    <div className="overflow-hidden rounded-xl border border-border bg-surface px-1.5 pt-1 pb-1.5">
      <table className="w-full table-fixed border-collapse text-left">
        <colgroup>
          {COLS.map((c, i) => (
            <col key={i} style={c.width ? { width: c.width } : undefined} />
          ))}
        </colgroup>
        <thead>
          <tr>
            {COLS.map((c, i) => (
              <th
                key={i}
                className="truncate border-b border-border px-2 py-2 text-[10px] font-semibold uppercase tracking-[0.045em] text-muted"
              >
                {c.short ? (
                  <>
                    <span className="2xl:hidden">{c.short}</span>
                    <span className="hidden 2xl:inline">{c.label}</span>
                  </>
                ) : (
                  c.label
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((b) => (
            <Fragment key={b.asset}>
              <Row b={b} board={board} open={openAsset === b.asset} onToggle={() => onToggle(b.asset)} />
              {openAsset === b.asset && (
                <tr className="border-b border-border bg-[#14141d]">
                  <td colSpan={COLS.length} className="px-4 pt-1 pb-5">
                    <BeliefDetail b={b} />
                  </td>
                </tr>
              )}
            </Fragment>
          ))}
        </tbody>
      </table>
      {rows.length === 0 && <p className="px-3 py-6 text-center text-sm text-muted">No names match these filters.</p>}
    </div>
  );
}
