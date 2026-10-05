import { Link } from "react-router-dom";
import type { HomeStatus } from "../../api/types";
import { useNow } from "../../hooks/useNow";
import { formatIn } from "../../lib/marketClock";
import { cn } from "../../lib/utils";

const NY = "America/New_York";

/** The day and time of an instant in New York, which every job is scheduled in. */
function when(iso: string, now: Date): string {
  const at = new Date(iso);
  const nyDate = (d: Date) => d.toLocaleDateString("en-CA", { timeZone: NY });
  const day =
    nyDate(at) === nyDate(now)
      ? "Today"
      : at.toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short", timeZone: NY });
  // Midnight items (the hawl) are a date, not a time.
  const time = at.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: NY });
  return time === "00:00" ? day : `${day} ${time}`;
}

export function UpcomingPanel({ items }: { items: HomeStatus["upcoming"] }) {
  const now = useNow(30_000);
  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <h2 className="mb-3 flex items-baseline justify-between text-xs font-semibold uppercase tracking-widest text-muted">
        Coming up
        <span className="text-xs font-normal normal-case tracking-normal">all automatic · New York times</span>
      </h2>
      <div>
        {items.slice(0, 6).map((item, i) => (
          <div
            key={`${item.label}-${item.at}`}
            className={cn("grid grid-cols-[92px_1fr_auto] items-baseline gap-2.5 py-2", i > 0 && "border-t border-border")}
          >
            <span className="text-xs text-muted">{when(item.at, now)}</span>
            <span className="text-sm">
              {item.label} <span className="text-muted">· {item.detail}</span>
            </span>
            <span className="whitespace-nowrap text-xs text-blue-400">{formatIn(item.at, now)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Check({ ok, children }: { ok: boolean; children: React.ReactNode }) {
  return (
    <div className={cn("flex gap-2", ok ? "text-accent" : "text-muted")}>
      <span>{ok ? "✓" : "○"}</span>
      <span>{children}</span>
    </div>
  );
}

export function GatePanel({ gate }: { gate: HomeStatus["gate"] }) {
  const te = gate.tracking_error;
  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <h2 className="mb-3 flex items-baseline justify-between text-xs font-semibold uppercase tracking-widest text-muted">
        Live-money gate
        <Link to="/core" className="text-xs font-normal normal-case tracking-normal hover:text-white">
          details →
        </Link>
      </h2>
      <p className="text-sm">
        <b className="tabular-nums text-white">
          {gate.days} of {gate.min_days}
        </b>{" "}
        <span className="text-muted">trading days on paper</span>
        {gate.ready && <span className="ml-2 font-semibold text-accent">Ready for live keys</span>}
      </p>
      <div className="my-2 h-2 overflow-hidden rounded bg-border">
        <i
          className={cn("block h-full", gate.ready ? "bg-accent" : "bg-warning")}
          style={{ width: `${Math.max(2, Math.min(100, (gate.days / gate.min_days) * 100))}%` }}
        />
      </div>
      <div className="mt-2.5 grid gap-1.5 text-xs">
        <Check ok={gate.monthly_runs > 0}>A monthly rebalance executed</Check>
        <Check ok={te !== null && te <= gate.max_tracking_error}>
          Follows its forward book (tracking error {te === null ? "not yet measurable" : `${(te * 100).toFixed(1)}%`}, limit{" "}
          {(gate.max_tracking_error * 100).toFixed(0)}%)
        </Check>
        <Check ok={gate.gap !== null && Math.abs(gate.gap) <= gate.max_gap}>
          No cumulative gap to the book ({gate.gap === null ? "not yet measurable" : `${(gate.gap * 100).toFixed(2)}%`}, limit ±
          {(gate.max_gap * 100).toFixed(0)}%)
        </Check>
        <Check ok={gate.clean}>Clean: no refused orders, no halted runs</Check>
      </div>
    </div>
  );
}
