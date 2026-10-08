import { cn } from "../../lib/utils";

const STYLES: Record<string, [string, string]> = {
  ok: ["bg-accent/10 text-accent border-accent/30", "✓ OK"],
  stale: ["bg-loss/10 text-loss border-loss/30", "! stale"],
  missing: ["bg-loss/10 text-loss border-loss/30", "! missing"],
  disabled: ["bg-border text-muted border-border", "off"],
  unknown: ["bg-sky-500/10 text-sky-300 border-sky-500/30", "○ first run"],
  idle: ["bg-border text-muted border-border", "◑ idle"],
  pending: ["bg-sky-500/10 text-sky-300 border-sky-500/30", "○ to come"],
  info: ["bg-border text-muted border-border", "info"],
};

/** A small pill for a heartbeat verdict. */
export function StatusBadge({ status, title }: { status: string; title?: string | null }) {
  const [style, label] = STYLES[status] ?? STYLES.info;
  return (
    <span
      title={title ?? undefined}
      className={cn("whitespace-nowrap rounded border px-1.5 py-0.5 text-[10px] font-semibold", style)}
    >
      {label}
    </span>
  );
}

const DOTS: Record<string, string> = {
  ok: "bg-accent",
  stale: "bg-loss",
  missing: "bg-loss",
  unknown: "bg-muted",
  info: "bg-muted",
};

export function Dot({ status }: { status: string }) {
  return <i className={cn("inline-block h-2 w-2 shrink-0 rounded-full", DOTS[status] ?? "bg-warning")} />;
}
