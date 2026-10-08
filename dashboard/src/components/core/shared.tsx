import type { ReactNode } from "react";
import { cn, formatPct } from "../../lib/utils";
import { toneOf } from "./format";

export function Panel({
  title,
  right,
  children,
  className,
}: {
  title: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("rounded-xl border border-border bg-surface p-4", className)}>
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h2 className="text-[11px] font-semibold uppercase tracking-widest text-muted">{title}</h2>
        {right && <div className="text-xs text-muted">{right}</div>}
      </div>
      {children}
    </section>
  );
}

export function SignedPct({ v, digits = 2 }: { v: number | null | undefined; digits?: number }) {
  return <span className={toneOf(v)}>{formatPct(v, digits, { signed: true })}</span>;
}
