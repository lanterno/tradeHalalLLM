import type { ReactNode } from "react";
import { cn } from "../lib/utils";

/** A titled card: the building block of the Core and Operations pages. */
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
