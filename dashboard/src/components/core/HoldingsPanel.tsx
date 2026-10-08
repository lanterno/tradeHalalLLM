import { useMemo, useState } from "react";
import type { CoreHolding, CoreStatus } from "../../api/types";
import { cn, formatPct, formatUsd, formatDay } from "../../lib/utils";
import { sectorColor, toneOf } from "./format";
import { Panel } from "../Panel";

type View = "top" | "sector" | "flat";
type Sort = "weight" | "drift" | "today" | "since";

const SORTS: Record<Sort, (h: CoreHolding) => number> = {
  weight: (h) => -(h.weight ?? 0),
  drift: (h) => -Math.abs(h.drift ?? 0) / (h.band || 1),
  today: (h) => -(h.today ?? -Infinity),
  since: (h) => -(h.since_buy ?? -Infinity),
};

function Toggle<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: [T, string][];
  onChange: (v: T) => void;
}) {
  return (
    <div className="flex overflow-hidden rounded-md border border-border text-xs">
      {options.map(([v, label]) => (
        <button
          key={v}
          type="button"
          onClick={() => onChange(v)}
          className={cn(
            "px-2.5 py-1",
            v === value ? "bg-surface-hover text-white" : "text-muted hover:text-white",
          )}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function SectorStrip({ data }: { data: CoreStatus }) {
  const base = data.equity || 1;
  return (
    <div>
      <div className="flex h-7 overflow-hidden rounded-md">
        {data.sectors.map((s) => (
          <div
            key={s.sector}
            title={`${s.sector}: ${formatPct(s.weight)} · ${s.count} names`}
            className="flex min-w-0 items-center gap-px border-r border-bg/60 last:border-r-0"
            style={{ width: `${((s.value ?? 0) / base) * 100}%`, background: `${sectorColor(s.sector)}33` }}
          >
            <i className="h-full w-1 shrink-0" style={{ background: sectorColor(s.sector) }} />
            <span className="truncate px-1 text-[10px] font-semibold text-white">
              {s.names.slice(0, 3).join(" ")}
            </span>
          </div>
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] tabular-nums text-muted">
        {data.sectors.map((s) => (
          <span key={s.sector} className="flex items-center gap-1.5">
            <i className="h-2 w-2 rounded-sm" style={{ background: sectorColor(s.sector) }} />
            {s.sector} <span className="text-white">{formatPct(s.weight)}</span>
            <span>{s.count}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

function Row({ h, rank, top }: { h: CoreHolding; rank?: number; top: number }) {
  const weight = h.weight ?? 0;
  const target = h.target ?? 0;
  const scale = (v: number) => `${Math.min(v / top, 1) * 100}%`;
  return (
    <tr className={cn("border-t border-border", h.to_sell && "bg-loss/5")}>
      <td className="w-6 py-1.5 pr-1 text-right text-[11px] text-muted">{rank ?? ""}</td>
      <td className="py-1.5 pr-3">
        <span className="font-semibold text-white">{h.symbol}</span>
        <span className="ml-2 hidden max-w-48 truncate align-bottom text-xs text-muted md:inline-block">
          {h.name}
        </span>
      </td>
      <td className="hidden w-[22%] py-1.5 pr-3 lg:table-cell">
        <div className="relative h-2.5 rounded bg-border/60">
          <div
            className="absolute inset-y-0 left-0 rounded"
            style={{ width: scale(weight), background: sectorColor(h.sector) }}
          />
          {target > 0 && (
            <i className="absolute -inset-y-0.5 w-0.5 bg-white" style={{ left: scale(target) }} />
          )}
        </div>
      </td>
      <td className="py-1.5 pr-3 text-right tabular-nums">
        <span className="text-white">{formatPct(h.weight, 2)}</span>
        <p className="text-[10px] text-muted">target {target ? formatPct(target, 2) : "—"}</p>
      </td>
      <td className="hidden py-1.5 pr-3 text-right text-xs tabular-nums sm:table-cell">
        {h.drift == null || h.band == null ? (
          <span className="text-muted">—</span>
        ) : (
          <>
            <span className={h.in_band ? "text-muted" : "text-warning"}>
              {formatPct(h.drift, 2, { signed: true })}
            </span>
            <span className="text-muted"> of ±{formatPct(h.band, 2)}</span>
          </>
        )}
      </td>
      <td className="hidden py-1.5 pr-3 text-right tabular-nums sm:table-cell">{formatUsd(h.value)}</td>
      <td className={cn("py-1.5 pr-3 text-right tabular-nums", toneOf(h.today))}>
        {formatPct(h.today, 2, { signed: true })}
      </td>
      <td className={cn("hidden py-1.5 pr-3 text-right tabular-nums md:table-cell", toneOf(h.since_buy))}>
        {formatPct(h.since_buy, 2, { signed: true })}
      </td>
      <td className="py-1.5 text-right text-xs">
        {h.to_sell ? (
          <span className="rounded bg-loss/15 px-1.5 py-0.5 text-loss">to be sold</span>
        ) : h.verdict === "halal" ? (
          <span className="text-accent">✓</span>
        ) : (
          <span className="text-muted">{h.verdict ?? "—"}</span>
        )}
      </td>
    </tr>
  );
}

function SectorGroup({
  name,
  rows,
  top,
  open,
  onToggle,
  equity,
}: {
  name: string;
  rows: CoreHolding[];
  top: number;
  open: boolean;
  onToggle: () => void;
  equity: number;
}) {
  const value = rows.reduce((s, h) => s + (h.value ?? 0), 0);
  const target = rows.reduce((s, h) => s + (h.target ?? 0), 0);
  const sells = rows.filter((h) => h.to_sell).length;
  const today = rows.reduce((s, h) => s + (h.today ?? 0) * (h.value ?? 0), 0) / (value || 1);
  return (
    <>
      <tr className="cursor-pointer border-t border-border hover:bg-surface-hover" onClick={onToggle}>
        <td className="py-2 pr-1 text-right text-muted">{open ? "▾" : "▸"}</td>
        <td className="whitespace-normal py-2 pr-3">
          <i className="mr-2 inline-block h-2 w-2 rounded-sm" style={{ background: sectorColor(name) }} />
          <span className="font-semibold text-white">{name}</span>
          <span className="ml-2 hidden text-xs text-muted sm:inline">
            {rows.length} · {rows.slice(0, 4).map((h) => h.symbol).join(", ")}
            {rows.length > 4 ? ", …" : ""}
          </span>
        </td>
        <td className="hidden lg:table-cell" />
        <td className="py-2 pr-3 text-right tabular-nums">
          <span className="text-white">{formatPct(value / equity, 2)}</span>
          <p className="text-[10px] text-muted">target {target ? formatPct(target, 2) : "—"}</p>
        </td>
        <td className="hidden py-2 pr-3 text-right text-xs text-muted sm:table-cell">
          {rows.every((h) => h.in_band === null)
            ? "—"
            : rows.every((h) => h.in_band !== false)
              ? "all in band"
              : `${rows.filter((h) => h.in_band === false).length} out of band`}
        </td>
        <td className="hidden py-2 pr-3 text-right tabular-nums sm:table-cell">{formatUsd(value)}</td>
        <td className={cn("py-2 pr-3 text-right tabular-nums", toneOf(today))}>
          {formatPct(today, 2, { signed: true })}
        </td>
        <td className="hidden md:table-cell" />
        <td className="py-2 text-right text-xs">
          {sells ? (
            <span className="rounded bg-loss/15 px-1.5 py-0.5 text-loss">
              {sells} <span className="sm:hidden">sell</span>
              <span className="hidden sm:inline">to be sold</span>
            </span>
          ) : (
            <span className="text-accent">✓ {rows.length}</span>
          )}
        </td>
      </tr>
      {open && rows.map((h) => <Row key={h.symbol} h={h} top={top} />)}
    </>
  );
}

export function HoldingsPanel({ data }: { data: CoreStatus }) {
  const [view, setView] = useState<View>("top");
  const [sort, setSort] = useState<Sort>("weight");
  const [query, setQuery] = useState("");
  const [onlySells, setOnlySells] = useState(false);
  const [open, setOpen] = useState<Record<string, boolean>>({});

  const held = useMemo(() => data.holdings.filter((h) => h.shares || h.target), [data.holdings]);
  const shown = useMemo(() => {
    const q = query.trim().toUpperCase();
    return held
      .filter((h) => !onlySells || h.to_sell)
      .filter(
        (h) =>
          !q ||
          h.symbol.includes(q) ||
          (h.name ?? "").toUpperCase().includes(q) ||
          h.sector.toUpperCase().includes(q),
      )
      .sort((a, b) => SORTS[sort](a) - SORTS[sort](b));
  }, [held, query, onlySells, sort]);
  const top = Math.max(...held.map((h) => Math.max(h.weight ?? 0, h.target ?? 0)), 0.0001);
  const equity = data.equity || 1;
  const noTargets = held.every((h) => !h.target);
  const searching = query.trim() !== "" || onlySells;

  const byWeight = [...shown].sort((a, b) => (b.weight ?? 0) - (a.weight ?? 0));
  const topRows = view === "top" && !searching ? byWeight.slice(0, 10) : [];
  const groupFrom = view === "top" && !searching ? byWeight.slice(10) : shown;
  const groups = new Map<string, CoreHolding[]>();
  for (const h of groupFrom) groups.set(h.sector, [...(groups.get(h.sector) ?? []), h]);
  const groupOrder = [...groups.entries()].sort(
    (a, b) => b[1].reduce((s, h) => s + (h.value ?? 0), 0) - a[1].reduce((s, h) => s + (h.value ?? 0), 0),
  );
  const isOpen = (name: string) => open[name] ?? (view === "sector" || searching);

  return (
    <Panel
      title="Holdings"
      right={
        <>
          {data.positions} names · weight against target
          {noTargets && " · targets come with the core book's first rebalance (tonight's research run)"}
        </>
      }
    >
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={`Search ${held.length} holdings…`}
          className="w-44 rounded-md border border-border bg-bg px-2.5 py-1 text-xs text-white placeholder:text-muted focus:border-muted focus:outline-none"
        />
        <Toggle<View>
          value={view}
          onChange={setView}
          options={[
            ["top", "Top 10 + sectors"],
            ["sector", "All by sector"],
            ["flat", "Flat"],
          ]}
        />
        <Toggle<Sort>
          value={sort}
          onChange={setSort}
          options={[
            ["weight", "Weight"],
            ["drift", "Drift"],
            ["today", "Today"],
            ["since", "Since buy"],
          ]}
        />
        <label className="ml-auto flex items-center gap-1.5 text-xs text-muted">
          <input type="checkbox" checked={onlySells} onChange={(e) => setOnlySells(e.target.checked)} />
          only names to be sold <span className="text-white">{data.to_sell.length}</span>
        </label>
      </div>

      <SectorStrip data={data} />

      {data.to_sell.map((s) => (
        <div
          key={s.symbol}
          className="mt-3 flex flex-wrap items-start gap-x-3 gap-y-1 rounded-lg border border-loss/30 bg-loss/5 px-3 py-2 text-xs"
        >
          <span className="rounded bg-loss/20 px-1.5 py-0.5 font-semibold text-loss">to be sold</span>
          <p className="min-w-0 flex-1">
            <b className="text-white">
              {s.symbol}
              {s.name ? ` · ${s.name}` : ""} fails the {data.screen_as_of ? formatDay(data.screen_as_of) : ""} screen
            </b>
            {s.reason && <span className="text-muted">: {s.reason}</span>}
          </p>
          <span className="text-muted">sold at the next run, {formatDay(data.next_check)} 15:40</span>
        </div>
      ))}

      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-[11px] uppercase tracking-wider text-muted">
            <tr>
              <th />
              <th className="py-1.5">Holding</th>
              <th className="hidden py-1.5 lg:table-cell">Weight vs target</th>
              <th className="py-1.5 pr-3 text-right">Weight</th>
              <th className="hidden py-1.5 pr-3 text-right sm:table-cell">Drift · band</th>
              <th className="hidden py-1.5 pr-3 text-right sm:table-cell">Value</th>
              <th className="py-1.5 pr-3 text-right">Today</th>
              <th className="hidden py-1.5 pr-3 text-right md:table-cell">Since buy</th>
              <th className="py-1.5 text-right">Screen</th>
            </tr>
          </thead>
          <tbody>
            {topRows.length > 0 && (
              <>
                <tr>
                  <td colSpan={9} className="whitespace-normal pt-2 text-[11px] uppercase tracking-wider text-muted">
                    Top 10 · {formatPct(data.top10_weight)} of the portfolio
                  </td>
                </tr>
                {topRows.map((h, i) => (
                  <Row key={h.symbol} h={h} rank={i + 1} top={top} />
                ))}
                <tr>
                  <td colSpan={9} className="whitespace-normal pt-4 text-[11px] uppercase tracking-wider text-muted">
                    The other {groupFrom.length} by sector · tap a sector to open it
                  </td>
                </tr>
              </>
            )}
            {view === "flat" && !searching
              ? shown.map((h, i) => <Row key={h.symbol} h={h} rank={i + 1} top={top} />)
              : groupOrder.map(([name, rows]) => (
                  <SectorGroup
                    key={name}
                    name={name}
                    rows={rows}
                    top={top}
                    equity={equity}
                    open={isOpen(name)}
                    onToggle={() => setOpen((o) => ({ ...o, [name]: !isOpen(name) }))}
                  />
                ))}
          </tbody>
        </table>
        {shown.length === 0 && <p className="py-6 text-center text-sm text-muted">Nothing matches.</p>}
      </div>
      <p className="mt-3 text-[11px] text-muted">
        A holding trades at the monthly rebalance only when it drifts outside its band: a quarter
        of its target, at least 0.2 points. Values{" "}
        {data.equity_source === "live" ? "from the bot's minute snapshot" : "at the last close"}; a
        dash marks a figure not yet known.
      </p>
    </Panel>
  );
}
