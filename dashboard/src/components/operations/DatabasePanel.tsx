import type { ReactNode } from "react";
import type { OperationsStatus, OpsBackup } from "../../api/types";
import { cn, formatBytes, formatDate, formatDayTime, formatPct, formatTime } from "../../lib/utils";
import { Panel } from "../Panel";
import { ago, until } from "./format";

function Stat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="rounded-lg border border-border p-3">
      <p className="text-[10px] uppercase tracking-wider text-muted">{label}</p>
      <p className="mt-0.5 text-lg font-bold tabular-nums text-white">{value}</p>
    </div>
  );
}

function Row({ k, v, tone }: { k: string; v: ReactNode; tone?: string }) {
  return (
    <div className="flex justify-between gap-3 py-0.5">
      <span className="text-muted">{k}</span>
      <span className={cn("text-right tabular-nums", tone ?? "text-white")}>{v}</span>
    </div>
  );
}

const backupTone = (b: OpsBackup) => (b.status === "ok" ? undefined : "text-loss");

export function DatabasePanel({ data }: { data: OperationsStatus }) {
  const db = data.database;
  const b = data.backups;
  const peak = Math.max(...db.biggest.map((t) => t.bytes), 1);
  const eventsPeak = Math.max(...db.shadow_events.map((d) => d.events), 1);
  const dumpMb = b.nightly.detail?.dump_mb as number | undefined;
  const liveKb = b.nightly.detail?.live_events_kb as number | undefined;
  return (
    <Panel title="Database & backups" right={`Postgres ${db.version.split(" ")[0]}`}>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="Size" value={formatBytes(db.bytes)} />
        <Stat label="Tables" value={db.tables} />
        <Stat
          label="Connections"
          value={
            <>
              {db.connections} <span className="text-xs font-normal text-muted">{db.active} active</span>
            </>
          }
        />
        <Stat label="Dead rows" value={formatPct(db.dead_ratio, 1)} />
      </div>

      <p className="mt-4 text-[11px] uppercase tracking-wider text-muted">
        Biggest tables{" "}
        <span className="normal-case tracking-normal">
          · shadow engine's in orange: {formatPct(db.shadow_share, 1)} of the database
        </span>
      </p>
      <ul className="mt-1 space-y-1 text-xs">
        {db.biggest.map((t) => (
          <li key={t.name} className="grid grid-cols-[minmax(0,9rem)_1fr_4rem] items-center gap-2">
            <span className="truncate font-mono text-white" title={t.name}>
              {t.name}
            </span>
            <span className="flex min-w-0 items-center gap-2">
              <span className="hidden truncate text-[10px] text-muted sm:inline">
                {t.rows.toLocaleString()} rows{t.dumped ? "" : " · not in dump"}
              </span>
              <span className="h-1.5 flex-1 rounded bg-border">
                <span
                  className={cn("block h-1.5 rounded", t.shadow ? "bg-orange-400" : "bg-muted")}
                  style={{ width: `${(t.bytes / peak) * 100}%` }}
                />
              </span>
            </span>
            <span className="text-right tabular-nums text-white">{formatBytes(t.bytes)}</span>
          </li>
        ))}
        {db.rest.count > 0 && (
          <li className="grid grid-cols-[minmax(0,9rem)_1fr_4rem] gap-2 text-muted">
            <span>{db.rest.count} other tables</span>
            <span />
            <span className="text-right tabular-nums">{formatBytes(db.rest.bytes)}</span>
          </li>
        )}
      </ul>
      <p className="mt-1 text-[10px] text-muted">
        "Not in dump": market data the evening run re-fetches, left out of the nightly backup.
      </p>

      <div className="mt-4 grid gap-4 text-xs sm:grid-cols-2">
        <div>
          <p className="mb-1 text-[11px] uppercase tracking-wider text-muted">Vacuum</p>
          {db.dead_most.map((t) => (
            <Row
              key={t.name}
              k={t.name}
              v={`${t.dead.toLocaleString()} dead · ${t.vacuumed ? formatTime(t.vacuumed) : "never vacuumed"}`}
            />
          ))}
          {db.dead_most.length === 0 && <Row k="dead rows" v="none" />}
          <Row
            k="transaction-ID age"
            v={`${(db.xid_age / 1e6).toFixed(1)}M of 200M`}
            tone={db.xid_age > 150e6 ? "text-loss" : undefined}
          />
        </div>
        <div>
          <p className="mb-1 text-[11px] uppercase tracking-wider text-muted">
            Shadow events written per day
          </p>
          {db.shadow_events.length ? (
            <div className="flex h-14 items-end gap-1">
              {db.shadow_events.map((d) => (
                <div
                  key={d.day}
                  className="flex-1 rounded-t bg-orange-400/70"
                  style={{ height: `${(d.events / eventsPeak) * 100}%` }}
                  title={`${formatDate(d.day)}: ${d.events.toLocaleString()} events`}
                />
              ))}
            </div>
          ) : (
            <p className="text-muted">none in the last 7 days</p>
          )}
        </div>
      </div>

      <div className="mt-4 grid gap-4 text-xs sm:grid-cols-2">
        <div>
          <p className="mb-1 text-[11px] uppercase tracking-wider text-muted">Nightly backup</p>
          <Row k="last" v={b.nightly.at ? ago(b.nightly.at, data.now) : "never recorded"} tone={backupTone(b.nightly)} />
          {dumpMb != null && <Row k="dump" v={`${formatBytes(dumpMb * 1024 * 1024)}, verified`} />}
          {liveKb != null && <Row k="live events" v={formatBytes(liveKb * 1024)} />}
          <Row
            k="off-site"
            v={b.offsite.at ? `${ago(b.offsite.at, data.now)} · ${String(b.offsite.detail?.snapshot ?? "")}` : "never recorded"}
            tone={backupTone(b.offsite)}
          />
          <Row k="next" v={`${formatDayTime(b.next_nightly)} ET · ${until(b.next_nightly, data.now)}`} />
          <Row k="alerts if" v={`no dump or copy in ${b.max_age_hours} h`} tone="text-muted" />
        </div>
        <div>
          <p className="mb-1 text-[11px] uppercase tracking-wider text-muted">Restore drill</p>
          <Row k="last" v={b.drill.at ? formatTime(b.drill.at) : "never recorded"} tone={backupTone(b.drill)} />
          {b.drill.at && <Row k="result" v={b.drill.detail?.ok ? "passed" : "failed"} />}
          <Row k="checks" v="every table back, schema match" />
          <Row k="next" v={`${formatDayTime(b.next_drill)} ET`} />
          <Row k="alerts if" v={`no drill in ${b.drill_max_age_days} days`} tone="text-muted" />
        </div>
      </div>
    </Panel>
  );
}

export function ConfigPanel({ data }: { data: OperationsStatus }) {
  const groups: [string, [string, string][]][] = [
    ["Core portfolio", data.config.core],
    ["LLM & alerts", data.config.llm],
    ["Day-trader", data.config.day_trader],
  ];
  return (
    <Panel title="Configuration" right="read-only · change it in .env and recreate the container">
      <div className="grid gap-6 lg:grid-cols-3">
        {groups.map(([title, rows]) => (
          <div key={title}>
            <p className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-muted">{title}</p>
            <dl className="divide-y divide-border text-xs">
              {rows.map(([k, v]) => (
                <div key={k} className="grid grid-cols-[minmax(0,10rem)_1fr] gap-3 py-1.5">
                  <dt className={cn("text-muted", /^[A-Z_]+$/.test(k) && "font-mono text-[11px]")}>{k}</dt>
                  <dd className="text-white">{v}</dd>
                </div>
              ))}
            </dl>
          </div>
        ))}
      </div>
      <p className="mt-3 text-[11px] text-muted">
        Names in capitals are environment variables; everything else is a decided value in
        config.py. Secrets never reach this page.
      </p>
    </Panel>
  );
}
