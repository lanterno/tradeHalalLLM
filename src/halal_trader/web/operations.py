"""The Operations page: is everything running, will tonight's jobs run, what
does it cost, is the data current -- in one read-only payload.

Every daily job is judged by the trading calendar (core/heartbeat.assess),
not by how long ago it ran; processes by their heartbeat's age; LLM spend
from the meter the caps enforce (llm_spend), not from a log; backups from
the beats the host's scripts write. The web sees none of the bot's
in-process state: the database is the whole picture.
"""

from __future__ import annotations

import calendar
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core import heartbeat as hb
from halal_trader.core.num import ratio
from halal_trader.market_hours import (
    MARKET_OPEN,
    MARKET_TZ,
    effective_close_time,
    is_trading_day,
    next_trading_day,
    previous_trading_day,
)

# The daily jobs, in the order they run.
JOBS: tuple[tuple[str, str], ...] = (
    (hb.RECOMMENDATION, "Stock of the day"),
    (hb.CORE_TRADE, "Core trade"),
    (hb.STOCK_EOD, "End of day"),
    (hb.STOCK_LEDGER, "Broker ledger sync"),
    (hb.WEEKLY_DIGEST, "Weekly digest"),
    (hb.RESEARCH, "Evening research"),
)

# Loops judged by heartbeat age: (component, label, cadence).
PROCESSES: tuple[tuple[str, str, str], ...] = (
    (hb.STOCK_PROCESS, "Stock bot", "beats every 60 s · stale after 3 min"),
    (hb.STOCK_MONITOR, "Stop-loss / take-profit monitor", "ticks every 30 s"),
    (hb.STOCK_CYCLE, "Trading cycle", "every 15 min, 10:00 to the close"),
    (hb.MARKET_SNAPSHOT, "Market snapshot", "every minute, 09:00-17:00 ET on trading days"),
    (hb.SHADOW_PROCESS, "Shadow engine", "beats every 60 s · proposals only, never trades"),
    (hb.WATCHDOG, "Watchdog", "every 5 min · alerts on Telegram"),
)

# The host's backup timer (infra/server/systemd/halabot-backup.timer).
BACKUP_AT_UTC = time(7, 0)
# Tables `just backup` dumps without their rows: market data a run re-fetches
# (the justfile's backup recipe; tests/test_operations.py keeps the two equal).
NOT_DUMPED = frozenset(
    {
        "daily_bars",
        "market_assets",
        "monthly_bars",
        "minute_bars",
        "events",
        "event_facts",
        "event_labels",
        "event_scores",
        "eps_facts",
        "annual_fundamentals",
        "etf_holdings",
    }
)
BIGGEST = 8
LLM_DAYS = 14


def _usd(v: float) -> float:
    """LLM spend: cents are too coarse for a day of a few cents."""
    return round(v, 4)


def _summary(component: str, detail: dict[str, Any] | None) -> str | None:
    """What a job's beat says it did, in a few words."""
    if detail is None:
        return {hb.WEEKLY_DIGEST: "sent", hb.STOCK_EOD: "day closed"}.get(component)
    d = detail
    if component == hb.RECOMMENDATION:
        return f"picked {d['symbol']}" if d.get("symbol") else "no pick"
    if component == hb.STOCK_LEDGER:
        return f"{d.get('broker_fills', 0)} broker fill(s) recorded"
    if component == hb.STOCK_EOD:
        return f"closed {d['date']}" if d.get("date") else "day closed"
    if component == hb.CORE_TRADE:
        return "refused: see the alert" if d.get("refused") else f"ran on {d.get('account')}"
    if component == hb.RESEARCH:
        parts = []
        if d.get("screened"):
            parts.append(f"{d['screened']:,} screened")
        if d.get("rescreen_for"):
            parts.append(f"re-screened {', '.join(d['rescreen_for'][:3])}")
        books = d.get("books") or {}
        parts.append(f"books: {', '.join(books)}" if books else "no forward book advanced")
        errors = d.get("errors") or 0
        parts.append(f"{errors} error(s)" if errors else "no errors")
        return " · ".join(parts)
    scalars = [f"{k} {v}" for k, v in d.items() if isinstance(v, (int, float, str, bool))]
    return " · ".join(scalars[:4]) or None


def _core_summary(run: Any) -> str:
    """The core's run row (core_runs), which says more than its beat."""
    if run.halted:
        return f"halted: {run.halted}"
    kind = "monthly rebalance" if run.monthly else "sells-only day"
    if not run.executed:
        return f"{kind} · planned only"
    screen = f" · screen of {run.screen_as_of:%a %d %b}" if run.screen_as_of else ""
    return f"{kind} · {run.orders} order(s){screen}"


def _jobs(
    beats: dict[str, hb.Beat],
    statuses: dict[str, hb.Status],
    now: datetime,
    *,
    core_run: Any = None,
) -> list[dict[str, Any]]:
    today = now.astimezone(MARKET_TZ).date()
    out = []
    for component, label in JOBS:
        job = hb.DAILY_JOBS[component]
        b = beats.get(component)
        st = statuses.get(component, hb.Status("unknown"))
        on_today = is_trading_day(today) and (job.weekday is None or today.weekday() == job.weekday)
        nxt = hb.next_run(job, now)
        today_at = hb.scheduled_at(job, today) if on_today else None
        summary = _summary(component, b.detail) if b else None
        if component == hb.CORE_TRADE and core_run is not None and b is not None:
            summary = _core_summary(core_run)
        out.append(
            {
                "component": component,
                "label": label,
                "at": job.at.strftime("%H:%M"),
                "weekday": calendar.day_abbr[job.weekday] if job.weekday is not None else None,
                "today_at": today_at.isoformat() if today_at else None,
                "due_today": today_at is not None and today_at <= now,
                "last": b.beat_at.isoformat() if b else None,
                # Today's run, not a catch-up of yesterday's that landed this morning.
                "ran_today": b is not None and today_at is not None and b.beat_at >= today_at,
                "summary": summary,
                "next": nxt.isoformat() if nxt else None,
                "status": st.status,
                "reason": st.reason,
            }
        )
    return out


def _processes(
    beats: dict[str, hb.Beat], statuses: dict[str, hb.Status], now: datetime, *, cycles_due: bool
) -> list[dict[str, Any]]:
    due = {hb.STOCK_CYCLE: cycles_due, hb.MARKET_SNAPSHOT: hb.snapshots_due_at(now)}
    out = []
    for component, label, cadence in PROCESSES:
        b = beats.get(component)
        st = statuses.get(component, hb.Status("unknown"))
        out.append(
            {
                "component": component,
                "label": label,
                "cadence": cadence,
                "due": due.get(component, True),
                "last": b.beat_at.isoformat() if b else None,
                "age_seconds": round(b.age(now).total_seconds(), 1) if b else None,
                "status": st.status,
                "reason": st.reason,
                "detail": b.detail if b else None,
            }
        )
    return out


def _backups(beats: dict[str, hb.Beat], now: datetime) -> dict[str, Any]:
    from halal_trader.research.daily import BACKUP_MAX_AGE_H, RESTORE_DRILL_MAX_AGE_D

    def row(component: str, max_age: timedelta) -> dict[str, Any]:
        b = beats.get(component)
        if b is None:
            return {"at": None, "detail": None, "status": "missing"}
        return {
            "at": b.beat_at.isoformat(),
            "detail": b.detail,
            "status": "stale" if b.age(now) > max_age else "ok",
        }

    next_nightly = datetime.combine(now.astimezone(UTC).date(), BACKUP_AT_UTC, UTC)
    if next_nightly <= now:
        next_nightly += timedelta(days=1)
    # The drill runs with the nightly backup on the 1st (UTC) of each month.
    first = now.astimezone(UTC).date().replace(day=1)
    drill_on = (
        first if next_nightly.date() == first else (first + timedelta(days=32)).replace(day=1)
    )
    return {
        "nightly": row(hb.BACKUP_NIGHTLY, timedelta(hours=BACKUP_MAX_AGE_H)),
        "offsite": row(hb.BACKUP_OFFSITE, timedelta(hours=BACKUP_MAX_AGE_H)),
        "drill": row(hb.BACKUP_RESTORE_DRILL, timedelta(days=RESTORE_DRILL_MAX_AGE_D)),
        "max_age_hours": BACKUP_MAX_AGE_H,
        "drill_max_age_days": RESTORE_DRILL_MAX_AGE_D,
        "next_nightly": next_nightly.isoformat(),
        "next_drill": datetime.combine(drill_on, BACKUP_AT_UTC, UTC).isoformat(),
    }


async def _llm(engine: AsyncEngine, settings: Any, now: datetime) -> dict[str, Any]:
    from halal_trader.core.llm.spend import POOLS

    today = now.astimezone(UTC).date()
    month = today.replace(day=1)
    since = min(month, today - timedelta(days=LLM_DAYS - 1))
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT day, consumer, calls, spent_usd FROM llm_spend "
                    "WHERE day >= :d ORDER BY day"
                ),
                {"d": since},
            )
        ).all()
    llm = settings.llm
    caps = {
        "live": (llm.daily_usd_cap, llm.monthly_live_usd),
        "research": (0.0, llm.monthly_research_usd),
    }
    yesterday = today - timedelta(days=1)
    consumers: dict[str, dict[str, Any]] = {}
    days: dict[date, dict[str, float]] = {
        today - timedelta(days=i): {} for i in range(LLM_DAYS - 1, -1, -1)
    }
    for r in rows:
        c = consumers.setdefault(
            r.consumer,
            {
                "consumer": r.consumer,
                "pool": POOLS.get(r.consumer, r.consumer),
                "today": 0.0,
                "calls_today": 0,
                "yesterday": 0.0,
                "month": 0.0,
                "calls_month": 0,
            },
        )
        spent = float(r.spent_usd)
        if r.day == today:
            c["today"] += spent
            c["calls_today"] += r.calls
        if r.day == yesterday:
            c["yesterday"] += spent
        if r.day >= month:
            c["month"] += spent
            c["calls_month"] += r.calls
        if r.day in days:
            days[r.day][r.consumer] = days[r.day].get(r.consumer, 0.0) + spent

    days_in_month = calendar.monthrange(today.year, today.month)[1]
    pools = []
    for pool, (daily_cap, monthly_cap) in caps.items():
        members = [c for c in consumers.values() if c["pool"] == pool]
        spent_today = sum(c["today"] for c in members)
        spent_month = sum(c["month"] for c in members)
        pools.append(
            {
                "pool": pool,
                "members": sorted(c for c, p in POOLS.items() if p == pool),
                "today": _usd(spent_today),
                "daily_cap": daily_cap or None,
                "month": _usd(spent_month),
                "monthly_cap": monthly_cap or None,
                # The month at this month's pace so far.
                "pace": _usd(spent_month / today.day * days_in_month),
            }
        )
    total_today = sum(c["today"] for c in consumers.values())
    for c in consumers.values():
        for k in ("today", "yesterday", "month"):
            c[k] = _usd(c[k])
    return {
        "day": today.isoformat(),
        "today": _usd(total_today),
        "calls_today": sum(c["calls_today"] for c in consumers.values()),
        "enforced": True,  # every meter enforces (core/llm/spend.py)
        "pools": pools,
        "consumers": sorted(consumers.values(), key=lambda c: -c["month"]),
        "days": [
            {"day": d.isoformat(), **{k: round(v, 4) for k, v in by.items()}}
            for d, by in days.items()
        ],
    }


async def _freshness(engine: AsyncEngine, now: datetime) -> list[dict[str, Any]]:
    from halal_trader.halal import strict
    from halal_trader.portfolio.core_executor import MAX_SCREEN_AGE

    today = now.astimezone(MARKET_TZ).date()
    # The newest close a finished evening run should have stored.
    research_at = hb.scheduled_at(hb.DAILY_JOBS[hb.RESEARCH], today)
    expected_close = (
        today
        if is_trading_day(today) and now >= research_at + hb.DAILY_JOBS[hb.RESEARCH].grace
        else previous_trading_day(today)
    )
    async with engine.connect() as conn:
        spy = (
            await conn.execute(
                text(
                    "SELECT max(day) AS day, max(fetched_at) AS at FROM daily_bars "
                    "WHERE symbol = 'SPY' AND adjustment = 'raw'"
                )
            )
        ).first()
        bars_rows = (
            await conn.execute(
                text("SELECT n_live_tup FROM pg_stat_user_tables WHERE relname = 'daily_bars'")
            )
        ).scalar()
        screen_as_of = await strict.newest_screen(engine)
        screen = (
            (
                await conn.execute(
                    text(
                        "SELECT count(*) AS n, count(*) FILTER (WHERE verdict = 'halal') AS halal, "
                        "max(method) AS method, max(screened_at) AS at "
                        "FROM halal_screen_results WHERE as_of = :d"
                    ),
                    {"d": screen_as_of},
                )
            ).first()
            if screen_as_of
            else None
        )
        sec = (
            await conn.execute(
                text("SELECT max(filed) AS filed, count(DISTINCT cik) AS companies FROM eps_facts")
            )
        ).first()
        snaps = (
            await conn.execute(
                text("SELECT account, taken_at, equity FROM account_snapshots ORDER BY account")
            )
        ).all()
        ledger = (
            await conn.execute(
                text(
                    "SELECT DISTINCT ON (account) account, day, equity, synced_at "
                    "FROM broker_equity ORDER BY account, day DESC"
                )
            )
        ).all()
        minute = (await conn.execute(text("SELECT max(ts) FROM minute_bars"))).scalar()

    def status(ok: bool | None) -> str:
        return "unknown" if ok is None else "ok" if ok else "stale"

    snapshot_due = hb.snapshots_due_at(now)
    out: list[dict[str, Any]] = [
        {
            "name": "Daily bars",
            "as_of": spy.day.isoformat() if spy and spy.day else None,
            "detail": f"~{int(bars_rows or 0):,} rows · SPY's newest close; "
            f"expected {expected_close:%a %d %b}",
            "status": status(spy.day >= expected_close if spy and spy.day else None),
        },
        {
            "name": "Halal screen",
            "as_of": screen_as_of.isoformat() if screen_as_of else None,
            "detail": (
                f"{screen.n:,} names, {screen.halal:,} halal · {screen.method}"
                f" · the core trades on one up to {MAX_SCREEN_AGE.days} days old"
                if screen
                else "no screen stored"
            ),
            "status": status(today - screen_as_of <= MAX_SCREEN_AGE if screen_as_of else None),
        },
        {
            "name": "SEC EDGAR",
            "as_of": sec.filed.isoformat() if sec and sec.filed else None,
            "detail": f"EPS facts for {sec.companies:,} companies · newest filing"
            if sec and sec.companies
            else "no EPS facts stored",
            "status": status(None if not sec or not sec.filed else True),
        },
    ]
    for s in snaps:
        age = now - s.taken_at
        out.append(
            {
                "name": f"Account snapshot · {s.account}",
                "as_of": s.taken_at.isoformat(),
                "detail": f"${float(s.equity):,.2f} · every minute 09:00-17:00 ET",
                "status": status(age <= timedelta(minutes=5) if snapshot_due else True),
            }
        )
    # Alpaca publishes a session's closing equity the next day, so the 16:30
    # sync brings in the session before: once today's sync is due the ledger
    # should reach yesterday's session, before it the one before that.
    ledger_job = hb.DAILY_JOBS[hb.STOCK_LEDGER]
    synced_today = (
        is_trading_day(today) and now >= hb.scheduled_at(ledger_job, today) + ledger_job.grace
    )
    expected_ledger = previous_trading_day(today)
    if not synced_today:
        expected_ledger = previous_trading_day(expected_ledger)
    for r in ledger:
        out.append(
            {
                "name": f"Broker equity · {r.account}",
                "as_of": r.day.isoformat(),
                "detail": f"${float(r.equity):,.2f} · Alpaca's daily ledger; a session's "
                f"close arrives the next day at 16:30, expected {expected_ledger:%a %d %b}",
                "status": status(r.day >= expected_ledger),
            }
        )
    out.append(
        {
            "name": "Minute bars",
            "as_of": minute.isoformat() if minute else None,
            "detail": "backfill only: no job keeps them current",
            "status": "info",
        }
    )
    return out


async def _database(engine: AsyncEngine) -> dict[str, Any]:
    async with engine.connect() as conn:
        head = (
            await conn.execute(
                text(
                    "SELECT pg_database_size(current_database()) AS size, "
                    "current_setting('server_version') AS version, "
                    "(SELECT count(*) FROM pg_stat_user_tables) AS tables, "
                    "(SELECT count(*) FROM pg_stat_activity "
                    " WHERE datname = current_database()) AS connections, "
                    "(SELECT count(*) FROM pg_stat_activity "
                    " WHERE datname = current_database() AND state = 'active') AS active, "
                    "(SELECT age(datfrozenxid) FROM pg_database "
                    " WHERE datname = current_database()) AS xid_age"
                )
            )
        ).one()
        tables = (
            await conn.execute(
                text(
                    "SELECT relname, pg_total_relation_size(relid) AS bytes, n_live_tup, "
                    "n_dead_tup, greatest(last_autovacuum, last_vacuum) AS vacuumed "
                    "FROM pg_stat_user_tables ORDER BY bytes DESC"
                )
            )
        ).all()
        revision = (
            await conn.execute(text("SELECT version_num FROM alembic_version"))
        ).scalar_one_or_none()
        # The shadow engine creates its tables itself (outside Alembic), so a
        # database it has not run against yet has none.
        shadow = (
            (
                await conn.execute(
                    text(
                        "SELECT (ingested_at AT TIME ZONE 'UTC')::date AS day, count(*) AS n "
                        "FROM hb_event_log WHERE ingested_at >= now() - interval '7 days' "
                        "GROUP BY 1 ORDER BY 1"
                    )
                )
            ).all()
            if await conn.scalar(text("SELECT to_regclass('hb_event_log') IS NOT NULL"))
            else []
        )
    live = sum(t.n_live_tup for t in tables)
    dead = sum(t.n_dead_tup for t in tables)
    total = sum(t.bytes for t in tables) or 1

    def table(t: Any) -> dict[str, Any]:
        return {
            "name": t.relname,
            "bytes": int(t.bytes),
            "rows": int(t.n_live_tup),
            "dead": int(t.n_dead_tup),
            "vacuumed": t.vacuumed.isoformat() if t.vacuumed else None,
            "shadow": t.relname.startswith("hb_"),
            "dumped": t.relname not in NOT_DUMPED,
        }

    rest = tables[BIGGEST:]
    return {
        "bytes": int(head.size),
        "version": head.version,
        "tables": int(head.tables),
        "connections": int(head.connections),
        "active": int(head.active),
        "dead_ratio": ratio(dead / (live + dead)) if live + dead else None,
        "xid_age": int(head.xid_age),
        "revision": revision,
        "shadow_share": ratio(sum(t.bytes for t in tables if t.relname.startswith("hb_")) / total),
        "biggest": [table(t) for t in tables[:BIGGEST]],
        "rest": {"count": len(rest), "bytes": int(sum(t.bytes for t in rest))},
        "dead_most": [
            table(t) for t in sorted(tables, key=lambda t: -t.n_dead_tup)[:3] if t.n_dead_tup
        ],
        "shadow_events": [{"day": r.day.isoformat(), "events": r.n} for r in shadow],
    }


def _config(settings: Any, *, core_enabled: bool) -> dict[str, Any]:
    from halal_trader.portfolio import core_executor, strict_core

    core_at = hb.DAILY_JOBS[hb.CORE_TRADE]
    s, llm = settings.stocks, settings.llm
    return {
        "core": [
            ["Core trading", "on" if core_enabled else "off: its Alpaca keys are not set"],
            ["Account", "paper" if settings.core.paper else "LIVE money"],
            ["Holdings targeted", f"{settings.core.top_n} largest that pass the screen"],
            [
                "Rebalance band",
                f"±{strict_core.BAND:.0%} of a holding's target, "
                f"floor {strict_core.BAND_FLOOR:.1%} of equity",
            ],
            [
                "Smallest trade",
                f"${core_executor.MIN_TRADE_FLOOR:,.0f} or "
                f"{core_executor.MIN_TRADE_FRACTION:.2%} of equity",
            ],
            ["Cash left uninvested", f"{core_executor.CASH_BUFFER:.0%} of equity"],
            ["Oldest screen it trades on", f"{core_executor.MAX_SCREEN_AGE.days} days"],
            [
                "Trades at",
                f"{core_at.at:%H:%M} ET ({core_at.early_close_at:%H:%M} on an early close) · "
                "monthly rebalance, forced sales daily",
            ],
        ],
        "llm": [
            ["Model", f"GLM-5.2 · {llm.model}"],
            ["Budget mode", "enforce: calls stop once a cap is spent; exits keep working"],
            ["Daily cap", f"${llm.daily_usd_cap:.2f} a day, live pool"],
            ["LLM_MONTHLY_LIVE_USD", f"${llm.monthly_live_usd:.2f} a month, stock bot + shadow"],
            ["LLM_MONTHLY_RESEARCH_USD", f"${llm.monthly_research_usd:.2f} a month, research"],
            [
                "Watchdog",
                f"every {settings.web.watchdog_interval_seconds // 60} min, alerts on Telegram",
            ],
        ],
        "day_trader": [
            ["Cycle", f"every {s.trading_interval_minutes} min in market hours"],
            ["Largest position", f"{s.max_position_pct:.0%} of equity"],
            ["Positions at once", str(s.max_simultaneous_positions)],
            ["Daily loss limit", f"{s.daily_loss_limit:.0%} from the day's first equity"],
            ["Daily return target", f"{s.daily_return_target:.0%}"],
            ["Max drawdown", f"{s.max_drawdown_pct:.0%}"],
            ["Monitor", f"every {s.monitor_interval_seconds:.0f} s: stops, targets, trailing"],
            [
                "News entries",
                f"{s.reactor_entry_size_fraction:.0%} size, trailing stop "
                f"{s.reactor_trailing_stop_distance_pct:.0%}",
            ],
        ],
    }


def _market(now: datetime) -> dict[str, Any]:
    today = now.astimezone(MARKET_TZ).date()
    trading = is_trading_day(today)
    opens = datetime.combine(today, MARKET_OPEN, MARKET_TZ) if trading else None
    closes = datetime.combine(today, effective_close_time(today), MARKET_TZ) if trading else None
    is_open = bool(opens and closes and opens <= now < closes)
    next_day = today if trading and opens and now < opens else next_trading_day(today)
    return {
        "today": today.isoformat(),
        "trading_day": trading,
        "open": is_open,
        "opens": opens.isoformat() if opens else None,
        "closes": closes.isoformat() if closes else None,
        "next_open": datetime.combine(next_day, MARKET_OPEN, MARKET_TZ).isoformat(),
    }


async def build(
    engine: AsyncEngine,
    settings: Any,
    *,
    web_started: datetime | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    from halal_trader import __version__
    from halal_trader.core.halt import get_status
    from halal_trader.db.admin import head as schema_head
    from halal_trader.portfolio.core_account import core_account

    now = now or datetime.now(UTC)
    beats = await hb.read_beats(engine)
    cycles_due = hb.cycles_due_at(now)
    statuses = hb.assess(beats, now=now, cycles_due=cycles_due)
    alive, why = hb.bot_liveness(beats, now=now, cycles_due=cycles_due)
    async with engine.connect() as conn:
        core_run = (
            await conn.execute(
                text(
                    "SELECT monthly, executed, orders, halted, screen_as_of FROM core_runs "
                    "WHERE account = :a ORDER BY recorded_at DESC LIMIT 1"
                ),
                {"a": core_account(settings.core.paper)},
            )
        ).first()
    jobs = _jobs(beats, statuses, now, core_run=core_run)
    processes = _processes(beats, statuses, now, cycles_due=cycles_due)
    backups = _backups(beats, now)
    database = await _database(engine)
    head = schema_head()

    problems = [
        f"{p['label']}: {p['reason']}" for p in processes if p["status"] in ("stale", "missing")
    ]
    problems += [f"{j['label']}: {j['reason']}" for j in jobs if j["status"] == "stale"]
    problems += [
        f"{what} {b['status']}"
        for what, b in (
            ("nightly backup", backups["nightly"]),
            ("off-site copy", backups["offsite"]),
        )
        if b["status"] != "ok"
    ]
    # The evening run's reading of what the LLM account can still pay for,
    # judged by the same rule its alert uses.
    from halal_trader.core.llm.credits import LLM_CREDITS, Credits

    llm = await _llm(engine, settings, now)
    credit = beats.get(LLM_CREDITS)
    llm["credits"] = None
    if credit is not None and credit.detail:
        d = credit.detail
        reading = Credits(
            d.get("balance_usd"),
            d.get("key_remaining_usd"),
            float(d.get("pace_usd_per_day") or 0.0),
            str(d.get("checked_at") or credit.beat_at.isoformat()),
        )
        problem = reading.problem()
        llm["credits"] = {
            "balance_usd": reading.balance_usd,
            "key_remaining_usd": reading.key_remaining_usd,
            "available_usd": reading.available_usd,
            "days_left": reading.days_left,
            "pace_usd_per_day": reading.pace_usd_per_day,
            "at": credit.beat_at.isoformat(),
            "low": problem is not None,
        }
        if problem:
            problems.append(problem)
    watchdog = beats.get(hb.WATCHDOG)
    return {
        "now": now.isoformat(),
        "paper": settings.core.paper,
        "market": _market(now),
        "fleet": {
            "verdict": "down" if not alive else "degraded" if problems else "healthy",
            "alive": alive,
            "reason": why,
            "beating": sum(1 for p in processes if p["status"] == "ok" and p["due"]),
            "jobs_due": sum(1 for j in jobs if j["due_today"]),
            "jobs_ran": sum(1 for j in jobs if j["ran_today"]),
            "problems": problems,
            "watchdog": watchdog.detail if watchdog else None,
        },
        "halt": (await get_status(engine)).to_json(),
        "jobs": jobs,
        "processes": processes,
        "llm": llm,
        "backups": backups,
        "freshness": await _freshness(engine, now),
        "database": database,
        "deploy": {
            "version": __version__,
            "revision": database["revision"],
            "expected_revision": head,
            "schema_ok": database["revision"] == head,
            "web_started": web_started.isoformat() if web_started else None,
        },
        "config": _config(settings, core_enabled=hb.core_on(beats)),
    }
