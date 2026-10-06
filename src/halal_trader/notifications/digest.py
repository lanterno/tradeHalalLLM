"""The weekly digest: one Telegram message each Friday evening.

Everything the operator would otherwise have to open the dashboard for:
the core account and its forward book against SPUS, the s1 research book,
the core's live-money gate, purification and the next zakat, LLM spend by
budget pool, and whether the evening runs went cleanly. Read-only; every
figure comes from the database the jobs already fill.
"""

from __future__ import annotations

from datetime import date, timedelta
from html import escape
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


async def _first_last(
    engine: AsyncEngine, sql: str, params: dict[str, Any]
) -> tuple[float, float] | None:
    async with engine.connect() as conn:
        rows = (await conn.execute(text(sql), params)).all()
    if len(rows) < 2:
        return None
    return float(rows[0][1]), float(rows[-1][1])


def _pct(pair: tuple[float, float] | None) -> str:
    return f"{pair[1] / pair[0] - 1:+.2%}" if pair else "n/a"


async def build(engine: AsyncEngine, settings: Any, *, today: date) -> str:
    week = today - timedelta(days=7)
    lines = [f"<b>Halal Trader — week to {today:%a %d %b %Y}</b>"]

    core = await _first_last(
        engine,
        "SELECT day, equity FROM broker_equity WHERE account = 'core' AND day >= :d ORDER BY day",
        {"d": week},
    )
    async with engine.connect() as conn:
        core_now = (
            await conn.execute(
                text(
                    "SELECT equity FROM broker_equity WHERE account = 'core' "
                    "ORDER BY day DESC LIMIT 1"
                )
            )
        ).scalar()
    spus = await _first_last(
        engine,
        "SELECT day, close FROM daily_bars WHERE symbol = 'SPUS' AND adjustment = 'all' "
        "AND day >= :d ORDER BY day",
        {"d": week},
    )
    book = await _first_last(
        engine,
        "SELECT day, nav FROM forward_book_days WHERE book = 'core' AND day >= :d ORDER BY day",
        {"d": week},
    )
    s1 = await _first_last(
        engine,
        "SELECT day, nav FROM forward_book_days WHERE book = 's1' AND day >= :d ORDER BY day",
        {"d": week},
    )
    lines.append(
        f"Core account: {f'${core_now:,.0f}' if core_now else 'no equity yet'} "
        f"({_pct(core)} this week) · core book {_pct(book)} · SPUS {_pct(spus)} · s1 {_pct(s1)}"
    )

    async with engine.connect() as conn:
        beats = {
            r.component: r.detail or {}
            for r in await conn.execute(text("SELECT component, detail FROM heartbeats"))
        }
    ready = beats.get("core.readiness", {})
    if ready:
        status = (
            "READY for live keys"
            if ready.get("ready")
            else "not yet: " + "; ".join(ready.get("failures", [])[:3])
        )
        lines.append(f"Live-money gate: {status}")

    execution = beats.get("core.execution", {})
    if execution:
        lines.append(
            f"Core fills (30 days): {execution.get('orders')} orders, "
            f"{execution.get('unfilled')} unfilled; vs arrival "
            f"{execution.get('vs_arrival_bps')} bps, vs close {execution.get('vs_close_bps')} bps "
            f"(book assumes {execution.get('book_cost_bps')})"
        )

    validation = beats.get("screen.validation", {})
    if validation:
        lines.append(
            f"Screen vs SPUS/HLAL: agree on {validation.get('agreement', 0):.0%} of their "
            f"{validation.get('etf_names')} names; large passes neither holds: "
            f"{len(validation.get('large_passes_no_etf_holds') or [])}"
        )
        missing = validation.get("missing_data") or []
        if missing:
            lines.append(
                f"ETF-held names doubtful only for missing data ({len(missing)}): "
                + ", ".join(missing[:10])
            )

    from halal_trader.halal.aaoifi_summary import compute_aaoifi_summary

    summary = await compute_aaoifi_summary(engine)
    lines.append(
        f"Halal: {summary.status}; purification outstanding "
        f"${summary.purification_outstanding_usd:,.2f}"
    )
    if settings.zakat.hawl_hijri:
        from halal_trader.compliance import zakat as z

        _, last = z.hawl_period(z.parse_hawl(settings.zakat.hawl_hijri), today)
        _, nxt = z.hawl_period(z.parse_hawl(settings.zakat.hawl_hijri), last + timedelta(days=360))
        lines.append(f"Next zakat: {nxt} ({z.hijri_label(nxt)}), in {(nxt - today).days} days")

    async with engine.connect() as conn:
        spend = {
            r.consumer: float(r.s)
            for r in await conn.execute(
                text(
                    "SELECT consumer, sum(spent_usd) AS s FROM llm_spend WHERE day >= :m "
                    "GROUP BY consumer"
                ),
                {"m": today.replace(day=1)},
            )
        }
    live = spend.get("stock", 0.0) + spend.get("shadow", 0.0)
    lines.append(
        f"LLM this month: live ${live:.2f} of ${settings.llm.monthly_live_usd:.0f}, research "
        f"${spend.get('research', 0.0):.2f} of ${settings.llm.monthly_research_usd:.0f}"
    )

    nightly, drill = beats.get("backup.nightly"), beats.get("backup.restore_drill")
    async with engine.connect() as conn:
        when = {
            r.component: r.beat_at
            for r in await conn.execute(
                text(
                    "SELECT component, beat_at FROM heartbeats "
                    "WHERE component IN ('backup.nightly', 'backup.restore_drill')"
                )
            )
        }
    if nightly is not None:
        lines.append(
            f"Backup: {when['backup.nightly']:%a %d %b} ({nightly.get('dump_mb')} MB); "
            + (
                f"last restore drill {when['backup.restore_drill']:%d %b}"
                if drill is not None
                else "no restore drill yet"
            )
        )

    research = beats.get("research.daily", {})
    if research:
        refresh = ", ".join(f"{k} {v}" for k, v in (research.get("event_refresh") or {}).items())
        lines.append(
            f"Evening run: {research.get('errors', 0)} error(s) last run"
            + (f"; added {refresh}" if refresh else "")
        )
    return "\n".join(escape(line) if not line.startswith("<b>") else line for line in lines)
