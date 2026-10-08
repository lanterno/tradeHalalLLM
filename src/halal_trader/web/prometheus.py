"""Prometheus exposition format — zero-dependency exporter.

The web is its own process: everything it exports is read from the
database at scrape time, the one channel it shares with the bot (the
bot's in-process state never reaches it). Exposed gauges:

* ``halal_trader_bot_running`` — 1 / 0, from the bot's heartbeat
* ``halal_trader_heartbeat_age_seconds{component}`` — every component's last beat
* ``halal_trader_drawdown_pct`` / ``halal_trader_portfolio_heat_pct`` — the
  day-trader's last cycle's risk read
* ``halal_trader_llm_spend_today_usd{pool}`` — today's (UTC) LLM spend
* ``halal_trader_account_equity_usd{account}`` /
  ``halal_trader_open_positions{account}`` — from the minute snapshots

Each metric has a ``HELP`` + ``TYPE`` header per Prometheus convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


@dataclass
class MetricSnapshot:
    """One snapshot of an exportable metric (scalar today; no histograms)."""

    name: str
    help_text: str
    value: float
    metric_type: str = "gauge"  # "gauge" | "counter"
    labels: dict[str, str] = field(default_factory=dict)


def render_metrics(snapshots: list[MetricSnapshot]) -> str:
    """Render a list of snapshots as Prometheus exposition format."""
    if not snapshots:
        return ""
    lines: list[str] = []
    seen_headers: set[str] = set()
    for snap in snapshots:
        if snap.name not in seen_headers:
            lines.append(f"# HELP {snap.name} {snap.help_text}")
            lines.append(f"# TYPE {snap.name} {snap.metric_type}")
            seen_headers.add(snap.name)
        if snap.labels:
            label_str = ",".join(f'{k}="{_escape(v)}"' for k, v in sorted(snap.labels.items()))
            lines.append(f"{snap.name}{{{label_str}}} {_format_value(snap.value)}")
        else:
            lines.append(f"{snap.name} {_format_value(snap.value)}")
    return "\n".join(lines) + "\n"


def _escape(value: str) -> str:
    """Escape Prometheus label values per the text-format spec."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_value(value: float) -> str:
    """Render a float without trailing zero noise but with finite precision."""
    if value != value:  # NaN
        return "NaN"
    return f"{value:g}"


async def bot_alive(engine: AsyncEngine) -> bool:
    """The stock bot's liveness from its heartbeat rows (core/heartbeat.py).

    The web is a separate process: ``RuntimeView.bot_running`` is never set
    in it, which is why ``halal_trader_bot_running`` always read 0.
    """
    from halal_trader.core.heartbeat import bot_liveness, read_beats

    try:
        beats = await read_beats(engine)
    except Exception:  # noqa: BLE001 -- cannot tell: report not running
        return False
    now = datetime.now(UTC)
    alive, _ = bot_liveness(beats, now=now, cycles_due=False)
    return alive


async def collect(engine: AsyncEngine) -> list[MetricSnapshot]:
    """Every gauge, read from the database now (see the module docstring)."""
    from halal_trader.core.heartbeat import cycle_risk, read_beats
    from halal_trader.core.llm.spend import POOLS
    from halal_trader.portfolio.snapshots import read_snapshots

    now = datetime.now(UTC)
    out = [
        MetricSnapshot(
            name="halal_trader_bot_running",
            help_text="1 if the stock bot's process heartbeat is fresh, 0 otherwise",
            value=1.0 if await bot_alive(engine) else 0.0,
        )
    ]
    for component, b in sorted((await read_beats(engine)).items()):
        out.append(
            MetricSnapshot(
                name="halal_trader_heartbeat_age_seconds",
                help_text="Seconds since the component last beat",
                value=round(b.age(now).total_seconds(), 1),
                labels={"component": component},
            )
        )
    risk, _ = await cycle_risk(engine)
    for key, name, help_text in (
        ("drawdown_pct", "halal_trader_drawdown_pct", "Day-trader drawdown from peak (fraction)"),
        (
            "portfolio_heat_pct",
            "halal_trader_portfolio_heat_pct",
            "Day-trader unrealized P&L as a fraction of equity",
        ),
    ):
        if risk and risk.get(key) is not None:
            out.append(MetricSnapshot(name=name, help_text=help_text, value=float(risk[key])))
    async with engine.connect() as conn:
        spend = {
            r.consumer: float(r.usd)
            for r in await conn.execute(
                text(
                    "SELECT consumer, sum(spent_usd) AS usd FROM llm_spend "
                    "WHERE day = :d GROUP BY consumer"
                ),
                {"d": now.date()},
            )
        }
    pools: dict[str, float] = {}
    for consumer, usd in spend.items():
        pool = POOLS.get(consumer, consumer)
        pools[pool] = pools.get(pool, 0.0) + usd
    for pool, usd in sorted(pools.items()):
        out.append(
            MetricSnapshot(
                name="halal_trader_llm_spend_today_usd",
                help_text="LLM spend today (UTC) by budget pool",
                value=round(usd, 6),
                labels={"pool": pool},
            )
        )
    for account, snap in sorted((await read_snapshots(engine)).items()):
        labels = {"account": account}
        out.append(
            MetricSnapshot(
                name="halal_trader_account_equity_usd",
                help_text="Account equity at the bot's latest minute snapshot",
                value=snap.equity,
                labels=labels,
            )
        )
        out.append(
            MetricSnapshot(
                name="halal_trader_open_positions",
                help_text="Open positions at the bot's latest minute snapshot",
                value=float(len(snap.positions)),
                labels=labels,
            )
        )
    return out
