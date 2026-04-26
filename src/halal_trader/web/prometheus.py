"""Prometheus exposition format — zero-dependency exporter.

The full ``prometheus_client`` library is overkill for a single-user
bot: we have a small fixed set of metrics, all read out of the DB or
in-memory state at scrape time. Generating the line-format response by
hand keeps deployment dependency-free.

Exposed metrics (all snapshot-style — Prometheus scrapes; we don't
push):

* ``halal_trader_cycle_latency_ms`` — last cycle's elapsed time
* ``halal_trader_llm_cost_today_usd`` — running spend total
* ``halal_trader_llm_cache_read_ratio`` — cache_read / total input
* ``halal_trader_open_positions`` — count per asset class
* ``halal_trader_drawdown_pct`` — current drawdown from peak
* ``halal_trader_bot_running`` — 1 / 0 liveness flag

Each metric has a ``HELP`` + ``TYPE`` header per Prometheus convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


def collect_default_snapshots(state: dict[str, Any]) -> list[MetricSnapshot]:
    """Pull standard halal-trader metrics out of the dashboard ``app_state``.

    The state dict is the single source of in-process truth — populated
    by the cycle, monitor, and analytics surfaces. This collector is the
    only place that knows the dict's shape; if a metric is absent we
    skip it rather than fabricate a zero so Prometheus alerting can
    detect the gap explicitly.
    """
    out: list[MetricSnapshot] = []

    if "bot_running" in state:
        out.append(
            MetricSnapshot(
                name="halal_trader_bot_running",
                help_text="1 if the bot's main loop is running, 0 otherwise",
                value=1.0 if state["bot_running"] else 0.0,
            )
        )

    risk = state.get("risk_state") or {}
    if isinstance(risk, dict):
        if "drawdown_pct" in risk and risk["drawdown_pct"] is not None:
            out.append(
                MetricSnapshot(
                    name="halal_trader_drawdown_pct",
                    help_text="Current portfolio drawdown from peak (fraction)",
                    value=float(risk["drawdown_pct"]),
                )
            )
        if "portfolio_heat_pct" in risk and risk["portfolio_heat_pct"] is not None:
            out.append(
                MetricSnapshot(
                    name="halal_trader_portfolio_heat_pct",
                    help_text="Portfolio unrealized P&L as fraction of equity",
                    value=float(risk["portfolio_heat_pct"]),
                )
            )

    if "last_cycle_latency_ms" in state:
        out.append(
            MetricSnapshot(
                name="halal_trader_cycle_latency_ms",
                help_text="Most recent cycle wall-clock duration in milliseconds",
                value=float(state["last_cycle_latency_ms"]),
            )
        )

    if "llm_cost_today_usd" in state:
        out.append(
            MetricSnapshot(
                name="halal_trader_llm_cost_today_usd",
                help_text="Running cumulative LLM spend for the current UTC day",
                value=float(state["llm_cost_today_usd"]),
            )
        )

    if "llm_cache_read_ratio" in state:
        out.append(
            MetricSnapshot(
                name="halal_trader_llm_cache_read_ratio",
                help_text="Fraction of input tokens served from prompt cache",
                value=float(state["llm_cache_read_ratio"]),
            )
        )

    open_positions = state.get("open_positions_by_asset") or {}
    if isinstance(open_positions, dict):
        for asset_class, count in open_positions.items():
            out.append(
                MetricSnapshot(
                    name="halal_trader_open_positions",
                    help_text="Open position count per asset class",
                    value=float(count),
                    labels={"asset_class": str(asset_class)},
                )
            )

    return out
