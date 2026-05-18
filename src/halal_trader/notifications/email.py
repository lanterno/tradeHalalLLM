"""Operator email notifications — end-of-day summary + weekly digest.

Round-4 wave 5.I: replaces the operator's daily login routine with a
beautifully-formatted HTML email at market close + a Sunday weekly
digest. Composed of:

* P&L summary (today / month / quarter)
* Trade roster (most-recent N filled, with side / price / status)
* Halal-compliance tile (the same `AAOIFISummary` the dashboard
  uses — single source of truth)
* Purification status (accrued, disbursed, outstanding)
* Risk snapshot (drawdown, portfolio heat, halt status)

Design notes:

* **Pure-Python composition**: this module renders the HTML. The
  actual SMTP send is in the Wave-5.I sender extension (TBD: SES
  / SendGrid / SMTP). Renderer-as-a-pure-function is testable
  with no I/O.
* **Inline CSS**: every email client interprets `<style>` blocks
  differently. Inline styles are the only thing that's reliable
  across Gmail / Outlook / Apple Mail.
* **Plaintext fallback**: every HTML email gets a plaintext twin
  so spam filters (and people on terminal mail clients) can read
  the content too.
* **Localised numbers**: thousands-separators + 2dp money
  formatting throughout, no ambiguous bare numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


# Status → (background-color, label) for the halal-compliance banner.
_STATUS_COLORS = {
    "compliant": ("#16a34a", "✓ Compliant"),
    "attention": ("#d97706", "⚠ Attention"),
    "violation": ("#dc2626", "✗ Violation"),
}


@dataclass
class EmailSummaryInput:
    """All the data the renderer needs in one bag.

    Composed by the caller from per-source queries (P&L repo, trade
    repo, AAOIFI summary, risk-state push). The renderer is pure —
    given this input, it returns deterministic HTML + plaintext.
    """

    operator_name: str
    period_label: str  # "Daily" or "Weekly"
    as_of: datetime  # When the summary was computed (UTC)

    # P&L
    today_pnl_usd: float = 0.0
    today_return_pct: float = 0.0
    month_pnl_usd: float = 0.0
    quarter_pnl_usd: float = 0.0

    # Trades
    trades_today: int = 0
    trades_this_week: int = 0
    recent_trades: list[dict[str, Any]] = field(default_factory=list)

    # Halal compliance (from AAOIFISummary)
    compliance_status: str = "compliant"  # compliant | attention | violation
    non_halal_fills_quarter: int = 0
    purification_accrued_usd: float = 0.0
    purification_outstanding_usd: float = 0.0

    # Risk
    drawdown_pct: float | None = None
    portfolio_heat_pct: float | None = None
    is_halted: bool = False
    halt_reason: str = ""

    # Optional notes (e.g. "shadow runner diverged 50bps this week")
    notable_events: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RenderedEmail:
    """Rendered subject + bodies. Caller hands to the SMTP layer."""

    subject: str
    html_body: str
    text_body: str


# ── Subject line ──────────────────────────────────────────


def render_subject(data: EmailSummaryInput) -> str:
    """Generate a one-line subject. Status emoji prefix lets operators
    triage their inbox at a glance."""
    emoji = {"compliant": "🟢", "attention": "🟠", "violation": "🔴"}.get(
        data.compliance_status, "⚪"
    )
    pnl_part = _money(data.today_pnl_usd)
    date_str = data.as_of.strftime("%Y-%m-%d")
    return f"{emoji} {data.period_label} summary {date_str} — {pnl_part}"


# ── HTML rendering ────────────────────────────────────────


def _money(amount: float) -> str:
    """`+$1,234.56` / `-$1,234.56` for P&L; clamps at 2dp."""
    sign = "+" if amount >= 0 else "-"
    return f"{sign}${abs(amount):,.2f}"


def _pct(p: float) -> str:
    """`+1.23%` / `-1.23%`."""
    sign = "+" if p >= 0 else ""
    return f"{sign}{p * 100:.2f}%"


def _bare_money(amount: float) -> str:
    """`$1,234.56` (no sign — for non-P&L amounts like purification owed)."""
    return f"${amount:,.2f}"


def _trade_row_html(trade: dict[str, Any]) -> str:
    """One row in the recent-trades table."""
    ts = str(trade.get("timestamp", ""))[:19]
    sym = trade.get("symbol") or trade.get("pair") or "?"
    side = (trade.get("side") or "").lower()
    side_color = "#16a34a" if side == "buy" else "#dc2626"
    qty = trade.get("quantity", 0) or trade.get("filled_quantity", 0) or 0
    price = trade.get("price") or trade.get("filled_price") or 0
    status = trade.get("status", "")
    return (
        f"<tr>"
        f'<td style="padding:6px 12px;color:#6b7280;font-size:13px;">{ts}</td>'
        f'<td style="padding:6px 12px;font-weight:600;">{sym}</td>'
        f'<td style="padding:6px 12px;color:{side_color};text-transform:uppercase;'
        f'font-size:12px;font-weight:700;">{side}</td>'
        f'<td style="padding:6px 12px;text-align:right;">{qty}</td>'
        f'<td style="padding:6px 12px;text-align:right;">{_bare_money(float(price or 0))}</td>'
        f'<td style="padding:6px 12px;color:#6b7280;font-size:12px;">{status}</td>'
        f"</tr>"
    )


def render_html(data: EmailSummaryInput) -> str:
    """Render the full HTML email body. Inline styles only.

    Pinned by tests: the rendered output contains every section
    label so a regression that drops a section is caught immediately.
    """
    color, status_label = _STATUS_COLORS.get(
        data.compliance_status, ("#6b7280", data.compliance_status)
    )
    pnl_color = "#16a34a" if data.today_pnl_usd >= 0 else "#dc2626"

    notable_html = ""
    if data.notable_events:
        items = "".join(f"<li>{e}</li>" for e in data.notable_events)
        notable_html = (
            f'<h3 style="margin:24px 0 8px;color:#111;">Notable</h3>'
            f'<ul style="color:#374151;line-height:1.6;">{items}</ul>'
        )

    halt_html = ""
    if data.is_halted:
        halt_html = (
            f'<div style="margin:16px 0;padding:12px;background:#fee2e2;'
            f'border-left:4px solid #dc2626;color:#7f1d1d;">'
            f"<strong>HALT ACTIVE</strong>: {data.halt_reason or '(no reason given)'}"
            f"</div>"
        )

    risk_rows: list[str] = []
    if data.drawdown_pct is not None:
        risk_rows.append(
            f'<tr><td style="padding:4px 0;color:#6b7280;">Drawdown</td>'
            f'<td style="padding:4px 0;text-align:right;font-weight:600;">'
            f"{_pct(data.drawdown_pct)}</td></tr>"
        )
    if data.portfolio_heat_pct is not None:
        risk_rows.append(
            f'<tr><td style="padding:4px 0;color:#6b7280;">Portfolio heat</td>'
            f'<td style="padding:4px 0;text-align:right;font-weight:600;">'
            f"{_pct(data.portfolio_heat_pct)}</td></tr>"
        )
    risk_table = (
        f'<table style="width:100%;margin-top:8px;">{"".join(risk_rows)}</table>'
        if risk_rows
        else ""
    )

    if data.recent_trades:
        trade_body = "".join(_trade_row_html(t) for t in data.recent_trades)
    else:
        trade_body = (
            '<tr><td colspan="6" style="padding:12px;text-align:center;color:#9ca3af;">'
            "No trades in this period.</td></tr>"
        )

    # The HTML template below has a few long lines (inline-styled
    # tags); each-tag-on-its-own-line would hurt readability without
    # changing meaning. Disable line-length for this f-string.
    return f"""<!DOCTYPE html>
<html><body style="margin:0;padding:24px;background:#f9fafb;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#111;">
<div style="max-width:640px;margin:0 auto;background:#fff;border-radius:8px;padding:24px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">

<h1 style="margin:0 0 4px;font-size:20px;color:#111;">{data.period_label} summary</h1>
<p style="margin:0 0 24px;color:#6b7280;font-size:14px;">
Hi {data.operator_name} — here's your halal-trading update for {data.as_of.strftime("%A, %B %-d, %Y")}.
</p>

<div style="display:inline-block;margin-bottom:16px;padding:6px 12px;background:{color};color:#fff;border-radius:4px;font-weight:600;font-size:13px;">
{status_label}
</div>

{halt_html}

<h2 style="margin:16px 0 8px;font-size:16px;color:#111;">P&amp;L</h2>
<table style="width:100%;border-collapse:collapse;">
  <tr>
    <td style="padding:4px 0;color:#6b7280;">Today</td>
    <td style="padding:4px 0;text-align:right;font-weight:700;color:{pnl_color};">{_money(data.today_pnl_usd)} ({_pct(data.today_return_pct)})</td>
  </tr>
  <tr>
    <td style="padding:4px 0;color:#6b7280;">This month</td>
    <td style="padding:4px 0;text-align:right;font-weight:600;">{_money(data.month_pnl_usd)}</td>
  </tr>
  <tr>
    <td style="padding:4px 0;color:#6b7280;">This quarter</td>
    <td style="padding:4px 0;text-align:right;font-weight:600;">{_money(data.quarter_pnl_usd)}</td>
  </tr>
</table>

<h2 style="margin:24px 0 8px;font-size:16px;color:#111;">Trades</h2>
<p style="margin:0 0 12px;color:#6b7280;font-size:14px;">
{data.trades_today} today · {data.trades_this_week} this week
</p>
<table style="width:100%;border-collapse:collapse;font-size:14px;">
<thead>
<tr style="background:#f3f4f6;color:#374151;font-size:12px;text-transform:uppercase;">
<th style="padding:6px 12px;text-align:left;">Time</th>
<th style="padding:6px 12px;text-align:left;">Symbol</th>
<th style="padding:6px 12px;text-align:left;">Side</th>
<th style="padding:6px 12px;text-align:right;">Qty</th>
<th style="padding:6px 12px;text-align:right;">Price</th>
<th style="padding:6px 12px;text-align:left;">Status</th>
</tr>
</thead>
<tbody>{trade_body}</tbody>
</table>

<h2 style="margin:24px 0 8px;font-size:16px;color:#111;">Halal compliance</h2>
<table style="width:100%;border-collapse:collapse;">
  <tr>
    <td style="padding:4px 0;color:#6b7280;">Non-halal fills (this quarter)</td>
    <td style="padding:4px 0;text-align:right;font-weight:600;">{data.non_halal_fills_quarter}</td>
  </tr>
  <tr>
    <td style="padding:4px 0;color:#6b7280;">Purification accrued</td>
    <td style="padding:4px 0;text-align:right;font-weight:600;">{_bare_money(data.purification_accrued_usd)}</td>
  </tr>
  <tr>
    <td style="padding:4px 0;color:#6b7280;">Purification outstanding</td>
    <td style="padding:4px 0;text-align:right;font-weight:600;">{_bare_money(data.purification_outstanding_usd)}</td>
  </tr>
</table>

<h2 style="margin:24px 0 8px;font-size:16px;color:#111;">Risk</h2>
{risk_table or '<p style="margin:8px 0 0;color:#6b7280;">No risk snapshot available this period.</p>'}

{notable_html}

<hr style="margin:32px 0 16px;border:none;border-top:1px solid #e5e7eb;">
<p style="margin:0;color:#9ca3af;font-size:12px;">
Generated by halal-trader at {data.as_of.strftime("%Y-%m-%d %H:%M:%S UTC")}.
You're receiving this because you opted into operator summaries.
</p>

</div></body></html>"""


# ── Plaintext fallback ────────────────────────────────────


def render_text(data: EmailSummaryInput) -> str:
    """Plaintext twin of the HTML body. Spam filters + terminal
    clients need this; not optional."""
    lines: list[str] = []
    lines.append(f"{data.period_label} summary for {data.operator_name}")
    lines.append("=" * 60)
    lines.append(f"Date: {data.as_of.strftime('%A, %B %-d, %Y')}")
    _, status_label = _STATUS_COLORS.get(
        data.compliance_status, ("#6b7280", data.compliance_status)
    )
    lines.append(f"Halal compliance: {status_label}")
    if data.is_halted:
        lines.append("")
        lines.append(f"⚠ HALT ACTIVE: {data.halt_reason or '(no reason given)'}")
    lines.append("")

    lines.append("P&L")
    lines.append("-" * 60)
    lines.append(
        f"  Today:        {_money(data.today_pnl_usd):>14}  ({_pct(data.today_return_pct)})"
    )
    lines.append(f"  This month:   {_money(data.month_pnl_usd):>14}")
    lines.append(f"  This quarter: {_money(data.quarter_pnl_usd):>14}")
    lines.append("")

    lines.append(f"Trades: {data.trades_today} today, {data.trades_this_week} this week")
    lines.append("-" * 60)
    if data.recent_trades:
        for t in data.recent_trades:
            ts = str(t.get("timestamp", ""))[:19]
            sym = t.get("symbol") or t.get("pair") or "?"
            side = (t.get("side") or "").upper()
            qty = t.get("quantity", 0) or t.get("filled_quantity", 0) or 0
            price = float(t.get("price") or t.get("filled_price") or 0)
            status = t.get("status", "")
            lines.append(f"  {ts}  {sym:<10}  {side:<4}  {qty}  @ {_bare_money(price)}  ({status})")
    else:
        lines.append("  (no trades in this period)")
    lines.append("")

    lines.append("Halal compliance")
    lines.append("-" * 60)
    lines.append(f"  Non-halal fills (this quarter): {data.non_halal_fills_quarter}")
    lines.append(f"  Purification accrued:           {_bare_money(data.purification_accrued_usd)}")
    lines.append(
        f"  Purification outstanding:       {_bare_money(data.purification_outstanding_usd)}"
    )
    lines.append("")

    if data.drawdown_pct is not None or data.portfolio_heat_pct is not None:
        lines.append("Risk")
        lines.append("-" * 60)
        if data.drawdown_pct is not None:
            lines.append(f"  Drawdown:       {_pct(data.drawdown_pct)}")
        if data.portfolio_heat_pct is not None:
            lines.append(f"  Portfolio heat: {_pct(data.portfolio_heat_pct)}")
        lines.append("")

    if data.notable_events:
        lines.append("Notable")
        lines.append("-" * 60)
        for e in data.notable_events:
            lines.append(f"  - {e}")
        lines.append("")

    lines.append("-" * 60)
    lines.append(f"Generated by halal-trader at {data.as_of.strftime('%Y-%m-%d %H:%M:%S UTC')}.")
    return "\n".join(lines)


def render_email(data: EmailSummaryInput) -> RenderedEmail:
    """Compose subject + HTML + plaintext into one bundle."""
    return RenderedEmail(
        subject=render_subject(data),
        html_body=render_html(data),
        text_body=render_text(data),
    )


def summary_from_aaoifi(
    aaoifi: Any,
    *,
    operator_name: str = "operator",
    period_label: str = "Daily",
    today_pnl_usd: float = 0.0,
    today_return_pct: float = 0.0,
    month_pnl_usd: float = 0.0,
    quarter_pnl_usd: float = 0.0,
    trades_today: int | None = None,
    trades_this_week: int = 0,
    recent_trades: list[dict[str, Any]] | None = None,
    drawdown_pct: float | None = None,
    portfolio_heat_pct: float | None = None,
    is_halted: bool = False,
    halt_reason: str = "",
    notable_events: list[str] | None = None,
) -> EmailSummaryInput:
    """Convenience: build an `EmailSummaryInput` from the AAOIFI
    summary + per-call overrides. Single source of truth for the
    halal slice (status / non-halal fills / purification numbers)."""
    return EmailSummaryInput(
        operator_name=operator_name,
        period_label=period_label,
        as_of=datetime.now(UTC),
        today_pnl_usd=today_pnl_usd,
        today_return_pct=today_return_pct,
        month_pnl_usd=month_pnl_usd,
        quarter_pnl_usd=quarter_pnl_usd,
        trades_today=trades_today if trades_today is not None else aaoifi.trades_today,
        trades_this_week=trades_this_week,
        recent_trades=recent_trades or [],
        compliance_status=aaoifi.status,
        non_halal_fills_quarter=aaoifi.non_halal_fills_quarter,
        purification_accrued_usd=aaoifi.purification_accrued_usd,
        purification_outstanding_usd=aaoifi.purification_outstanding_usd,
        drawdown_pct=drawdown_pct,
        portfolio_heat_pct=portfolio_heat_pct,
        is_halted=is_halted,
        halt_reason=halt_reason,
        notable_events=notable_events or [],
    )
