"""Tests for the operator end-of-day email renderer.

The renderer is a pure function — given an `EmailSummaryInput`, it
returns deterministic HTML + plaintext + subject. No I/O. Tests
pin the *content* (every section is rendered, dollar / pct
formatting is consistent) and the *halal-status discipline* (any
violation visible at a glance from the subject line emoji).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from halal_trader.notifications.email import (
    EmailSummaryInput,
    RenderedEmail,
    render_email,
    render_html,
    render_subject,
    render_text,
    summary_from_aaoifi,
)


def _input(**overrides) -> EmailSummaryInput:
    base = dict(
        operator_name="ahmed",
        period_label="Daily",
        as_of=datetime(2026, 4, 25, 16, 0, tzinfo=UTC),
    )
    base.update(overrides)
    return EmailSummaryInput(**base)


# ── Subject ────────────────────────────────────────────────


def test_subject_has_status_emoji_and_pnl():
    s = render_subject(_input(today_pnl_usd=125.50))
    assert "🟢" in s  # compliant default
    assert "+$125.50" in s
    assert "Daily" in s
    assert "2026-04-25" in s


def test_subject_emoji_red_on_violation():
    """A non-halal trade in the quarter must be visible from the
    *inbox preview* — not just deep inside the email."""
    s = render_subject(_input(compliance_status="violation", today_pnl_usd=100))
    assert "🔴" in s
    assert "🟢" not in s


def test_subject_emoji_amber_on_attention():
    s = render_subject(_input(compliance_status="attention"))
    assert "🟠" in s


def test_subject_negative_pnl_shows_minus_sign():
    s = render_subject(_input(today_pnl_usd=-250.0))
    assert "-$250.00" in s


def test_subject_includes_period_label():
    """Daily vs Weekly distinguishable from the inbox without
    opening the email."""
    daily = render_subject(_input(period_label="Daily"))
    weekly = render_subject(_input(period_label="Weekly"))
    assert "Daily" in daily
    assert "Weekly" in weekly


# ── HTML rendering ────────────────────────────────────────


def test_html_includes_every_section():
    """Pin so a refactor that drops a section (e.g. Halal compliance
    block) breaks here loudly."""
    html = render_html(_input(today_pnl_usd=100, recent_trades=[]))
    for section in ("P&amp;L", "Trades", "Halal compliance", "Risk"):
        assert section in html


def test_html_renders_compliance_status_label_and_color():
    """The status banner uses one of the three known label/color
    pairs. Pin so a refactor that introduces a new status doesn't
    silently render as the default grey."""
    for status, label_part in (
        ("compliant", "Compliant"),
        ("attention", "Attention"),
        ("violation", "Violation"),
    ):
        html = render_html(_input(compliance_status=status))
        assert label_part in html


def test_html_halt_banner_only_when_halted():
    no_halt = render_html(_input(is_halted=False))
    halted = render_html(_input(is_halted=True, halt_reason="loss limit hit"))
    assert "HALT ACTIVE" not in no_halt
    assert "HALT ACTIVE" in halted
    assert "loss limit hit" in halted


def test_html_halt_banner_with_no_reason_uses_fallback():
    """A halt with empty reason still shows the banner — operator must
    see the halt is on, even if the reason was lost."""
    halted = render_html(_input(is_halted=True, halt_reason=""))
    assert "HALT ACTIVE" in halted
    assert "(no reason given)" in halted


def test_html_pnl_color_green_on_positive():
    html = render_html(_input(today_pnl_usd=150.0))
    assert "#16a34a" in html  # green hex


def test_html_pnl_color_red_on_negative():
    html = render_html(_input(today_pnl_usd=-150.0))
    assert "#dc2626" in html  # red hex


def test_html_renders_trades_table_with_data():
    trades = [
        {
            "timestamp": "2026-04-25T10:30:00",
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10,
            "price": 150.0,
            "status": "filled",
        },
        {
            "timestamp": "2026-04-25T11:45:00",
            "symbol": "MSFT",
            "side": "sell",
            "quantity": 5,
            "price": 410.5,
            "status": "filled",
        },
    ]
    html = render_html(_input(recent_trades=trades))
    assert "AAPL" in html
    assert "MSFT" in html
    assert "$150.00" in html
    assert "$410.50" in html


def test_html_renders_no_trades_placeholder():
    """Empty trades list → friendly placeholder, not a broken
    empty table that looks like a bug."""
    html = render_html(_input(recent_trades=[]))
    assert "No trades in this period" in html


def test_html_handles_crypto_pair_field_alias():
    """Crypto trades use `pair` instead of `symbol`. Renderer must
    handle both — pinned so a refactor that hard-codes `symbol`
    breaks here."""
    trades = [
        {
            "timestamp": "2026-04-25T10:30:00",
            "pair": "BTCUSDT",
            "side": "buy",
            "quantity": 0.001,
            "price": 50_000.0,
            "status": "filled",
        }
    ]
    html = render_html(_input(recent_trades=trades))
    assert "BTCUSDT" in html
    assert "$50,000.00" in html


def test_html_includes_purification_amounts():
    html = render_html(
        _input(
            purification_accrued_usd=12.50,
            purification_outstanding_usd=3.25,
        )
    )
    assert "$12.50" in html
    assert "$3.25" in html


def test_html_risk_section_omitted_when_no_data():
    """When neither drawdown nor heat is set, the risk block shows a
    "no snapshot available" placeholder, not an empty table."""
    html = render_html(_input(drawdown_pct=None, portfolio_heat_pct=None))
    assert "No risk snapshot" in html


def test_html_risk_section_shown_when_drawdown_only():
    html = render_html(_input(drawdown_pct=0.05))
    assert "Drawdown" in html
    assert "+5.00%" in html


def test_html_risk_section_shown_when_heat_only():
    html = render_html(_input(portfolio_heat_pct=0.02))
    assert "Portfolio heat" in html


def test_html_notable_events_rendered_as_list():
    html = render_html(_input(notable_events=["shadow runner diverged 50bps", "VIX spike 22→35"]))
    assert "Notable" in html
    assert "shadow runner diverged 50bps" in html
    assert "VIX spike" in html


def test_html_notable_section_omitted_when_empty():
    html = render_html(_input(notable_events=[]))
    assert "Notable" not in html


def test_html_includes_operator_name_in_greeting():
    html = render_html(_input(operator_name="aisha"))
    assert "Hi aisha" in html


def test_html_pnl_amounts_use_thousands_separators():
    """Pin: `$1,234.56`, not `$1234.56` — readability matters."""
    html = render_html(_input(today_pnl_usd=12345.67))
    assert "$12,345.67" in html


def test_html_compliance_status_unknown_renders_with_default():
    """An unrecognised status (future addition that this version
    doesn't know about) renders with grey + the raw label —
    defensive fallback rather than crash."""
    html = render_html(_input(compliance_status="experimental"))
    assert "experimental" in html


# ── Plaintext rendering ──────────────────────────────────


def test_text_includes_every_section_label():
    text = render_text(_input())
    for label in ("P&L", "Trades:", "Halal compliance"):
        assert label in text


def test_text_includes_operator_name():
    text = render_text(_input(operator_name="omar"))
    assert "omar" in text


def test_text_renders_trades_one_per_line():
    trades = [
        {
            "timestamp": "2026-04-25T10:30:00",
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10,
            "price": 150.0,
            "status": "filled",
        }
    ]
    text = render_text(_input(recent_trades=trades))
    assert "AAPL" in text
    assert "BUY" in text  # uppercased side
    assert "$150.00" in text


def test_text_no_trades_placeholder():
    text = render_text(_input(recent_trades=[]))
    assert "(no trades in this period)" in text


def test_text_renders_halt_banner():
    text = render_text(_input(is_halted=True, halt_reason="loss limit"))
    assert "HALT ACTIVE" in text
    assert "loss limit" in text


def test_text_skips_risk_section_when_no_data():
    text = render_text(_input())
    assert "Risk\n" not in text


def test_text_renders_notable_events():
    text = render_text(_input(notable_events=["x", "y"]))
    assert "Notable" in text
    assert "- x" in text
    assert "- y" in text


# ── render_email composes all three ───────────────────────


def test_render_email_returns_rendered_bundle():
    out = render_email(_input(today_pnl_usd=100))
    assert isinstance(out, RenderedEmail)
    assert out.subject
    assert out.html_body.startswith("<!DOCTYPE html>")
    assert "P&L" in out.text_body


def test_rendered_email_is_frozen():
    import pytest

    out = render_email(_input())
    with pytest.raises(Exception):
        out.subject = "different"  # type: ignore[misc]


# ── summary_from_aaoifi adapter ───────────────────────────


@dataclass
class _StubAAOIFI:
    """Mirrors the public surface of `AAOIFISummary` for the adapter."""

    status: str = "compliant"
    trades_today: int = 3
    non_halal_fills_quarter: int = 0
    purification_accrued_usd: float = 5.0
    purification_outstanding_usd: float = 2.0


def test_summary_from_aaoifi_threads_compliance_through():
    inp = summary_from_aaoifi(
        _StubAAOIFI(status="violation", non_halal_fills_quarter=1),
    )
    assert inp.compliance_status == "violation"
    assert inp.non_halal_fills_quarter == 1


def test_summary_from_aaoifi_uses_aaoifi_trades_today_by_default():
    """No explicit `trades_today` override → take it from the
    AAOIFI summary so we stay consistent with the dashboard tile."""
    inp = summary_from_aaoifi(_StubAAOIFI(trades_today=7))
    assert inp.trades_today == 7


def test_summary_from_aaoifi_explicit_trades_today_wins():
    """Caller can override with a per-broker / per-period number."""
    inp = summary_from_aaoifi(_StubAAOIFI(trades_today=7), trades_today=42)
    assert inp.trades_today == 42


def test_summary_from_aaoifi_threads_purification_numbers():
    inp = summary_from_aaoifi(
        _StubAAOIFI(purification_accrued_usd=10.0, purification_outstanding_usd=4.0),
    )
    assert inp.purification_accrued_usd == 10.0
    assert inp.purification_outstanding_usd == 4.0


def test_summary_from_aaoifi_default_period_label_daily():
    inp = summary_from_aaoifi(_StubAAOIFI())
    assert inp.period_label == "Daily"


def test_summary_from_aaoifi_weekly_label_passthrough():
    inp = summary_from_aaoifi(_StubAAOIFI(), period_label="Weekly")
    assert inp.period_label == "Weekly"
