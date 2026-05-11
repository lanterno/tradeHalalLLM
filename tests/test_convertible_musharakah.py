"""Tests for halal/convertible_musharakah.py — Round-5 Wave 6.D."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.halal.convertible_musharakah import (
    ConversionEvent,
    ConversionTrigger,
    ConvertibleNote,
    NoteStatus,
    conversion_price,
    convert,
    render_conversion,
    render_note,
    wakalah_fee_owed,
)


def _note(
    note_id: str = "N1",
    investor: str = "alice",
    issuer: str = "bob",
    principal: float = 250_000.0,
    issue_date: date = date(2026, 6, 1),
    maturity_date: date = date(2028, 6, 1),
    valuation_cap: float | None = 5_000_000.0,
    discount: float = 0.20,
    wakalah_fee: float = 0.0,
    triggers: tuple[ConversionTrigger, ...] | None = None,
    revenue_threshold: float | None = None,
) -> ConvertibleNote:
    if triggers is None:
        triggers = (ConversionTrigger.NEXT_ROUND,)
    return ConvertibleNote(
        note_id=note_id,
        investor_id=investor,
        issuer_id=issuer,
        principal_usd=principal,
        issue_date=issue_date,
        maturity_date=maturity_date,
        valuation_cap_usd=valuation_cap,
        discount_pct=discount,
        wakalah_fee_annual_pct=wakalah_fee,
        triggers=triggers,
        revenue_threshold_usd=revenue_threshold,
    )


# --- Note validation -----------------------------------------------------


def test_note_valid():
    n = _note()
    assert n.principal_usd == 250_000.0


def test_note_self_dealing_rejected():
    with pytest.raises(ValueError):
        _note(investor="x", issuer="x")


def test_note_negative_principal_rejected():
    with pytest.raises(ValueError):
        _note(principal=-1.0)


def test_note_maturity_before_issue_rejected():
    with pytest.raises(ValueError):
        _note(issue_date=date(2026, 6, 1), maturity_date=date(2026, 5, 1))


def test_note_discount_above_20pct_rejected():
    """Pin: halal-cap on discount = 20%."""
    with pytest.raises(ValueError):
        _note(discount=0.25)


def test_note_wakalah_fee_above_2pct_rejected():
    """Pin: Wakalah fee > 2% reads as interest."""
    with pytest.raises(ValueError):
        _note(wakalah_fee=0.05)


def test_note_revenue_trigger_requires_threshold():
    with pytest.raises(ValueError):
        _note(triggers=(ConversionTrigger.REVENUE,), revenue_threshold=None)


def test_note_no_triggers_rejected():
    with pytest.raises(ValueError):
        _note(triggers=tuple())


def test_note_negative_revenue_threshold_rejected():
    with pytest.raises(ValueError):
        _note(
            triggers=(ConversionTrigger.REVENUE,),
            revenue_threshold=-1000.0,
        )


def test_note_immutable():
    n = _note()
    with pytest.raises(AttributeError):
        n.principal_usd = 0.0  # type: ignore[misc]


# --- ConversionEvent validation -----------------------------------------


def test_event_next_round_requires_price():
    with pytest.raises(ValueError):
        ConversionEvent(
            trigger=ConversionTrigger.NEXT_ROUND,
            event_date=date(2027, 1, 1),
        )


def test_event_revenue_requires_revenue():
    with pytest.raises(ValueError):
        ConversionEvent(
            trigger=ConversionTrigger.REVENUE,
            event_date=date(2027, 1, 1),
        )


def test_event_exit_requires_exit_price():
    with pytest.raises(ValueError):
        ConversionEvent(
            trigger=ConversionTrigger.EXIT,
            event_date=date(2027, 1, 1),
        )


# --- conversion_price ---------------------------------------------------


def test_conversion_price_next_round_with_discount():
    n = _note(discount=0.20, valuation_cap=None)
    e = ConversionEvent(
        trigger=ConversionTrigger.NEXT_ROUND,
        event_date=date(2027, 1, 1),
        next_round_price_per_share=10.0,
    )
    # 10 × 0.80 = 8.
    assert conversion_price(n, e) == pytest.approx(8.0)


def test_conversion_price_exit_with_discount():
    n = _note(discount=0.10)
    e = ConversionEvent(
        trigger=ConversionTrigger.EXIT,
        event_date=date(2027, 1, 1),
        exit_price_per_share=20.0,
    )
    assert conversion_price(n, e) == pytest.approx(18.0)


# --- convert — NEXT_ROUND -----------------------------------------------


def test_convert_next_round_basic():
    """Pin: shares = principal / discounted_price."""
    n = _note(principal=100_000, discount=0.20, valuation_cap=None)
    e = ConversionEvent(
        trigger=ConversionTrigger.NEXT_ROUND,
        event_date=date(2027, 1, 1),
        next_round_price_per_share=10.0,
    )
    res = convert(n, e)
    # Discounted price = 8; shares = 100,000 / 8 = 12,500.
    assert res.shares_issued == pytest.approx(12_500.0)
    assert res.discount_applied
    assert not res.cap_applied


def test_convert_with_cap_applies_cap():
    """When cap-implied price < discounted price, cap dominates.

    cap_implied = round_price × cap/post_money = 10 × 5M/20M = 2.5;
    discounted = 10 × 0.80 = 8. Cap < discount → cap wins.
    """
    n = _note(principal=100_000, discount=0.20, valuation_cap=5_000_000.0)
    e = ConversionEvent(
        trigger=ConversionTrigger.NEXT_ROUND,
        event_date=date(2027, 1, 1),
        next_round_price_per_share=10.0,
        next_round_post_money_usd=20_000_000.0,
    )
    res = convert(n, e)
    assert res.cap_applied
    assert res.effective_price_per_share == pytest.approx(2.5)


def test_convert_cap_skipped_without_post_money():
    """Without post_money, cap math is undefined; only discount applies."""
    n = _note(principal=100_000, discount=0.20, valuation_cap=5_000_000.0)
    e = ConversionEvent(
        trigger=ConversionTrigger.NEXT_ROUND,
        event_date=date(2027, 1, 1),
        next_round_price_per_share=10.0,
    )
    res = convert(n, e)
    assert not res.cap_applied
    assert res.discount_applied
    assert res.effective_price_per_share == pytest.approx(8.0)


def test_convert_status_must_be_outstanding():
    n = _note()
    n_converted = ConvertibleNote(
        note_id=n.note_id,
        investor_id=n.investor_id,
        issuer_id=n.issuer_id,
        principal_usd=n.principal_usd,
        issue_date=n.issue_date,
        maturity_date=n.maturity_date,
        valuation_cap_usd=n.valuation_cap_usd,
        discount_pct=n.discount_pct,
        triggers=n.triggers,
        status=NoteStatus.CONVERTED,
    )
    e = ConversionEvent(
        trigger=ConversionTrigger.NEXT_ROUND,
        event_date=date(2027, 1, 1),
        next_round_price_per_share=10.0,
    )
    with pytest.raises(ValueError):
        convert(n_converted, e)


def test_convert_event_must_match_configured_triggers():
    n = _note(triggers=(ConversionTrigger.NEXT_ROUND,))
    e = ConversionEvent(
        trigger=ConversionTrigger.EXIT,
        event_date=date(2027, 1, 1),
        exit_price_per_share=10.0,
    )
    with pytest.raises(ValueError):
        convert(n, e)


# --- convert — EXIT -----------------------------------------------------


def test_convert_exit_path():
    n = _note(
        principal=100_000,
        discount=0.10,
        triggers=(ConversionTrigger.EXIT,),
        valuation_cap=None,
    )
    e = ConversionEvent(
        trigger=ConversionTrigger.EXIT,
        event_date=date(2027, 1, 1),
        exit_price_per_share=20.0,
    )
    res = convert(n, e)
    # Discounted price = 18; shares = 100,000 / 18 ≈ 5555.56.
    assert res.shares_issued == pytest.approx(100_000 / 18)


# --- convert — REVENUE / MATURITY_DATE -----------------------------------


def test_convert_revenue_uses_cap():
    n = _note(
        principal=100_000,
        triggers=(ConversionTrigger.REVENUE,),
        revenue_threshold=1_000_000.0,
        valuation_cap=5_000_000.0,
    )
    e = ConversionEvent(
        trigger=ConversionTrigger.REVENUE,
        event_date=date(2027, 1, 1),
        revenue_at_event=1_500_000.0,
    )
    res = convert(n, e)
    # shares = 100k / 5M = 0.02.
    assert res.shares_issued == pytest.approx(0.02)


def test_convert_revenue_without_cap_rejected():
    n = _note(
        triggers=(ConversionTrigger.REVENUE,),
        revenue_threshold=1_000_000.0,
        valuation_cap=None,
    )
    e = ConversionEvent(
        trigger=ConversionTrigger.REVENUE,
        event_date=date(2027, 1, 1),
        revenue_at_event=1_500_000.0,
    )
    with pytest.raises(ValueError):
        convert(n, e)


def test_convert_maturity_date_uses_cap():
    n = _note(
        principal=100_000,
        triggers=(ConversionTrigger.MATURITY_DATE,),
        valuation_cap=5_000_000.0,
    )
    e = ConversionEvent(
        trigger=ConversionTrigger.MATURITY_DATE,
        event_date=date(2028, 6, 1),
    )
    res = convert(n, e)
    assert res.shares_issued == pytest.approx(0.02)


# --- wakalah_fee_owed ----------------------------------------------------


def test_wakalah_fee_zero_at_issue():
    n = _note(wakalah_fee=0.02)
    assert wakalah_fee_owed(n, as_of=n.issue_date) == 0.0


def test_wakalah_fee_simple_interest_arithmetic():
    """Pin: simple, NOT compound."""
    n = _note(
        principal=100_000,
        wakalah_fee=0.02,
        issue_date=date(2026, 1, 1),
    )
    fee = wakalah_fee_owed(n, as_of=date(2027, 1, 1))
    # 100k × 2% × (365/365) = 2000.
    assert fee == pytest.approx(2_000.0, rel=0.01)


def test_wakalah_fee_zero_when_rate_is_zero():
    n = _note(wakalah_fee=0.0)
    assert wakalah_fee_owed(n, as_of=date(2027, 6, 1)) == 0.0


def test_wakalah_fee_zero_when_as_of_before_issue():
    n = _note(
        wakalah_fee=0.02,
        issue_date=date(2026, 6, 1),
    )
    assert wakalah_fee_owed(n, as_of=date(2026, 1, 1)) == 0.0


# --- Render --------------------------------------------------------------


def test_render_note_no_secret_leak():
    n = _note(investor="alice@example.com", issuer="bob@example.com")
    out = render_note(n)
    assert "alice@example.com" not in out
    assert "bob@example.com" not in out


def test_render_note_uncapped_marker():
    n = _note(valuation_cap=None)
    out = render_note(n)
    assert "uncapped" in out


def test_render_conversion_flags():
    n = _note(principal=100_000, discount=0.20, valuation_cap=500_000.0)
    e = ConversionEvent(
        trigger=ConversionTrigger.NEXT_ROUND,
        event_date=date(2027, 1, 1),
        next_round_price_per_share=10.0,
    )
    res = convert(n, e)
    out = render_conversion(res)
    assert "Conversion" in out
    assert "shares" in out


def test_render_conversion_no_flags_path():
    n = _note(principal=100_000, discount=0.0, valuation_cap=None)
    e = ConversionEvent(
        trigger=ConversionTrigger.NEXT_ROUND,
        event_date=date(2027, 1, 1),
        next_round_price_per_share=10.0,
    )
    res = convert(n, e)
    out = render_conversion(res)
    assert "[" not in out  # no flag bracket
