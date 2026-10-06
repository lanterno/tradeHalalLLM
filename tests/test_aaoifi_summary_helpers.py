"""Tests for the pure helpers + dataclasses in `halal/aaoifi_summary.py`.

The DB-aggregating `compute_aaoifi_summary` is covered in
tests/test_aaoifi_summary_db.py. This file pins the in-memory surface —
the period helper and the status / compliance / outstanding-purification
properties — so a refactor that flips a comparison breaks here first.
"""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.halal.aaoifi_summary import AAOIFISummary, AccountCompliance, periods

# ── Periods (New York calendar days) ──────────────────────


@pytest.mark.parametrize(
    ("today", "quarter"),
    [
        (date(2026, 1, 15), date(2026, 1, 1)),
        (date(2026, 3, 31), date(2026, 1, 1)),
        (date(2026, 4, 1), date(2026, 4, 1)),
        (date(2026, 6, 20), date(2026, 4, 1)),
        (date(2026, 9, 1), date(2026, 7, 1)),
        (date(2026, 12, 31), date(2026, 10, 1)),
        (date(2025, 8, 15), date(2025, 7, 1)),
    ],
)
def test_the_quarter_starts_on_the_first_of_its_first_month(today: date, quarter: date) -> None:
    q, m, t = periods(today)
    assert q == quarter
    assert m == today.replace(day=1)
    assert t == today


# ── AAOIFISummary properties ──────────────────────────────


def _account(account: str = "core", **verdicts: int) -> AccountCompliance:
    return AccountCompliance(
        account=account,
        label=account,
        trades_today=0,
        trades_this_month=0,
        trades_this_quarter=sum(verdicts.values()),
        buys_this_quarter=sum(verdicts.values()),
        buy_verdicts={"halal": 0, "doubtful": 0, "not_halal": 0, "unscreened": 0, **verdicts},
    )


def _summary(
    *,
    non_halal_fills: int = 0,
    accrued: float = 0.0,
    disbursed: float = 0.0,
) -> AAOIFISummary:
    """Build a summary with controllable invariants for the property tests."""
    return AAOIFISummary(
        quarter_start=date(2026, 4, 1),
        month_start=date(2026, 4, 1),
        today_start=date(2026, 4, 25),
        accounts=(_account("core", halal=3), _account("paper", not_halal=non_halal_fills)),
        purification_accrued_usd=accrued,
        purification_disbursed_usd=disbursed,
    )


def test_an_account_violates_on_any_buy_the_screen_did_not_hold_halal() -> None:
    assert _account(halal=5).status == "compliant"
    for verdict in ("doubtful", "not_halal", "unscreened"):
        acct = _account(halal=5, **{verdict: 1})
        assert acct.non_halal_buys_quarter == 1 and acct.status == "violation"


def test_trade_counts_sum_the_accounts() -> None:
    s = _summary(non_halal_fills=2)
    assert s.trades_this_quarter == 5 and s.non_halal_fills_quarter == 2


def test_outstanding_is_accrued_minus_disbursed():
    s = _summary(accrued=100.0, disbursed=30.0)
    assert s.purification_outstanding_usd == 70.0


def test_outstanding_floors_at_zero_when_disbursed_exceeds_accrued():
    """Defensive: if the operator over-disbursed (or a refund flow
    later gets wired), outstanding stays >= 0 rather than going
    negative."""
    s = _summary(accrued=10.0, disbursed=50.0)
    assert s.purification_outstanding_usd == 0.0


def test_outstanding_zero_when_both_zero():
    s = _summary()
    assert s.purification_outstanding_usd == 0.0


def test_is_compliant_true_when_zero_non_halal_fills():
    s = _summary(non_halal_fills=0)
    assert s.is_compliant is True


def test_is_compliant_false_with_any_non_halal_fill():
    """A single non-halal buy flips compliance to False — the whole
    point of the tile. Pin so a refactor doesn't accidentally
    threshold this."""
    s = _summary(non_halal_fills=1)
    assert s.is_compliant is False


def test_status_violation_takes_priority_over_attention():
    """Even with outstanding purification, a non-halal buy renders
    as 'violation' — the more severe state wins."""
    s = _summary(non_halal_fills=1, accrued=100.0, disbursed=0.0)
    assert s.status == "violation"


def test_status_attention_when_outstanding_purification():
    """No violations + outstanding purification → 'attention' (amber tile)."""
    s = _summary(non_halal_fills=0, accrued=100.0, disbursed=0.0)
    assert s.status == "attention"


def test_status_attention_when_partially_disbursed():
    s = _summary(non_halal_fills=0, accrued=100.0, disbursed=50.0)
    assert s.status == "attention"


def test_status_compliant_when_all_disbursed():
    s = _summary(non_halal_fills=0, accrued=100.0, disbursed=100.0)
    assert s.status == "compliant"


def test_status_compliant_when_no_purification_owed():
    s = _summary()
    assert s.status == "compliant"


def test_status_treats_sub_one_cent_as_compliant():
    """Defensive: floating-point residue under $0.01 is rounding
    noise; tile shouldn't go amber for a rounding-error
    outstanding amount."""
    s = _summary(accrued=100.001, disbursed=100.0)
    assert s.status == "compliant"


def test_status_treats_one_cent_or_more_as_attention():
    """A real residual disbursement obligation flips to attention."""
    s = _summary(accrued=100.02, disbursed=100.0)
    assert s.status == "attention"


def test_summary_dataclass_is_frozen():
    s = _summary()
    with pytest.raises(Exception):
        s.purification_accrued_usd = 999  # type: ignore[misc]
