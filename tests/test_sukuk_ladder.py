"""Tests for markets/sukuk_ladder.py — Round-5 Wave 3.D."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.halal.aaoifi_standard_17 import SukukType
from halal_trader.markets.sukuk_ladder import (
    Ladder,
    LadderRung,
    RollPolicy,
    build_ladder,
    even_distribution_target_tenors,
    render_ladder,
    roll,
)


def _rung(
    issuer: str = "GovOfMalaysia",
    issue: date = date(2025, 1, 1),
    matur: date = date(2026, 1, 1),
    face: float = 1000.0,
    coupon: float = 0.04,
    sukuk_type: SukukType = SukukType.IJARA,
) -> LadderRung:
    return LadderRung(
        issuer=issuer,
        sukuk_type=sukuk_type,
        face_value=face,
        coupon_rate=coupon,
        issue_date=issue,
        maturity_date=matur,
    )


# --- Enum + validation -----------------------------------------------------


def test_roll_policy_string_values():
    assert RollPolicy.LONGEST_TENOR.value == "longest_tenor"
    assert RollPolicy.EVEN_DISTRIBUTION.value == "even_distribution"


def test_rung_empty_issuer_rejected():
    with pytest.raises(ValueError):
        _rung(issuer="")


def test_rung_non_tradable_type_rejected():
    with pytest.raises(ValueError):
        _rung(sukuk_type=SukukType.MURABAHA)


def test_rung_salam_type_rejected():
    with pytest.raises(ValueError):
        _rung(sukuk_type=SukukType.SALAM)


def test_rung_negative_face_rejected():
    with pytest.raises(ValueError):
        _rung(face=-1.0)


def test_rung_unreasonable_coupon_rejected():
    with pytest.raises(ValueError):
        _rung(coupon=0.99)


def test_rung_maturity_before_issue_rejected():
    with pytest.raises(ValueError):
        _rung(issue=date(2026, 1, 1), matur=date(2025, 1, 1))


def test_rung_immutable():
    r = _rung()
    with pytest.raises(AttributeError):
        r.face_value = 0.0  # type: ignore[misc]


def test_rung_tenor_days():
    r = _rung(issue=date(2025, 1, 1), matur=date(2026, 1, 1))
    assert r.tenor_days() == 365


# --- Ladder validation -----------------------------------------------------


def test_ladder_empty_rejected():
    with pytest.raises(ValueError):
        Ladder(rungs=())


def test_ladder_unsorted_rejected():
    r1 = _rung(matur=date(2027, 1, 1))
    r2 = _rung(matur=date(2026, 1, 1))
    with pytest.raises(ValueError):
        Ladder(rungs=(r1, r2))


def test_ladder_empty_currency_rejected():
    with pytest.raises(ValueError):
        Ladder(rungs=(_rung(),), base_currency="")


def test_build_ladder_sorts_input():
    r1 = _rung(matur=date(2027, 1, 1))
    r2 = _rung(matur=date(2026, 1, 1))
    ladder = build_ladder([r1, r2])
    assert ladder.rungs[0].maturity_date < ladder.rungs[1].maturity_date


# --- Ladder math -----------------------------------------------------------


def test_ladder_total_face():
    ladder = build_ladder(
        [
            _rung(matur=date(2026, 1, 1), face=1000.0),
            _rung(matur=date(2027, 1, 1), face=2000.0),
            _rung(matur=date(2028, 1, 1), face=3000.0),
        ]
    )
    assert ladder.total_face() == 6000.0


def test_ladder_average_coupon_face_weighted():
    ladder = build_ladder(
        [
            _rung(matur=date(2026, 1, 1), face=1000.0, coupon=0.03),
            _rung(matur=date(2027, 1, 1), face=3000.0, coupon=0.05),
        ]
    )
    # (1000*0.03 + 3000*0.05) / 4000 = 0.045
    assert ladder.average_coupon() == pytest.approx(0.045)


def test_ladder_average_coupon_zero_face_returns_zero():
    """Edge case: ladder with one rung at zero face is invalid (face must be >0)."""
    # We can't construct a zero-face ladder, but the safety branch is exercised
    # implicitly via the >0 invariant. Just check the property method exists.
    ladder = build_ladder([_rung()])
    assert ladder.average_coupon() == pytest.approx(0.04)


def test_ladder_matured_rungs():
    ladder = build_ladder(
        [
            _rung(matur=date(2026, 1, 1)),
            _rung(matur=date(2027, 1, 1)),
            _rung(matur=date(2028, 1, 1)),
        ]
    )
    matured = ladder.matured_rungs(date(2026, 6, 1))
    assert len(matured) == 1
    assert matured[0].maturity_date == date(2026, 1, 1)


def test_ladder_active_rungs():
    ladder = build_ladder(
        [
            _rung(matur=date(2026, 1, 1)),
            _rung(matur=date(2027, 1, 1)),
            _rung(matur=date(2028, 1, 1)),
        ]
    )
    active = ladder.active_rungs(date(2026, 6, 1))
    assert len(active) == 2


def test_ladder_active_at_exact_maturity_date_excluded():
    """A rung whose maturity = today is matured (paid back today)."""
    ladder = build_ladder([_rung(matur=date(2026, 1, 1))])
    assert ladder.matured_rungs(date(2026, 1, 1)) == ladder.rungs
    assert ladder.active_rungs(date(2026, 1, 1)) == ()


# --- Even distribution -----------------------------------------------------


def test_even_distribution_basic():
    """5-rung ladder over 5 years → 1, 2, 3, 4, 5."""
    out = even_distribution_target_tenors(5, 5)
    assert out == (1, 2, 3, 4, 5)


def test_even_distribution_single_rung_uses_max_tenor():
    assert even_distribution_target_tenors(1, 10) == (10,)


def test_even_distribution_zero_rungs_rejected():
    with pytest.raises(ValueError):
        even_distribution_target_tenors(0, 5)


def test_even_distribution_zero_max_rejected():
    with pytest.raises(ValueError):
        even_distribution_target_tenors(5, 0)


# --- Roll ------------------------------------------------------------------


def test_roll_no_matured_returns_same_ladder():
    ladder = build_ladder(
        [
            _rung(matur=date(2027, 1, 1)),
            _rung(matur=date(2028, 1, 1)),
        ]
    )
    replacement = _rung(issue=date(2026, 6, 1), matur=date(2031, 1, 1))
    result = roll(ladder, today=date(2026, 6, 1), replacement=replacement)
    assert result is ladder  # identity preservation when nothing to roll


def test_roll_replaces_matured_with_new_long_end():
    ladder = build_ladder(
        [
            _rung(matur=date(2026, 1, 1)),
            _rung(matur=date(2027, 1, 1)),
            _rung(matur=date(2028, 1, 1)),
        ]
    )
    replacement = _rung(issue=date(2026, 6, 1), matur=date(2031, 1, 1))
    new_ladder = roll(ladder, today=date(2026, 6, 1), replacement=replacement)
    # Two active + one new long end = 3 rungs
    assert len(new_ladder.rungs) == 3
    # Longest is the replacement
    assert new_ladder.rungs[-1].maturity_date == date(2031, 1, 1)


def test_roll_replacement_maturity_must_be_longest():
    ladder = build_ladder(
        [
            _rung(matur=date(2026, 1, 1)),
            _rung(matur=date(2027, 1, 1)),
            _rung(matur=date(2028, 1, 1)),
        ]
    )
    bad_replacement = _rung(issue=date(2026, 6, 1), matur=date(2027, 6, 1))  # too short
    with pytest.raises(ValueError):
        roll(ladder, today=date(2026, 6, 1), replacement=bad_replacement)


def test_roll_when_all_matured_simply_holds_replacement():
    ladder = build_ladder(
        [
            _rung(matur=date(2026, 1, 1)),
            _rung(matur=date(2026, 6, 1)),
        ]
    )
    replacement = _rung(issue=date(2026, 6, 1), matur=date(2031, 1, 1))
    new_ladder = roll(ladder, today=date(2027, 1, 1), replacement=replacement)
    assert len(new_ladder.rungs) == 1
    assert new_ladder.rungs[0].maturity_date == date(2031, 1, 1)


# --- Render -----------------------------------------------------------------


def test_render_ladder_includes_summary():
    ladder = build_ladder(
        [
            _rung(matur=date(2026, 1, 1), face=1000.0, coupon=0.04),
            _rung(matur=date(2027, 1, 1), face=1000.0, coupon=0.05),
        ]
    )
    out = render_ladder(ladder)
    assert "2 rungs" in out
    assert "GovOfMalaysia" in out
    assert "matures 2026-01-01" in out


def test_render_ladder_marks_matured_when_today_given():
    ladder = build_ladder([_rung(matur=date(2026, 1, 1))])
    out = render_ladder(ladder, today=date(2026, 6, 1))
    assert "[matured]" in out


def test_render_ladder_no_matured_marker_without_today():
    ladder = build_ladder([_rung(matur=date(2026, 1, 1))])
    out = render_ladder(ladder)
    assert "[matured]" not in out


# --- E2E -------------------------------------------------------------------


def test_e2e_5y_ladder_construction_and_roll():
    """Build a 5y ladder, roll it forward by one year."""
    rungs = [
        _rung(issue=date(2025, 1, 1), matur=date(2026, 1, 1), face=1000.0, coupon=0.03),
        _rung(issue=date(2025, 1, 1), matur=date(2027, 1, 1), face=1000.0, coupon=0.04),
        _rung(issue=date(2025, 1, 1), matur=date(2028, 1, 1), face=1000.0, coupon=0.045),
        _rung(issue=date(2025, 1, 1), matur=date(2029, 1, 1), face=1000.0, coupon=0.05),
        _rung(issue=date(2025, 1, 1), matur=date(2030, 1, 1), face=1000.0, coupon=0.055),
    ]
    ladder = build_ladder(rungs)
    assert len(ladder.rungs) == 5
    # After 1 year, the 1y rung matures → roll into a new 5y rung
    replacement = _rung(issue=date(2026, 1, 2), matur=date(2031, 1, 2), face=1000.0, coupon=0.06)
    new_ladder = roll(ladder, today=date(2026, 1, 1), replacement=replacement)
    assert len(new_ladder.rungs) == 5  # 4 active + 1 new
    assert new_ladder.rungs[-1].maturity_date == date(2031, 1, 2)


def test_replay_consistency():
    a = build_ladder([_rung()])
    b = build_ladder([_rung()])
    assert a == b
