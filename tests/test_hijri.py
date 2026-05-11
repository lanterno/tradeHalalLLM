"""Tests for i18n/hijri.py — Round-5 Wave 23."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.i18n.hijri import (
    HijriDate,
    HijriMonth,
    add_days,
    gregorian_to_hijri,
    hijri_difference_days,
    hijri_month_length,
    hijri_to_gregorian,
    hijri_year_length,
    is_hijri_leap_year,
    month_name,
    render_dual,
    render_hijri,
)

# --- is_hijri_leap_year + cycle ----------------


def test_is_leap_year_known_pattern():
    # Pattern: 2,5,7,10,13,16,18,21,24,26,29 in each 30y cycle.
    assert is_hijri_leap_year(2)
    assert is_hijri_leap_year(5)
    assert is_hijri_leap_year(7)
    assert is_hijri_leap_year(29)
    assert is_hijri_leap_year(32)  # 32 ≡ 2 mod 30


def test_is_leap_year_non_leap():
    assert not is_hijri_leap_year(1)
    assert not is_hijri_leap_year(3)
    assert not is_hijri_leap_year(30)


def test_is_leap_year_zero_rejected():
    with pytest.raises(ValueError):
        is_hijri_leap_year(0)


def test_year_lengths_pinned():
    assert hijri_year_length(1) == 354
    assert hijri_year_length(2) == 355  # leap


def test_cycle_total_days():
    """Pin: 30-year cycle = 19×354 + 11×355 = 10631 days."""
    total = sum(hijri_year_length(y) for y in range(1, 31))
    assert total == 10_631


# --- hijri_month_length ------------------------


def test_month_length_odd_30():
    assert hijri_month_length(1, HijriMonth.MUHARRAM) == 30
    assert hijri_month_length(1, HijriMonth.RABI_AL_AWWAL) == 30
    assert hijri_month_length(1, HijriMonth.DHU_AL_QADAH) == 30


def test_month_length_even_29():
    assert hijri_month_length(1, HijriMonth.SAFAR) == 29
    assert hijri_month_length(1, HijriMonth.JUMADA_AL_THANI) == 29


def test_month_length_dhu_al_hijjah_leap():
    """Pin: Dhu al-Hijjah is 30 in leap years, 29 otherwise."""
    assert hijri_month_length(1, HijriMonth.DHU_AL_HIJJAH) == 29  # non-leap
    assert hijri_month_length(2, HijriMonth.DHU_AL_HIJJAH) == 30  # leap


# --- HijriDate validation ----------------------


def test_hijri_date_valid():
    d = HijriDate(year=1447, month=HijriMonth.MUHARRAM, day=1)
    assert d.day == 1


def test_hijri_date_zero_year_rejected():
    with pytest.raises(ValueError):
        HijriDate(year=0, month=HijriMonth.MUHARRAM, day=1)


def test_hijri_date_zero_day_rejected():
    with pytest.raises(ValueError):
        HijriDate(year=1447, month=HijriMonth.MUHARRAM, day=0)


def test_hijri_date_day_above_month_length_rejected():
    """Safar has 29 days."""
    with pytest.raises(ValueError):
        HijriDate(year=1447, month=HijriMonth.SAFAR, day=30)


def test_hijri_date_day_31_rejected_for_30d_month():
    with pytest.raises(ValueError):
        HijriDate(year=1447, month=HijriMonth.MUHARRAM, day=31)


def test_hijri_date_30_dhu_al_hijjah_only_in_leap():
    HijriDate(year=2, month=HijriMonth.DHU_AL_HIJJAH, day=30)  # leap: OK
    with pytest.raises(ValueError):
        HijriDate(year=1, month=HijriMonth.DHU_AL_HIJJAH, day=30)


def test_hijri_date_immutable():
    d = HijriDate(year=1447, month=HijriMonth.MUHARRAM, day=1)
    with pytest.raises(AttributeError):
        d.day = 5  # type: ignore[misc]


def test_hijri_date_ordering():
    a = HijriDate(year=1447, month=HijriMonth.MUHARRAM, day=1)
    b = HijriDate(year=1447, month=HijriMonth.SAFAR, day=1)
    assert a < b
    assert not (b < a)


# --- gregorian ↔ hijri round trip ---------------


def test_round_trip_well_known_date():
    """1 Muharram 1447 AH ≈ 6 July 2025."""
    h_in = HijriDate(year=1447, month=HijriMonth.MUHARRAM, day=1)
    g_out = hijri_to_gregorian(h_in)
    h_back = gregorian_to_hijri(g_out)
    assert h_back == h_in


def test_round_trip_today_class_dates():
    """Multiple round trips."""
    for g_in in (
        date(2026, 5, 11),
        date(2026, 1, 1),
        date(2030, 12, 31),
        date(1990, 6, 1),
        date(2024, 2, 29),
    ):
        h = gregorian_to_hijri(g_in)
        g_back = hijri_to_gregorian(h)
        assert g_back == g_in


def test_round_trip_hijri_dates():
    for year, month, day in (
        (1, HijriMonth.MUHARRAM, 1),
        (2, HijriMonth.DHU_AL_HIJJAH, 30),  # leap year
        (1447, HijriMonth.RAMADAN, 15),
        (1500, HijriMonth.SHAWWAL, 1),
    ):
        h_in = HijriDate(year=year, month=month, day=day)
        g = hijri_to_gregorian(h_in)
        h_back = gregorian_to_hijri(g)
        assert h_back == h_in


# --- hijri_difference_days ---------------------


def test_difference_zero_for_same_date():
    h = HijriDate(year=1447, month=HijriMonth.MUHARRAM, day=1)
    assert hijri_difference_days(h, h) == 0


def test_difference_positive_when_b_after_a():
    a = HijriDate(year=1447, month=HijriMonth.MUHARRAM, day=1)
    b = HijriDate(year=1447, month=HijriMonth.MUHARRAM, day=15)
    assert hijri_difference_days(a, b) == 14


def test_difference_negative_when_b_before_a():
    a = HijriDate(year=1447, month=HijriMonth.MUHARRAM, day=15)
    b = HijriDate(year=1447, month=HijriMonth.MUHARRAM, day=1)
    assert hijri_difference_days(a, b) == -14


def test_difference_across_year():
    a = HijriDate(year=1447, month=HijriMonth.MUHARRAM, day=1)
    b = HijriDate(year=1448, month=HijriMonth.MUHARRAM, day=1)
    assert hijri_difference_days(a, b) == hijri_year_length(1447)


# --- add_days ----------------------------------


def test_add_days_zero_noop():
    d = HijriDate(year=1447, month=HijriMonth.MUHARRAM, day=1)
    assert add_days(d, 0) == d


def test_add_days_30_advances_month():
    d = HijriDate(year=1447, month=HijriMonth.MUHARRAM, day=1)
    # Muharram has 30 days; +30 → 1 Safar.
    new = add_days(d, 30)
    assert new == HijriDate(year=1447, month=HijriMonth.SAFAR, day=1)


def test_add_days_negative():
    d = HijriDate(year=1447, month=HijriMonth.SAFAR, day=1)
    # -1 → 30 Muharram.
    new = add_days(d, -1)
    assert new == HijriDate(year=1447, month=HijriMonth.MUHARRAM, day=30)


def test_add_days_across_year():
    d = HijriDate(year=1447, month=HijriMonth.MUHARRAM, day=1)
    new = add_days(d, hijri_year_length(1447))
    assert new == HijriDate(year=1448, month=HijriMonth.MUHARRAM, day=1)


# --- month_name / Render -----------------------


def test_month_name_returns_english():
    assert month_name(HijriMonth.RAMADAN) == "Ramadan"
    assert month_name(HijriMonth.MUHARRAM) == "Muharram"


def test_render_hijri_format():
    d = HijriDate(year=1447, month=HijriMonth.RAMADAN, day=15)
    out = render_hijri(d)
    assert out == "15 Ramadan 1447 AH"


def test_render_dual_uses_both():
    g = date(2026, 5, 11)
    out = render_dual(g)
    assert "2026-05-11" in out
    assert "AH" in out


def test_render_dual_with_explicit_hijri():
    g = date(2026, 5, 11)
    h = HijriDate(year=1447, month=HijriMonth.DHU_AL_QADAH, day=23)
    out = render_dual(g, h)
    assert "23 Dhu al-Qa'dah 1447 AH" in out


# --- Edge / sanity --------------------------------


def test_hijri_epoch_maps_to_known_gregorian():
    """1 Muharram 1 AH (tabular) ≈ 19 July 622 CE proleptic Gregorian."""
    h = HijriDate(year=1, month=HijriMonth.MUHARRAM, day=1)
    g = hijri_to_gregorian(h)
    assert g.year == 622
    assert g.month == 7
    # Day can be 19 ± 1 due to tabular calendar conventions; we pin to 19.
    assert g.day == 19


def test_hijri_year_lengths_alternate_correctly():
    """Pin: years follow the 11-of-30 leap pattern → 354/355 mix."""
    leap_count = sum(1 for y in range(1, 31) if is_hijri_leap_year(y))
    assert leap_count == 11
