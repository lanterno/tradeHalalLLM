"""Tests for i18n/locale_format.py — Round-5 Wave 23."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.i18n.locale_format import (
    DateFormat,
    DigitShape,
    LocaleFormat,
    format_currency,
    format_date,
    format_for,
    format_number,
    shape_digits,
)
from halal_trader.i18n.translations import Locale

# --- LocaleFormat validation ---------------------


def test_format_for_returns_profile():
    p = format_for(Locale.EN)
    assert p.decimal_sep == "."


def test_format_for_covers_every_locale():
    """Pin: every Locale has a format profile."""
    for L in Locale:
        format_for(L)


def test_invalid_format_decimal_thousands_collision():
    with pytest.raises(ValueError):
        LocaleFormat(
            locale=Locale.EN,
            digit_shape=DigitShape.WESTERN_ARABIC,
            decimal_sep=",",
            thousands_sep=",",
            currency_symbol="$",
            currency_after=False,
            date_format=DateFormat.MDY,
        )


def test_invalid_format_empty_decimal():
    with pytest.raises(ValueError):
        LocaleFormat(
            locale=Locale.EN,
            digit_shape=DigitShape.WESTERN_ARABIC,
            decimal_sep="",
            thousands_sep=",",
            currency_symbol="$",
            currency_after=False,
            date_format=DateFormat.MDY,
        )


def test_invalid_format_empty_currency():
    with pytest.raises(ValueError):
        LocaleFormat(
            locale=Locale.EN,
            digit_shape=DigitShape.WESTERN_ARABIC,
            decimal_sep=".",
            thousands_sep=",",
            currency_symbol="",
            currency_after=False,
            date_format=DateFormat.MDY,
        )


# --- format_number — basic ----------------------


def test_format_number_en():
    assert format_number(1234.56, Locale.EN) == "1,234.56"


def test_format_number_id_uses_dot_thousands_comma_decimal():
    assert format_number(1234.56, Locale.ID) == "1.234,56"


def test_format_number_fr_space_thousands_comma_decimal():
    assert format_number(1234.56, Locale.FR) == "1 234,56"


def test_format_number_es_dot_thousands_comma_decimal():
    assert format_number(1234.56, Locale.ES) == "1.234,56"


def test_format_number_tr_dot_thousands_comma_decimal():
    assert format_number(1234.56, Locale.TR) == "1.234,56"


# --- format_number — Arabic-Indic digits --------


def test_format_number_arabic_uses_arabic_indic():
    out = format_number(1234.56, Locale.AR)
    # Should contain Arabic-Indic digits.
    assert "١" in out  # digit "1"
    assert "٢" in out  # digit "2"


def test_format_number_urdu_uses_extended_arabic_indic():
    out = format_number(1234.56, Locale.UR)
    # Extended Arabic-Indic ۱-۹.
    assert "۱" in out
    assert "۲" in out


def test_format_number_bengali_uses_bengali_digits():
    out = format_number(1234.56, Locale.BN)
    assert "১" in out
    assert "২" in out


# --- format_number — negative -------------------


def test_format_number_negative_has_minus():
    assert format_number(-100, Locale.EN, decimals=0) == "-100"


def test_format_number_zero():
    assert format_number(0, Locale.EN, decimals=2) == "0.00"


# --- format_number — decimals -------------------


def test_format_number_decimals_override():
    assert format_number(1.2345, Locale.EN, decimals=4) == "1.2345"


def test_format_number_zero_decimals():
    assert format_number(1234.56, Locale.EN, decimals=0) == "1,235"


def test_format_number_negative_decimals_rejected():
    with pytest.raises(ValueError):
        format_number(1.0, Locale.EN, decimals=-1)


def test_format_number_excessive_decimals_rejected():
    with pytest.raises(ValueError):
        format_number(1.0, Locale.EN, decimals=20)


# --- format_currency ----------------------------


def test_currency_en_dollar_before():
    assert format_currency(1234.56, Locale.EN) == "$1,234.56"


def test_currency_fr_euro_after():
    assert format_currency(1234.56, Locale.FR) == "1 234,56 €"


def test_currency_ar_riyal_after():
    out = format_currency(1234.56, Locale.AR)
    assert out.endswith("ر.س")
    assert "،" not in out  # uses Arabic thousands separator instead


def test_currency_tr_lira_after():
    assert format_currency(1234.56, Locale.TR) == "1.234,56 ₺"


# --- format_date — variants ---------------------


def test_format_date_iso():
    out = format_date(date(2026, 5, 11), Locale.EN, override_format=DateFormat.ISO)
    assert out == "2026-05-11"


def test_format_date_dmy():
    out = format_date(date(2026, 5, 11), Locale.EN, override_format=DateFormat.DMY)
    assert out == "11/05/2026"


def test_format_date_mdy():
    out = format_date(date(2026, 5, 11), Locale.EN, override_format=DateFormat.MDY)
    assert out == "05/11/2026"


def test_format_date_text_dmy():
    out = format_date(date(2026, 5, 11), Locale.EN, override_format=DateFormat.TEXT_DMY)
    assert out == "11 May 2026"


def test_format_date_default_per_locale_en_mdy():
    out = format_date(date(2026, 5, 11), Locale.EN)
    assert out == "05/11/2026"


def test_format_date_default_per_locale_fr_dmy():
    out = format_date(date(2026, 5, 11), Locale.FR)
    assert out == "11/05/2026"


def test_format_date_arabic_uses_arabic_digits():
    out = format_date(date(2026, 5, 11), Locale.AR, override_format=DateFormat.ISO)
    assert "٢" in out


def test_format_date_bengali_uses_bengali_digits():
    out = format_date(date(2026, 5, 11), Locale.BN, override_format=DateFormat.ISO)
    assert "২" in out


# --- shape_digits -------------------------------


def test_shape_digits_en_unchanged():
    assert shape_digits("123abc", Locale.EN) == "123abc"


def test_shape_digits_arabic():
    out = shape_digits("123abc", Locale.AR)
    assert "١" in out
    assert "abc" in out  # non-digit pass through


def test_shape_digits_persian():
    out = shape_digits("123abc", Locale.FA)
    assert "۱" in out


def test_shape_digits_bengali():
    out = shape_digits("123abc", Locale.BN)
    assert "১" in out


# --- DigitShape exhaustiveness -----------------


def test_digit_shape_western():
    out = shape_digits("0123456789", Locale.EN)
    assert out == "0123456789"


def test_digit_shape_arabic_indic_all_digits():
    out = shape_digits("0123456789", Locale.AR)
    expected = "٠١٢٣٤٥٦٧٨٩"
    assert out == expected


def test_digit_shape_extended_arabic_indic_all_digits():
    out = shape_digits("0123456789", Locale.FA)
    expected = "۰۱۲۳۴۵۶۷۸۹"
    assert out == expected


def test_digit_shape_bengali_all_digits():
    out = shape_digits("0123456789", Locale.BN)
    expected = "০১২৩৪৫৬৭৮৯"
    assert out == expected


# --- LocaleFormat dataclass invariants --------


def test_locale_format_is_frozen():
    p = format_for(Locale.EN)
    with pytest.raises(AttributeError):
        p.decimal_sep = "x"  # type: ignore[misc]
