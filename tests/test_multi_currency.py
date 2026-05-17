"""Tests for core/multi_currency.py — Round-5 Wave 18.H."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.core.multi_currency import (
    CurrencyAmount,
    FxRate,
    FxRateBook,
    TranslationMethod,
    render_translation,
    total_in,
    translate,
    translate_batch,
)


def _book_with_rates() -> FxRateBook:
    book = FxRateBook()
    book.add(FxRate("USD", "SAR", 3.75, date(2026, 5, 1)))
    book.add(FxRate("USD", "EUR", 0.92, date(2026, 5, 1)))
    book.add(FxRate("USD", "GBP", 0.78, date(2026, 5, 1)))
    book.add(FxRate("USD", "MYR", 4.50, date(2026, 5, 1)))
    return book


# --- Validation ----------------------------------------------------


def test_translation_method_string_values():
    assert TranslationMethod.SPOT_AT_TRADE.value == "spot_at_trade"
    assert TranslationMethod.DAILY_AVG.value == "daily_avg"
    assert TranslationMethod.MONTHLY_AVG.value == "monthly_avg"
    assert TranslationMethod.YEAR_END_SPOT.value == "year_end_spot"
    assert TranslationMethod.ZAKAT_DATE_SPOT.value == "zakat_date_spot"


def test_fx_rate_short_currency_rejected():
    with pytest.raises(ValueError):
        FxRate("US", "SAR", 3.75, date(2026, 5, 1))


def test_fx_rate_zero_rate_rejected():
    with pytest.raises(ValueError):
        FxRate("USD", "SAR", 0.0, date(2026, 5, 1))


def test_fx_rate_same_currency_must_be_one():
    with pytest.raises(ValueError):
        FxRate("USD", "USD", 1.5, date(2026, 5, 1))


def test_amount_short_currency_rejected():
    with pytest.raises(ValueError):
        CurrencyAmount(amount=100, currency="US", as_of_date=date(2026, 5, 1))


# --- Rate book -----------------------------------------------------


def test_book_add_and_lookup():
    book = FxRateBook()
    book.add(FxRate("USD", "EUR", 0.92, date(2026, 5, 1)))
    assert book.lookup("USD", "EUR", date(2026, 5, 1)) == 0.92


def test_book_same_currency_unity():
    book = FxRateBook()
    assert book.lookup("USD", "USD", date(2026, 5, 1)) == 1.0


def test_book_inverse_lookup():
    book = FxRateBook()
    book.add(FxRate("USD", "EUR", 0.92, date(2026, 5, 1)))
    inv = book.lookup("EUR", "USD", date(2026, 5, 1))
    assert inv == pytest.approx(1 / 0.92)


def test_book_missing_rate_raises():
    book = FxRateBook()
    with pytest.raises(KeyError):
        book.lookup("USD", "ZZZ", date(2026, 5, 1))


def test_book_idempotent_same_value():
    """Re-adding the same rate is a no-op."""
    book = FxRateBook()
    book.add(FxRate("USD", "EUR", 0.92, date(2026, 5, 1)))
    book.add(FxRate("USD", "EUR", 0.92, date(2026, 5, 1)))
    assert book.lookup("USD", "EUR", date(2026, 5, 1)) == 0.92


def test_book_overwrite_different_value_rejected():
    book = FxRateBook()
    book.add(FxRate("USD", "EUR", 0.92, date(2026, 5, 1)))
    with pytest.raises(ValueError):
        book.add(FxRate("USD", "EUR", 0.95, date(2026, 5, 1)))


def test_book_different_dates_independent():
    book = FxRateBook()
    book.add(FxRate("USD", "EUR", 0.92, date(2026, 5, 1)))
    book.add(FxRate("USD", "EUR", 0.93, date(2026, 5, 2)))
    assert book.lookup("USD", "EUR", date(2026, 5, 1)) == 0.92
    assert book.lookup("USD", "EUR", date(2026, 5, 2)) == 0.93


# --- Translation -------------------------------------------------


def test_translate_basic():
    book = _book_with_rates()
    amount = CurrencyAmount(amount=100, currency="USD", as_of_date=date(2026, 5, 1))
    result = translate(amount, target_currency="SAR", book=book)
    assert result.amount == pytest.approx(375.0)
    assert result.currency == "SAR"


def test_translate_same_currency_unchanged():
    book = _book_with_rates()
    amount = CurrencyAmount(amount=100, currency="USD", as_of_date=date(2026, 5, 1))
    result = translate(amount, target_currency="USD", book=book)
    assert result.amount == 100


def test_translate_inverse_via_book():
    book = _book_with_rates()
    amount = CurrencyAmount(amount=375, currency="SAR", as_of_date=date(2026, 5, 1))
    result = translate(amount, target_currency="USD", book=book)
    assert result.amount == pytest.approx(100.0)


def test_translate_missing_rate_raises():
    book = _book_with_rates()
    amount = CurrencyAmount(amount=100, currency="USD", as_of_date=date(2026, 5, 2))
    with pytest.raises(KeyError):
        translate(amount, target_currency="SAR", book=book)


def test_translate_short_target_rejected():
    book = _book_with_rates()
    amount = CurrencyAmount(amount=100, currency="USD", as_of_date=date(2026, 5, 1))
    with pytest.raises(ValueError):
        translate(amount, target_currency="US", book=book)


def test_translate_with_override_date_uses_override():
    book = FxRateBook()
    book.add(FxRate("USD", "EUR", 0.92, date(2026, 5, 1)))
    book.add(FxRate("USD", "EUR", 0.95, date(2026, 12, 31)))  # year-end
    amount = CurrencyAmount(amount=100, currency="USD", as_of_date=date(2026, 5, 1))
    yearend = translate(
        amount,
        target_currency="EUR",
        book=book,
        method=TranslationMethod.YEAR_END_SPOT,
        override_date=date(2026, 12, 31),
    )
    assert yearend.amount == pytest.approx(95.0)


def test_translate_batch():
    book = _book_with_rates()
    amounts = [
        CurrencyAmount(100, "USD", date(2026, 5, 1)),
        CurrencyAmount(200, "EUR", date(2026, 5, 1)),
    ]
    results = translate_batch(amounts, target_currency="USD", book=book)
    assert results[0].amount == 100
    assert results[1].amount == pytest.approx(200 / 0.92)


# --- Total ---------------------------------------------------------


def test_total_in_sums_heterogeneous():
    book = _book_with_rates()
    amounts = [
        CurrencyAmount(100, "USD", date(2026, 5, 1)),
        CurrencyAmount(375, "SAR", date(2026, 5, 1)),  # = $100
        CurrencyAmount(92, "EUR", date(2026, 5, 1)),  # = $100
    ]
    total = total_in(amounts, "USD", book=book)
    assert total == pytest.approx(300.0)


def test_total_in_empty_zero():
    book = _book_with_rates()
    assert total_in([], "USD", book=book) == 0


# --- Render --------------------------------------------------------


def test_render_translation():
    src = CurrencyAmount(100, "USD", date(2026, 5, 1))
    dst = CurrencyAmount(375, "SAR", date(2026, 5, 1))
    out = render_translation(src, dst, TranslationMethod.SPOT_AT_TRADE)
    assert "100" in out
    assert "USD" in out
    assert "SAR" in out
    assert "spot_at_trade" in out


# --- E2E ---------------------------------------------------


def test_e2e_saudi_diaspora_zakat_translation():
    """User holds SAR + USD + EUR; Zakat reporting in USD."""
    book = _book_with_rates()
    holdings = [
        CurrencyAmount(50000, "USD", date(2026, 5, 1)),
        CurrencyAmount(100000, "SAR", date(2026, 5, 1)),  # ~26.7k USD
        CurrencyAmount(20000, "EUR", date(2026, 5, 1)),  # ~21.7k USD
    ]
    total_usd = total_in(holdings, "USD", book=book)
    expected = 50000 + 100000 / 3.75 + 20000 / 0.92
    assert total_usd == pytest.approx(expected)


def test_replay_consistency():
    book = _book_with_rates()
    amount = CurrencyAmount(100, "USD", date(2026, 5, 1))
    a = translate(amount, target_currency="SAR", book=book)
    b = translate(amount, target_currency="SAR", book=book)
    assert a == b
