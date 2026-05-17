"""Tests for the localisation engine."""

from __future__ import annotations

import dataclasses
import warnings
from datetime import UTC, datetime

import pytest

from halal_trader.web.i18n import (
    Locale,
    LocaleProfile,
    MessageCatalog,
    default_profile,
    format_currency,
    format_date,
    format_number,
    format_percent,
    is_rtl,
    is_translation_complete,
    render_locale_profile,
    translate,
)

_NOW = datetime(2026, 5, 1, tzinfo=UTC)


def _en_catalog() -> MessageCatalog:
    return MessageCatalog(
        locale=Locale.EN,
        messages={
            "welcome": "Welcome",
            "dashboard.title": "Dashboard",
            "trade.confirm": "Confirm trade for {symbol} at {price}",
            "halal.permissible": "Permissible per AAOIFI Standard 21",
        },
    )


def _ar_catalog() -> MessageCatalog:
    return MessageCatalog(
        locale=Locale.AR,
        messages={
            "welcome": "أهلا وسهلا",
            "dashboard.title": "لوحة التحكم",
            "trade.confirm": "تأكيد التداول لـ {symbol} بسعر {price}",
        },
    )


# ---------------------------------------------------------------------------
# Locale + RTL flag
# ---------------------------------------------------------------------------


def test_locale_string_values() -> None:
    assert Locale.EN.value == "en"
    assert Locale.AR.value == "ar"
    assert Locale.UR.value == "ur"
    assert Locale.MS.value == "ms"
    assert Locale.ID.value == "id"
    assert Locale.FR.value == "fr"
    assert Locale.TR.value == "tr"


def test_arabic_is_rtl() -> None:
    assert is_rtl(Locale.AR) is True


def test_urdu_is_rtl() -> None:
    assert is_rtl(Locale.UR) is True


def test_english_is_ltr() -> None:
    assert is_rtl(Locale.EN) is False


def test_malay_is_ltr() -> None:
    assert is_rtl(Locale.MS) is False


def test_french_is_ltr() -> None:
    assert is_rtl(Locale.FR) is False


def test_turkish_is_ltr() -> None:
    assert is_rtl(Locale.TR) is False


# ---------------------------------------------------------------------------
# LocaleProfile validation
# ---------------------------------------------------------------------------


def test_default_profile_for_every_locale() -> None:
    for loc in Locale:
        p = default_profile(loc)
        assert p.locale is loc


def test_default_profile_arabic_is_rtl() -> None:
    p = default_profile(Locale.AR)
    assert p.is_rtl is True
    assert p.currency_code == "SAR"
    assert p.currency_symbol == "ر.س"
    assert p.currency_prefix is False


def test_default_profile_english_is_ltr() -> None:
    p = default_profile(Locale.EN)
    assert p.is_rtl is False
    assert p.currency_code == "USD"
    assert p.currency_symbol == "$"
    assert p.currency_prefix is True


def test_default_profile_indonesian_uses_european_decimals() -> None:
    """Pin: ID uses `.` as thousands separator and `,` as decimal."""

    p = default_profile(Locale.ID)
    assert p.thousands_separator == "."
    assert p.decimal_separator == ","


def test_default_profile_french_uses_space_thousands() -> None:
    p = default_profile(Locale.FR)
    assert p.thousands_separator == " "
    assert p.decimal_separator == ","


def test_profile_rejects_invalid_currency_code() -> None:
    with pytest.raises(ValueError, match="currency_code"):
        LocaleProfile(
            locale=Locale.EN,
            is_rtl=False,
            currency_code="USDD",  # too long
            currency_symbol="$",
            currency_prefix=True,
            decimal_separator=".",
            thousands_separator=",",
        )


def test_profile_rejects_empty_currency_symbol() -> None:
    with pytest.raises(ValueError, match="currency_symbol"):
        LocaleProfile(
            locale=Locale.EN,
            is_rtl=False,
            currency_code="USD",
            currency_symbol="",
            currency_prefix=True,
            decimal_separator=".",
            thousands_separator=",",
        )


def test_profile_rejects_empty_decimal_separator() -> None:
    with pytest.raises(ValueError, match="decimal_separator"):
        LocaleProfile(
            locale=Locale.EN,
            is_rtl=False,
            currency_code="USD",
            currency_symbol="$",
            currency_prefix=True,
            decimal_separator="",
            thousands_separator=",",
        )


# ---------------------------------------------------------------------------
# MessageCatalog validation
# ---------------------------------------------------------------------------


def test_catalog_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="key must be non-empty"):
        MessageCatalog(locale=Locale.EN, messages={"": "Welcome"})


def test_catalog_rejects_non_string_value() -> None:
    with pytest.raises(ValueError, match="must be str"):
        MessageCatalog(locale=Locale.EN, messages={"welcome": 42})  # type: ignore[dict-item]


def test_catalog_has_returns_correct() -> None:
    c = _en_catalog()
    assert c.has("welcome") is True
    assert c.has("missing") is False


def test_catalog_get_returns_string() -> None:
    c = _en_catalog()
    assert c.get("welcome") == "Welcome"
    assert c.get("missing") is None


# ---------------------------------------------------------------------------
# translate happy path
# ---------------------------------------------------------------------------


def test_translate_english() -> None:
    catalogs = {Locale.EN: _en_catalog()}
    result = translate(key="welcome", locale=Locale.EN, catalogs=catalogs)
    assert result == "Welcome"


def test_translate_arabic() -> None:
    catalogs = {Locale.EN: _en_catalog(), Locale.AR: _ar_catalog()}
    result = translate(key="welcome", locale=Locale.AR, catalogs=catalogs)
    assert result == "أهلا وسهلا"


def test_translate_with_placeholders() -> None:
    catalogs = {Locale.EN: _en_catalog()}
    result = translate(
        key="trade.confirm",
        locale=Locale.EN,
        catalogs=catalogs,
        symbol="AAPL",
        price="$190.50",
    )
    assert result == "Confirm trade for AAPL at $190.50"


def test_translate_arabic_with_placeholders() -> None:
    catalogs = {Locale.EN: _en_catalog(), Locale.AR: _ar_catalog()}
    result = translate(
        key="trade.confirm",
        locale=Locale.AR,
        catalogs=catalogs,
        symbol="AAPL",
        price="$190.50",
    )
    assert "AAPL" in result
    assert "$190.50" in result


# ---------------------------------------------------------------------------
# Fallback semantics
# ---------------------------------------------------------------------------


def test_missing_translation_falls_back_to_english() -> None:
    """Pin: AR catalog missing 'halal.permissible' falls back to EN."""

    catalogs = {Locale.EN: _en_catalog(), Locale.AR: _ar_catalog()}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = translate(key="halal.permissible", locale=Locale.AR, catalogs=catalogs)
    assert result == "Permissible per AAOIFI Standard 21"
    # confirm a warning was emitted
    assert any("missing" in str(x.message) for x in w)


def test_missing_locale_entirely_falls_back_to_english() -> None:
    """Pin: a locale with no catalog falls back to English."""

    catalogs = {Locale.EN: _en_catalog()}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = translate(key="welcome", locale=Locale.AR, catalogs=catalogs)
    assert result == "Welcome"
    assert any("missing" in str(x.message) for x in w)


def test_missing_in_both_returns_key_with_warning() -> None:
    """Pin: missing in both target + fallback → returns key, warns."""

    catalogs = {Locale.EN: _en_catalog()}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = translate(key="totally.missing.key", locale=Locale.AR, catalogs=catalogs)
    assert result == "totally.missing.key"
    assert any("returning key as-is" in str(x.message) for x in w)


def test_missing_translation_never_returns_empty() -> None:
    """Pin: a missing translation NEVER returns empty string."""

    catalogs: dict[Locale, MessageCatalog] = {}
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = translate(key="any_key", locale=Locale.AR, catalogs=catalogs)
    assert result != ""
    assert result == "any_key"


def test_english_target_does_not_warn_about_fallback() -> None:
    """Pin: when target locale IS the fallback, no fallback warning."""

    catalogs = {Locale.EN: _en_catalog()}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        translate(key="welcome", locale=Locale.EN, catalogs=catalogs)
    # No warnings expected since the target is the fallback and the key exists
    assert len(w) == 0


# ---------------------------------------------------------------------------
# Placeholder validation
# ---------------------------------------------------------------------------


def test_translate_rejects_empty_key() -> None:
    catalogs = {Locale.EN: _en_catalog()}
    with pytest.raises(ValueError, match="key must be non-empty"):
        translate(key="", locale=Locale.EN, catalogs=catalogs)


def test_translate_rejects_forbidden_placeholder_password() -> None:
    """Pin: i18n templates can't be used as a secret-leakage vector."""

    catalogs = {Locale.EN: _en_catalog()}
    with pytest.raises(ValueError, match="forbidden"):
        translate(
            key="welcome",
            locale=Locale.EN,
            catalogs=catalogs,
            password="hunter2",
        )


def test_translate_rejects_forbidden_placeholder_api_key() -> None:
    catalogs = {Locale.EN: _en_catalog()}
    with pytest.raises(ValueError, match="forbidden"):
        translate(
            key="welcome",
            locale=Locale.EN,
            catalogs=catalogs,
            api_key="sk-abc123",
        )


def test_translate_rejects_forbidden_placeholder_secret() -> None:
    catalogs = {Locale.EN: _en_catalog()}
    with pytest.raises(ValueError, match="forbidden"):
        translate(
            key="welcome",
            locale=Locale.EN,
            catalogs=catalogs,
            secret="xxx",
        )


def test_translate_rejects_forbidden_placeholder_token() -> None:
    catalogs = {Locale.EN: _en_catalog()}
    with pytest.raises(ValueError, match="forbidden"):
        translate(
            key="welcome",
            locale=Locale.EN,
            catalogs=catalogs,
            token="abc",
        )


def test_translate_raises_on_missing_placeholder() -> None:
    """Pin: template references {symbol} but caller didn't provide it."""

    catalogs = {Locale.EN: _en_catalog()}
    with pytest.raises(KeyError, match="placeholder"):
        translate(
            key="trade.confirm",
            locale=Locale.EN,
            catalogs=catalogs,
            symbol="AAPL",
            # price missing
        )


# ---------------------------------------------------------------------------
# format_number
# ---------------------------------------------------------------------------


def test_format_number_english() -> None:
    p = default_profile(Locale.EN)
    assert format_number(1234567.89, profile=p) == "1,234,567.89"


def test_format_number_indonesian_swaps_separators() -> None:
    """Pin: ID uses `.` thousands and `,` decimal — opposite of English."""

    p = default_profile(Locale.ID)
    assert format_number(1234567.89, profile=p) == "1.234.567,89"


def test_format_number_french_uses_space_thousands() -> None:
    p = default_profile(Locale.FR)
    assert format_number(1234567.89, profile=p) == "1 234 567,89"


def test_format_number_zero_decimals() -> None:
    p = default_profile(Locale.EN)
    assert format_number(1234, profile=p, decimals=0) == "1,234"


def test_format_number_rejects_negative_decimals() -> None:
    p = default_profile(Locale.EN)
    with pytest.raises(ValueError, match="decimals"):
        format_number(1234, profile=p, decimals=-1)


def test_format_number_handles_negative_value() -> None:
    p = default_profile(Locale.EN)
    assert format_number(-1234.56, profile=p) == "-1,234.56"


def test_format_number_handles_zero() -> None:
    p = default_profile(Locale.EN)
    assert format_number(0, profile=p) == "0.00"


# ---------------------------------------------------------------------------
# format_currency
# ---------------------------------------------------------------------------


def test_format_currency_english_uses_prefix() -> None:
    p = default_profile(Locale.EN)
    assert format_currency(1000, profile=p) == "$1,000.00"


def test_format_currency_arabic_uses_suffix() -> None:
    """Pin: AR places currency symbol after the number."""

    p = default_profile(Locale.AR)
    result = format_currency(1000, profile=p)
    assert "ر.س" in result
    # symbol comes after
    assert result.endswith("ر.س")


def test_format_currency_french_uses_suffix() -> None:
    p = default_profile(Locale.FR)
    result = format_currency(1000, profile=p)
    assert "€" in result
    assert result.endswith("€")


def test_format_currency_indonesian_prefix_with_swapped_separators() -> None:
    p = default_profile(Locale.ID)
    assert format_currency(1234567, profile=p) == "Rp1.234.567,00"


def test_format_currency_zero_decimals() -> None:
    p = default_profile(Locale.EN)
    assert format_currency(1000, profile=p, decimals=0) == "$1,000"


# ---------------------------------------------------------------------------
# format_percent
# ---------------------------------------------------------------------------


def test_format_percent_english() -> None:
    p = default_profile(Locale.EN)
    assert format_percent(0.05, profile=p) == "5.00%"


def test_format_percent_arabic() -> None:
    p = default_profile(Locale.AR)
    assert format_percent(0.05, profile=p) == "5.00%"


def test_format_percent_french_uses_comma_decimal() -> None:
    p = default_profile(Locale.FR)
    assert format_percent(0.05, profile=p) == "5,00%"


def test_format_percent_handles_negative() -> None:
    p = default_profile(Locale.EN)
    assert format_percent(-0.025, profile=p) == "-2.50%"


def test_format_percent_handles_above_100() -> None:
    p = default_profile(Locale.EN)
    assert format_percent(2.5, profile=p) == "250.00%"


# ---------------------------------------------------------------------------
# format_date
# ---------------------------------------------------------------------------


def test_format_date_english_iso() -> None:
    p = default_profile(Locale.EN)
    assert format_date(_NOW, profile=p) == "2026-05-01"


def test_format_date_french_iso() -> None:
    p = default_profile(Locale.FR)
    assert format_date(_NOW, profile=p) == "2026-05-01"


def test_format_date_arabic_dmy() -> None:
    """Pin: AR uses DD-MM-YYYY (Arabic-readable order)."""

    p = default_profile(Locale.AR)
    assert format_date(_NOW, profile=p) == "01-05-2026"


def test_format_date_urdu_dmy() -> None:
    p = default_profile(Locale.UR)
    assert format_date(_NOW, profile=p) == "01-05-2026"


def test_format_date_rejects_naive_datetime() -> None:
    p = default_profile(Locale.EN)
    with pytest.raises(ValueError, match="timezone-aware"):
        format_date(datetime(2026, 5, 1), profile=p)


# ---------------------------------------------------------------------------
# is_translation_complete
# ---------------------------------------------------------------------------


def test_is_translation_complete_returns_true_when_all_present() -> None:
    catalog = _en_catalog()
    is_done, missing = is_translation_complete(
        catalog=catalog,
        required_keys=("welcome", "dashboard.title"),
    )
    assert is_done is True
    assert missing == ()


def test_is_translation_complete_returns_missing_keys() -> None:
    catalog = _en_catalog()
    is_done, missing = is_translation_complete(
        catalog=catalog,
        required_keys=("welcome", "missing.one", "missing.two"),
    )
    assert is_done is False
    assert "missing.one" in missing
    assert "missing.two" in missing
    assert "welcome" not in missing


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_locale_profile_is_frozen() -> None:
    p = default_profile(Locale.EN)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.currency_code = "EUR"  # type: ignore[misc]


def test_message_catalog_is_frozen() -> None:
    c = _en_catalog()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.locale = Locale.AR  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Render output
# ---------------------------------------------------------------------------


def test_render_locale_profile_english() -> None:
    p = default_profile(Locale.EN)
    text = render_locale_profile(p)
    assert "🌐" in text
    assert "en" in text
    assert "LTR" in text
    assert "USD" in text
    assert "$" in text


def test_render_locale_profile_arabic() -> None:
    p = default_profile(Locale.AR)
    text = render_locale_profile(p)
    assert "ar" in text
    assert "RTL" in text
    assert "ر.س" in text
    assert "SAR" in text
    assert "suffix" in text


def test_render_locale_profile_french_shows_space_thousands() -> None:
    p = default_profile(Locale.FR)
    text = render_locale_profile(p)
    assert "1 234" in text  # space thousands separator visible


# ---------------------------------------------------------------------------
# End-to-end realistic scenarios
# ---------------------------------------------------------------------------


def test_arabic_user_journey() -> None:
    """A Saudi user sees Arabic UI, RTL layout, SAR currency formatting."""

    catalogs = {Locale.EN: _en_catalog(), Locale.AR: _ar_catalog()}
    profile = default_profile(Locale.AR)

    # Welcome message in Arabic
    welcome = translate(key="welcome", locale=Locale.AR, catalogs=catalogs)
    assert welcome == "أهلا وسهلا"

    # RTL flag for UI mirroring
    assert profile.is_rtl is True

    # Currency in SAR with suffix position
    price = format_currency(1000.50, profile=profile)
    assert "ر.س" in price
    assert price.endswith("ر.س")

    # Date in DD-MM-YYYY
    date = format_date(_NOW, profile=profile)
    assert date == "01-05-2026"


def test_indonesian_user_journey() -> None:
    """An Indonesian user sees Bahasa UI, LTR, IDR with European decimals."""

    profile = default_profile(Locale.ID)
    assert profile.is_rtl is False

    # Currency in IDR with European-style decimal/thousands
    price = format_currency(1234567, profile=profile)
    assert price == "Rp1.234.567,00"


def test_partially_translated_french_falls_back_per_key() -> None:
    """A FR catalog with only some keys → mixed-language UI per key."""

    fr_catalog = MessageCatalog(
        locale=Locale.FR,
        messages={"welcome": "Bienvenue"},  # only welcome translated
    )
    catalogs = {Locale.EN: _en_catalog(), Locale.FR: fr_catalog}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Translated key returns French
        assert translate(key="welcome", locale=Locale.FR, catalogs=catalogs) == "Bienvenue"
        # Untranslated key falls back to English
        assert translate(key="dashboard.title", locale=Locale.FR, catalogs=catalogs) == "Dashboard"
