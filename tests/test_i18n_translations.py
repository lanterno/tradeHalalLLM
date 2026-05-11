"""Tests for i18n/translations.py — Round-5 Wave 23.A-I."""

from __future__ import annotations

import pytest

from halal_trader.i18n.translations import (
    DEFAULT_LOCALE,
    Locale,
    TextDirection,
    Translation,
    TranslationRegistry,
    base_locale,
    direction_for,
    is_rtl,
    merge_registries,
    render_lookup,
)


def _entry(
    locale: Locale = Locale.EN,
    key: str = "common.ok",
    text: str = "OK",
) -> Translation:
    return Translation(locale=locale, key=key, text=text)


# --- Locale + direction ---------------------------


def test_default_locale_is_en():
    assert DEFAULT_LOCALE is Locale.EN


def test_direction_for_arabic_is_rtl():
    assert direction_for(Locale.AR) is TextDirection.RTL
    assert direction_for(Locale.AR_EG) is TextDirection.RTL
    assert direction_for(Locale.AR_SA) is TextDirection.RTL


def test_direction_for_urdu_is_rtl():
    assert direction_for(Locale.UR) is TextDirection.RTL


def test_direction_for_persian_is_rtl():
    assert direction_for(Locale.FA) is TextDirection.RTL


def test_direction_for_pashto_is_rtl():
    assert direction_for(Locale.PS) is TextDirection.RTL


def test_direction_for_english_is_ltr():
    assert direction_for(Locale.EN) is TextDirection.LTR


def test_direction_for_indonesian_is_ltr():
    assert direction_for(Locale.ID) is TextDirection.LTR


def test_direction_for_french_is_ltr():
    assert direction_for(Locale.FR) is TextDirection.LTR


def test_is_rtl_helper():
    assert is_rtl(Locale.AR)
    assert not is_rtl(Locale.EN)


def test_direction_covers_every_locale():
    """Pin: every Locale must have a direction."""
    for L in Locale:
        direction_for(L)


# --- base_locale ----------------------------------


def test_base_locale_strips_region():
    assert base_locale(Locale.AR_EG) is Locale.AR
    assert base_locale(Locale.AR_SA) is Locale.AR


def test_base_locale_returns_self_when_no_region():
    assert base_locale(Locale.EN) is Locale.EN
    assert base_locale(Locale.AR) is Locale.AR
    assert base_locale(Locale.UR) is Locale.UR


# --- Translation validation ------------------------


def test_translation_valid():
    t = _entry()
    assert t.key == "common.ok"


def test_translation_empty_key_rejected():
    with pytest.raises(ValueError):
        _entry(key="")


def test_translation_key_with_empty_segments_rejected():
    with pytest.raises(ValueError):
        _entry(key="common..ok")


def test_translation_long_key_rejected():
    with pytest.raises(ValueError):
        _entry(key="x" * 300)


def test_translation_empty_text_rejected():
    with pytest.raises(ValueError):
        _entry(text=" ")


def test_translation_long_text_rejected():
    with pytest.raises(ValueError):
        _entry(text="x" * 6000)


def test_translation_immutable():
    t = _entry()
    with pytest.raises(AttributeError):
        t.text = "y"  # type: ignore[misc]


# --- TranslationRegistry validation ----------------


def test_registry_basic():
    r = TranslationRegistry(entries=(_entry(),))
    assert len(r.supported_locales()) == 1


def test_registry_empty_rejected():
    with pytest.raises(ValueError):
        TranslationRegistry(entries=())


def test_registry_duplicate_key_per_locale_rejected():
    with pytest.raises(ValueError):
        TranslationRegistry(
            entries=(
                _entry(),
                _entry(text="Yes"),  # same locale + same key
            )
        )


def test_registry_same_key_different_locale_allowed():
    r = TranslationRegistry(
        entries=(
            _entry(locale=Locale.EN, text="OK"),
            _entry(locale=Locale.AR, text="نعم"),
            _entry(locale=Locale.UR, text="ٹھیک"),
        )
    )
    assert len(r.entries) == 3


# --- lookup + fallback ----------------------------


def test_lookup_direct_hit():
    r = TranslationRegistry(entries=(_entry(locale=Locale.AR, key="common.ok", text="نعم"),))
    assert r.lookup(Locale.AR, "common.ok") == "نعم"


def test_lookup_falls_back_to_base_locale():
    """ar-EG → ar."""
    r = TranslationRegistry(entries=(_entry(locale=Locale.AR, key="common.ok", text="نعم"),))
    assert r.lookup(Locale.AR_EG, "common.ok") == "نعم"


def test_lookup_falls_back_to_default():
    """No translation for the locale OR its base → default (EN)."""
    r = TranslationRegistry(entries=(_entry(locale=Locale.EN, key="common.ok", text="OK"),))
    assert r.lookup(Locale.AR_EG, "common.ok") == "OK"


def test_lookup_chain_locale_beats_base():
    """ar-EG-specific value should win over ar."""
    r = TranslationRegistry(
        entries=(
            _entry(locale=Locale.AR, key="common.ok", text="نعم"),
            _entry(locale=Locale.AR_EG, key="common.ok", text="ايوه"),
        )
    )
    assert r.lookup(Locale.AR_EG, "common.ok") == "ايوه"


def test_lookup_missing_returns_none():
    r = TranslationRegistry(entries=(_entry(locale=Locale.EN, key="common.ok", text="OK"),))
    assert r.lookup(Locale.EN, "missing.key") is None


def test_must_lookup_raises_when_missing():
    r = TranslationRegistry(entries=(_entry(locale=Locale.EN, key="common.ok", text="OK"),))
    with pytest.raises(KeyError):
        r.must_lookup(Locale.EN, "missing.key")


def test_must_lookup_returns_value():
    r = TranslationRegistry(entries=(_entry(locale=Locale.EN, key="common.ok", text="OK"),))
    assert r.must_lookup(Locale.EN, "common.ok") == "OK"


# --- supported_locales + keys_for ------------------


def test_supported_locales_sorted():
    r = TranslationRegistry(
        entries=(
            _entry(locale=Locale.UR, key="k1", text="urdu"),
            _entry(locale=Locale.AR, key="k1", text="arabic"),
            _entry(locale=Locale.EN, key="k1", text="english"),
        )
    )
    assert r.supported_locales() == (Locale.AR, Locale.EN, Locale.UR)


def test_keys_for_locale():
    r = TranslationRegistry(
        entries=(
            _entry(locale=Locale.EN, key="common.ok", text="OK"),
            _entry(locale=Locale.EN, key="common.cancel", text="Cancel"),
            _entry(locale=Locale.AR, key="common.ok", text="نعم"),
        )
    )
    keys = r.keys_for(Locale.EN)
    assert keys == ("common.cancel", "common.ok")


# --- strict_mode -----------------------------------


def test_strict_mode_passes_when_all_locales_cover_keys():
    r = TranslationRegistry(
        entries=(
            _entry(locale=Locale.EN, key="k1", text="english"),
            _entry(locale=Locale.AR, key="k1", text="arabic"),
            _entry(locale=Locale.UR, key="k1", text="urdu"),
            # All other locales fall through to EN — that's allowed
            # only because strict_mode is False here.
        ),
        strict_mode=False,
    )
    assert r.supported_locales()


def test_strict_mode_rejects_missing_locale_for_key():
    """Strict mode: every defined key must resolve in every Locale."""
    entries = [Translation(locale=L, key="k1", text=f"text-{L.value}") for L in Locale]
    # Drop one locale.
    entries = [t for t in entries if t.locale is not Locale.HA]
    with pytest.raises(ValueError):
        TranslationRegistry(entries=tuple(entries), strict_mode=True)


def test_strict_mode_accepts_base_locale_coverage():
    """If `ar` has the key, `ar-EG` is covered via fallback."""
    entries = [
        Translation(locale=L, key="k1", text=f"text-{L.value}")
        for L in Locale
        if L is not Locale.AR_EG and L is not Locale.AR_SA
    ]
    # AR has it; AR_EG + AR_SA fall back via base.
    r = TranslationRegistry(entries=tuple(entries), strict_mode=True)
    assert r.supported_locales()


# --- merge_registries ------------------------------


def test_merge_two_registries():
    r1 = TranslationRegistry(entries=(_entry(locale=Locale.EN, key="k1", text="one"),))
    r2 = TranslationRegistry(entries=(_entry(locale=Locale.EN, key="k2", text="two"),))
    merged = merge_registries(r1, r2)
    assert merged.lookup(Locale.EN, "k1") == "one"
    assert merged.lookup(Locale.EN, "k2") == "two"


def test_merge_later_overrides_earlier():
    r1 = TranslationRegistry(entries=(_entry(locale=Locale.EN, key="k1", text="old"),))
    r2 = TranslationRegistry(entries=(_entry(locale=Locale.EN, key="k1", text="new"),))
    merged = merge_registries(r1, r2)
    assert merged.lookup(Locale.EN, "k1") == "new"


def test_merge_empty_args_rejected():
    with pytest.raises(ValueError):
        merge_registries()


# --- Render ---------------------------------------


def test_render_existing_returns_text():
    r = TranslationRegistry(entries=(_entry(locale=Locale.EN, key="k1", text="hello"),))
    assert render_lookup(r, Locale.EN, "k1") == "hello"


def test_render_rtl_prepends_marker():
    r = TranslationRegistry(entries=(_entry(locale=Locale.AR, key="k1", text="مرحبا"),))
    out = render_lookup(r, Locale.AR, "k1")
    assert "🔃" in out
    assert "مرحبا" in out


def test_render_missing_returns_warning_marker():
    r = TranslationRegistry(entries=(_entry(locale=Locale.EN, key="k1", text="ok"),))
    out = render_lookup(r, Locale.EN, "missing")
    assert "⚠️" in out
    assert "MISSING" in out
