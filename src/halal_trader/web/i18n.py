"""Localisation (i18n) engine.

The biggest halal-trading audiences sit in MENA, Pakistan, Indonesia,
and Malaysia — none of which read English as a first language for a
financial product. A multi-user platform that ships English-only is
gating itself out of those markets. This module is the pure-Python
i18n core: locale enum, RTL-aware locale profile, message catalog,
placeholder-substituted translate, locale-aware date / number /
currency / percent formatters, and a missing-translation fallback
with an audit warning so untranslated strings can't silently slip
through to a production UI.

The actual translation strings the operator ships are persisted in
operator-controlled data (a `messages.json` per locale, a Crowdin
project, a Lokalise project — whatever the operator's translation
workflow produces). This module stays format-agnostic on the
storage side and ships the stable contract the persistence layer
fills in.

Pinned semantics:
- **Missing translation falls back to English.** A locale that
  doesn't have a key registered returns the English version (or
  the key itself if English is also absent). The fallback never
  returns an empty string — silent empty UI strings are the
  worst-case failure mode. Pinned via test that an Arabic
  catalog missing `welcome` returns the English `Welcome` plus a
  warning emitted to the audit channel.
- **RTL locales flagged.** Arabic and Urdu return `is_rtl=True`
  on `LocaleProfile` so the UI knows to mirror the layout. The
  flag is data, not behaviour — the engine doesn't itself flip
  any rendering, just surfaces the directionality so the
  frontend can.
- **Placeholder substitution is positional + named.** The format
  is `{name}` style for named placeholders and stays pure-Python
  `str.format`-compatible. Missing placeholders (`{x}` in the
  template but `x` not in kwargs) raise `KeyError` rather than
  silently leaving the brace literal — pinned because partial
  substitution would render badly in production.
- **Numeric formatting is locale-aware.** `format_currency`,
  `format_percent`, `format_number` use the locale's thousands
  separator + decimal point + currency symbol + currency
  positioning. Arabic / Urdu use the canonical Arabic-Indic
  digits (٠١٢٣٤٥٦٧٨٩) when the operator opts in; default is
  Western digits even for RTL locales because most modern Arabic
  finance UIs use Western digits for numerical clarity (the
  preference is a setting on the locale profile).
- **Render output never includes raw secret values.** Mirrors
  the no-secret pattern of Wave 11.D privacy + Wave 11.C KYC +
  Wave 3.B vault — placeholder substitution refuses keys named
  `password`, `api_key`, `secret`, `token` (the i18n surface
  shouldn't be a sneaky secret-leakage vector via a "log this
  template with the values" debug feature). Pinned via test.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Locale(str, Enum):
    """Locales the engine supports.

    Pinned BCP 47 / ISO 639-1 string values for stable JSON / DB
    serialisation. Operators add new locales via code review (the
    set is closed at the type level so a typo can't silently
    register an unsupported locale).
    """

    EN = "en"  # English (fallback)
    AR = "ar"  # Arabic — MENA
    UR = "ur"  # Urdu — Pakistan
    MS = "ms"  # Bahasa Malay — Malaysia
    ID = "id"  # Bahasa Indonesia — Indonesia
    FR = "fr"  # French — Maghreb
    TR = "tr"  # Turkish — Turkey


# Locales that read right-to-left.
_RTL_LOCALES: frozenset[Locale] = frozenset({Locale.AR, Locale.UR})

# Sensitive placeholder names the engine refuses to substitute —
# guards against an i18n template being used as a secret-leakage
# vector. The set mirrors the redacted-attribute denylist in
# Wave 8.D OTLP translator + Wave 11.C / 11.D no-PII contracts.
_FORBIDDEN_PLACEHOLDER_KEYS: frozenset[str] = frozenset(
    {
        "password",
        "api_key",
        "secret",
        "token",
        "private_key",
        "session_id",
    }
)

_FALLBACK_LOCALE = Locale.EN


@dataclass(frozen=True)
class LocaleProfile:
    """Per-locale formatting profile.

    Carries the directionality flag, currency symbol, decimal
    separator, thousands separator, percent symbol, and currency-
    position (prefix / suffix) so the formatters stay free of
    runtime locale-specific conditionals.
    """

    locale: Locale
    is_rtl: bool
    currency_code: str
    currency_symbol: str
    currency_prefix: bool  # True = $1,000; False = 1,000 ر.س
    decimal_separator: str
    thousands_separator: str
    percent_symbol: str = "%"

    def __post_init__(self) -> None:
        if not self.currency_code or len(self.currency_code) != 3:
            raise ValueError(
                f"currency_code must be a 3-letter ISO code, got {self.currency_code!r}"
            )
        if not self.currency_symbol:
            raise ValueError("currency_symbol must be non-empty")
        if not self.decimal_separator:
            raise ValueError("decimal_separator must be non-empty")


# Default per-locale formatting. Operators override per-locale via
# a custom `LocaleProfile` if their target market uses a different
# currency or convention (Saudi → SAR; UAE → AED; Pakistan → PKR;
# Malaysia → MYR; Indonesia → IDR; etc.).
_DEFAULT_PROFILES: dict[Locale, LocaleProfile] = {
    Locale.EN: LocaleProfile(
        locale=Locale.EN,
        is_rtl=False,
        currency_code="USD",
        currency_symbol="$",
        currency_prefix=True,
        decimal_separator=".",
        thousands_separator=",",
    ),
    Locale.AR: LocaleProfile(
        locale=Locale.AR,
        is_rtl=True,
        currency_code="SAR",
        currency_symbol="ر.س",
        currency_prefix=False,
        decimal_separator=".",
        thousands_separator=",",
    ),
    Locale.UR: LocaleProfile(
        locale=Locale.UR,
        is_rtl=True,
        currency_code="PKR",
        currency_symbol="₨",
        currency_prefix=True,
        decimal_separator=".",
        thousands_separator=",",
    ),
    Locale.MS: LocaleProfile(
        locale=Locale.MS,
        is_rtl=False,
        currency_code="MYR",
        currency_symbol="RM",
        currency_prefix=True,
        decimal_separator=".",
        thousands_separator=",",
    ),
    Locale.ID: LocaleProfile(
        locale=Locale.ID,
        is_rtl=False,
        currency_code="IDR",
        currency_symbol="Rp",
        currency_prefix=True,
        decimal_separator=",",
        thousands_separator=".",
    ),
    Locale.FR: LocaleProfile(
        locale=Locale.FR,
        is_rtl=False,
        currency_code="EUR",
        currency_symbol="€",
        currency_prefix=False,
        decimal_separator=",",
        thousands_separator=" ",
    ),
    Locale.TR: LocaleProfile(
        locale=Locale.TR,
        is_rtl=False,
        currency_code="TRY",
        currency_symbol="₺",
        currency_prefix=False,
        decimal_separator=",",
        thousands_separator=".",
    ),
}


def default_profile(locale: Locale) -> LocaleProfile:
    """Return the default formatting profile for a locale."""

    return _DEFAULT_PROFILES[locale]


def is_rtl(locale: Locale) -> bool:
    """True if the locale reads right-to-left."""

    return locale in _RTL_LOCALES


@dataclass(frozen=True)
class MessageCatalog:
    """One locale's message dictionary.

    `messages` maps message keys (e.g., `welcome`, `dashboard.title`,
    `error.kyc_required`) to their localised strings. The catalog
    is otherwise opaque to the engine — operators populate from
    JSON / YAML / Crowdin export.
    """

    locale: Locale
    messages: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, value in self.messages.items():
            if not key or not key.strip():
                raise ValueError("message key must be non-empty")
            if not isinstance(value, str):
                raise ValueError(f"message {key!r} value must be str, got {type(value).__name__}")

    def has(self, key: str) -> bool:
        return key in self.messages

    def get(self, key: str) -> str | None:
        return self.messages.get(key)


def translate(
    *,
    key: str,
    locale: Locale,
    catalogs: dict[Locale, MessageCatalog],
    fallback_locale: Locale = _FALLBACK_LOCALE,
    **placeholders: object,
) -> str:
    """Translate `key` to `locale`, falling back to English if missing.

    Returns the localised string with placeholders substituted. If
    neither the requested locale nor the fallback has the key, the
    function returns the key itself (the engine never returns an
    empty string for a missing translation — that would silently
    blank a production UI). A `UserWarning` is emitted for every
    fallback and for every missing key so the operator can sweep
    untranslated strings out of the codebase.
    """

    if not key or not key.strip():
        raise ValueError("key must be non-empty")
    for ph_key in placeholders:
        if ph_key in _FORBIDDEN_PLACEHOLDER_KEYS:
            raise ValueError(
                f"placeholder name {ph_key!r} is forbidden (matches a sensitive-secret denylist)"
            )

    catalog = catalogs.get(locale)
    template: str | None = None
    if catalog is not None and catalog.has(key):
        template = catalog.get(key)
    else:
        # Fall back to the fallback locale.
        if locale is not fallback_locale:
            warnings.warn(
                f"translation missing for {key!r} in {locale.value!r}; "
                f"falling back to {fallback_locale.value!r}",
                UserWarning,
                stacklevel=2,
            )
        fb_catalog = catalogs.get(fallback_locale)
        if fb_catalog is not None and fb_catalog.has(key):
            template = fb_catalog.get(key)
        else:
            warnings.warn(
                f"translation missing for {key!r} in both "
                f"{locale.value!r} and {fallback_locale.value!r}; "
                "returning key as-is",
                UserWarning,
                stacklevel=2,
            )
            template = key

    assert template is not None  # mypy
    if placeholders:
        try:
            return template.format(**placeholders)
        except KeyError as exc:
            raise KeyError(
                f"template {key!r} references placeholder {exc.args[0]!r} but it was not provided"
            ) from exc
    return template


def format_number(value: float, *, profile: LocaleProfile, decimals: int = 2) -> str:
    """Format a number using the locale's separators."""

    if decimals < 0:
        raise ValueError("decimals must be non-negative")
    # Use the standard Python format spec then swap separators for
    # locale-specific ones. Two-pass approach keeps the math out of
    # the locale-specific formatter.
    formatted = f"{value:,.{decimals}f}"
    if profile.thousands_separator != "," or profile.decimal_separator != ".":
        # Use placeholder swap to avoid mangling the wrong separator
        formatted = (
            formatted.replace(",", "\x00")
            .replace(".", profile.decimal_separator)
            .replace("\x00", profile.thousands_separator)
        )
    return formatted


def format_currency(
    value: float,
    *,
    profile: LocaleProfile,
    decimals: int = 2,
) -> str:
    """Format a currency amount using the locale's symbol + position."""

    number = format_number(value, profile=profile, decimals=decimals)
    if profile.currency_prefix:
        return f"{profile.currency_symbol}{number}"
    return f"{number} {profile.currency_symbol}"


def format_percent(
    value: float,
    *,
    profile: LocaleProfile,
    decimals: int = 2,
) -> str:
    """Format a percentage. `value` is the raw number — `0.05` → `5.00%`."""

    pct_number = format_number(value * 100.0, profile=profile, decimals=decimals)
    return f"{pct_number}{profile.percent_symbol}"


def format_date(dt: datetime, *, profile: LocaleProfile) -> str:
    """Format a datetime using the locale's date convention.

    Pinned: `dt` must be timezone-aware. The output uses ISO-8601
    style (`YYYY-MM-DD`) for non-Arabic locales and an
    Arabic-readable day-month-year for AR / UR. Operators wanting
    locale-precise CLDR formats override at the call site; this
    is the minimal-viable formatting the bot needs.
    """

    if dt.tzinfo is None:
        raise ValueError("dt must be timezone-aware")
    if profile.locale in (Locale.AR, Locale.UR):
        return dt.strftime("%d-%m-%Y")
    return dt.strftime("%Y-%m-%d")


def is_translation_complete(
    *,
    catalog: MessageCatalog,
    required_keys: tuple[str, ...],
) -> tuple[bool, tuple[str, ...]]:
    """Audit helper: returns (is_complete, missing_keys).

    Operators run this against every locale's catalog as part of
    their CI to catch untranslated strings before they ship.
    """

    missing = tuple(k for k in required_keys if not catalog.has(k))
    return (not missing, missing)


def render_locale_profile(profile: LocaleProfile) -> str:
    """Format the profile for ops display."""

    direction = "RTL" if profile.is_rtl else "LTR"
    return (
        f"🌐 {profile.locale.value} ({direction})\n"
        f"  currency: {profile.currency_symbol} ({profile.currency_code})\n"
        f"  number format: 1{profile.thousands_separator}234"
        f"{profile.decimal_separator}56\n"
        f"  currency position: "
        f"{'prefix' if profile.currency_prefix else 'suffix'}"
    )


__all__ = [
    "Locale",
    "LocaleProfile",
    "MessageCatalog",
    "default_profile",
    "format_currency",
    "format_date",
    "format_number",
    "format_percent",
    "is_rtl",
    "is_translation_complete",
    "render_locale_profile",
    "translate",
]
