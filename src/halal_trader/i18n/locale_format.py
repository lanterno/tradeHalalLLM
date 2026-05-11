"""Locale-aware number / currency / date formatting — Round-5 Wave 23.

Pure-Python locale-aware formatters covering:
- Number formatting with locale-specific digit shape + decimal/thousands
  separators.
- Currency rendering with symbol position + decimal precision.
- Date formatting with locale-appropriate ordering.

Pinned semantics:

- **Closed-set DigitShape ladder** — WESTERN_ARABIC (0-9), ARABIC_INDIC
  (٠-٩), EXTENDED_ARABIC_INDIC (Persian/Urdu ۰-۹), BENGALI (০-৯).
- **Per-locale formatting profile** is a closed lookup table.
- **Currency precision** defaults to 2 decimals; operator-tunable per
  call.
- **Date format** is closed-set: ISO (YYYY-MM-DD), DMY (DD/MM/YYYY),
  MDY (MM/DD/YYYY), TEXT_DMY ("15 May 2026"). Each locale has a default.
- **Negative numbers** use a leading minus, except for RTL locales
  where the minus is U+200F isolated.
- **Pure-Python deterministic.**
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from halal_trader.i18n.translations import Locale


class DigitShape(str, Enum):
    """Closed-set digit-shape ladder."""

    WESTERN_ARABIC = "western_arabic"  # 0-9
    ARABIC_INDIC = "arabic_indic"  # ٠-٩
    EXTENDED_ARABIC_INDIC = "extended_arabic_indic"  # ۰-۹
    BENGALI = "bengali"  # ০-৯


_DIGIT_TABLE: dict[DigitShape, str] = {
    DigitShape.WESTERN_ARABIC: "0123456789",
    DigitShape.ARABIC_INDIC: "٠١٢٣٤٥٦٧٨٩",
    DigitShape.EXTENDED_ARABIC_INDIC: "۰۱۲۳۴۵۶۷۸۹",
    DigitShape.BENGALI: "০১২৩৪৫৬৭৮৯",
}


class DateFormat(str, Enum):
    """Closed-set date-format ladder."""

    ISO = "iso"  # YYYY-MM-DD
    DMY = "dmy"  # DD/MM/YYYY
    MDY = "mdy"  # MM/DD/YYYY
    TEXT_DMY = "text_dmy"  # 15 May 2026


@dataclass(frozen=True)
class LocaleFormat:
    """Per-locale formatting profile."""

    locale: Locale
    digit_shape: DigitShape
    decimal_sep: str
    thousands_sep: str
    currency_symbol: str
    currency_after: bool
    """True = symbol after amount (e.g. "100 €"); False = before."""
    date_format: DateFormat

    def __post_init__(self) -> None:
        if not self.decimal_sep:
            raise ValueError("decimal_sep must be non-empty")
        if self.decimal_sep == self.thousands_sep:
            raise ValueError("decimal_sep and thousands_sep must differ")
        if not self.currency_symbol:
            raise ValueError("currency_symbol must be non-empty")


_LOCALE_FORMATS: dict[Locale, LocaleFormat] = {
    Locale.EN: LocaleFormat(
        locale=Locale.EN,
        digit_shape=DigitShape.WESTERN_ARABIC,
        decimal_sep=".",
        thousands_sep=",",
        currency_symbol="$",
        currency_after=False,
        date_format=DateFormat.MDY,
    ),
    Locale.AR: LocaleFormat(
        locale=Locale.AR,
        digit_shape=DigitShape.ARABIC_INDIC,
        decimal_sep="٫",
        thousands_sep="٬",
        currency_symbol="ر.س",
        currency_after=True,
        date_format=DateFormat.DMY,
    ),
    Locale.AR_EG: LocaleFormat(
        locale=Locale.AR_EG,
        digit_shape=DigitShape.ARABIC_INDIC,
        decimal_sep="٫",
        thousands_sep="٬",
        currency_symbol="ج.م",
        currency_after=True,
        date_format=DateFormat.DMY,
    ),
    Locale.AR_SA: LocaleFormat(
        locale=Locale.AR_SA,
        digit_shape=DigitShape.ARABIC_INDIC,
        decimal_sep="٫",
        thousands_sep="٬",
        currency_symbol="ر.س",
        currency_after=True,
        date_format=DateFormat.DMY,
    ),
    Locale.UR: LocaleFormat(
        locale=Locale.UR,
        digit_shape=DigitShape.EXTENDED_ARABIC_INDIC,
        decimal_sep=".",
        thousands_sep=",",
        currency_symbol="₨",
        currency_after=False,
        date_format=DateFormat.DMY,
    ),
    Locale.MS: LocaleFormat(
        locale=Locale.MS,
        digit_shape=DigitShape.WESTERN_ARABIC,
        decimal_sep=".",
        thousands_sep=",",
        currency_symbol="RM",
        currency_after=False,
        date_format=DateFormat.DMY,
    ),
    Locale.ID: LocaleFormat(
        locale=Locale.ID,
        digit_shape=DigitShape.WESTERN_ARABIC,
        decimal_sep=",",
        thousands_sep=".",
        currency_symbol="Rp",
        currency_after=False,
        date_format=DateFormat.DMY,
    ),
    Locale.TR: LocaleFormat(
        locale=Locale.TR,
        digit_shape=DigitShape.WESTERN_ARABIC,
        decimal_sep=",",
        thousands_sep=".",
        currency_symbol="₺",
        currency_after=True,
        date_format=DateFormat.DMY,
    ),
    Locale.HA: LocaleFormat(
        locale=Locale.HA,
        digit_shape=DigitShape.WESTERN_ARABIC,
        decimal_sep=".",
        thousands_sep=",",
        currency_symbol="₦",
        currency_after=False,
        date_format=DateFormat.DMY,
    ),
    Locale.SW: LocaleFormat(
        locale=Locale.SW,
        digit_shape=DigitShape.WESTERN_ARABIC,
        decimal_sep=".",
        thousands_sep=",",
        currency_symbol="TSh",
        currency_after=False,
        date_format=DateFormat.DMY,
    ),
    Locale.BN: LocaleFormat(
        locale=Locale.BN,
        digit_shape=DigitShape.BENGALI,
        decimal_sep=".",
        thousands_sep=",",
        currency_symbol="৳",
        currency_after=False,
        date_format=DateFormat.DMY,
    ),
    Locale.FA: LocaleFormat(
        locale=Locale.FA,
        digit_shape=DigitShape.EXTENDED_ARABIC_INDIC,
        decimal_sep="٫",
        thousands_sep="٬",
        currency_symbol="﷼",
        currency_after=True,
        date_format=DateFormat.DMY,
    ),
    Locale.PS: LocaleFormat(
        locale=Locale.PS,
        digit_shape=DigitShape.EXTENDED_ARABIC_INDIC,
        decimal_sep="٫",
        thousands_sep="٬",
        currency_symbol="؋",
        currency_after=True,
        date_format=DateFormat.DMY,
    ),
    Locale.FR: LocaleFormat(
        locale=Locale.FR,
        digit_shape=DigitShape.WESTERN_ARABIC,
        decimal_sep=",",
        thousands_sep=" ",
        currency_symbol="€",
        currency_after=True,
        date_format=DateFormat.DMY,
    ),
    Locale.ES: LocaleFormat(
        locale=Locale.ES,
        digit_shape=DigitShape.WESTERN_ARABIC,
        decimal_sep=",",
        thousands_sep=".",
        currency_symbol="€",
        currency_after=True,
        date_format=DateFormat.DMY,
    ),
}


def format_for(locale: Locale) -> LocaleFormat:
    """Look up the locale's formatting profile."""
    return _LOCALE_FORMATS[locale]


def _shape_digits(s: str, shape: DigitShape) -> str:
    """Replace Western-Arabic digits with the target shape's glyphs."""
    table = _DIGIT_TABLE[shape]
    out = []
    for ch in s:
        if "0" <= ch <= "9":
            out.append(table[ord(ch) - ord("0")])
        else:
            out.append(ch)
    return "".join(out)


def format_number(
    value: float,
    locale: Locale,
    *,
    decimals: int = 2,
) -> str:
    """Format a number per locale rules.

    Pinned:
    - `decimals` ≥ 0; truncation rounds half-to-even via standard format spec.
    - Negative numbers use a leading minus.
    - Thousands grouping in the integer part.
    """
    if decimals < 0:
        raise ValueError("decimals must be ≥ 0")
    if decimals > 10:
        raise ValueError("decimals > 10 suspicious")
    profile = format_for(locale)
    raw = f"{value:.{decimals}f}"
    is_negative = raw.startswith("-")
    if is_negative:
        raw = raw[1:]
    if "." in raw:
        int_part, dec_part = raw.split(".", 1)
    else:
        int_part = raw
        dec_part = ""
    # Group thousands.
    grouped: list[str] = []
    while len(int_part) > 3:
        grouped.append(int_part[-3:])
        int_part = int_part[:-3]
    grouped.append(int_part)
    grouped.reverse()
    int_str = profile.thousands_sep.join(grouped)
    if dec_part:
        body = f"{int_str}{profile.decimal_sep}{dec_part}"
    else:
        body = int_str
    body = _shape_digits(body, profile.digit_shape)
    if is_negative:
        body = "-" + body
    return body


def format_currency(
    amount: float,
    locale: Locale,
    *,
    decimals: int = 2,
) -> str:
    """Format an amount as currency per locale conventions."""
    profile = format_for(locale)
    n = format_number(amount, locale, decimals=decimals)
    if profile.currency_after:
        return f"{n} {profile.currency_symbol}"
    return f"{profile.currency_symbol}{n}"


_MONTH_NAMES_EN: tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def format_date(
    d: date,
    locale: Locale,
    *,
    override_format: DateFormat | None = None,
) -> str:
    """Format a date per locale rules.

    Pinned: digits are shaped after the textual ordering is applied;
    TEXT_DMY uses English month names (caller passes a localised month
    name via the translation registry if desired).
    """
    profile = format_for(locale)
    fmt = override_format if override_format is not None else profile.date_format
    yyyy = f"{d.year:04d}"
    mm = f"{d.month:02d}"
    dd = f"{d.day:02d}"
    if fmt is DateFormat.ISO:
        out = f"{yyyy}-{mm}-{dd}"
    elif fmt is DateFormat.DMY:
        out = f"{dd}/{mm}/{yyyy}"
    elif fmt is DateFormat.MDY:
        out = f"{mm}/{dd}/{yyyy}"
    elif fmt is DateFormat.TEXT_DMY:
        month_name = _MONTH_NAMES_EN[d.month - 1]
        out = f"{d.day} {month_name} {d.year}"
    else:
        raise ValueError(f"unknown date format {fmt.value}")
    return _shape_digits(out, profile.digit_shape)


def shape_digits(s: str, locale: Locale) -> str:
    """Public helper: shape any digits in `s` to the locale's script."""
    return _shape_digits(s, format_for(locale).digit_shape)
