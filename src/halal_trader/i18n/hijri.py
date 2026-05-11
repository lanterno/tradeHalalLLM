"""Hijri ↔ Gregorian calendar converter — Round-5 Wave 23.

Tabular Islamic civil calendar (also known as the Umm-al-Qura
approximation for non-observational use). Civil sukuk + Zakat
schedules use this calendar widely; this module ships:

- Closed-set HijriMonth enum.
- `gregorian_to_hijri` / `hijri_to_gregorian` converters.
- `HijriDate` dataclass with arithmetic helpers.
- Sukuk-maturity sanity helper (`hijri_difference_days`).

This module uses the **tabular** calendar (deterministic) rather than
observed-moon (which requires astronomical lookups). AAOIFI Standard 35
permits the tabular calendar for civil/contractual use; observational
adjustments are an operator-supplied delta. The standard tabular
calendar has 11 leap years per 30-year cycle in years 2,5,7,10,13,
16,18,21,24,26,29 (the "Kuwaiti" pattern).

Pinned semantics:

- **Closed-set HijriMonth ladder** — Muharram → Dhu al-Hijjah.
- **Tabular calendar month lengths**: odd-numbered months 30 days,
  even-numbered months 29 days, with Dhu al-Hijjah extended to 30 in
  leap years.
- **Civil epoch**: 1 Muharram 1 AH = 16 July 622 CE (Julian) =
  19 July 622 CE (Gregorian, proleptic).
- **Conversion formula** uses the standard tabular Islamic calendar
  arithmetic (see Reingold-Dershowitz).
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class HijriMonth(int, Enum):
    """Closed-set Hijri-month ladder (1-12)."""

    MUHARRAM = 1
    SAFAR = 2
    RABI_AL_AWWAL = 3
    RABI_AL_THANI = 4
    JUMADA_AL_AWWAL = 5
    JUMADA_AL_THANI = 6
    RAJAB = 7
    SHABAN = 8
    RAMADAN = 9
    SHAWWAL = 10
    DHU_AL_QADAH = 11
    DHU_AL_HIJJAH = 12


_MONTH_NAMES_EN: dict[HijriMonth, str] = {
    HijriMonth.MUHARRAM: "Muharram",
    HijriMonth.SAFAR: "Safar",
    HijriMonth.RABI_AL_AWWAL: "Rabi al-Awwal",
    HijriMonth.RABI_AL_THANI: "Rabi al-Thani",
    HijriMonth.JUMADA_AL_AWWAL: "Jumada al-Awwal",
    HijriMonth.JUMADA_AL_THANI: "Jumada al-Thani",
    HijriMonth.RAJAB: "Rajab",
    HijriMonth.SHABAN: "Sha'ban",
    HijriMonth.RAMADAN: "Ramadan",
    HijriMonth.SHAWWAL: "Shawwal",
    HijriMonth.DHU_AL_QADAH: "Dhu al-Qa'dah",
    HijriMonth.DHU_AL_HIJJAH: "Dhu al-Hijjah",
}


# Leap years in the 30-year cycle (1-indexed within cycle).
_LEAP_YEAR_OFFSETS: frozenset[int] = frozenset({2, 5, 7, 10, 13, 16, 18, 21, 24, 26, 29})


def is_hijri_leap_year(year: int) -> bool:
    """True iff `year` is a leap year in the tabular Islamic calendar."""
    if year <= 0:
        raise ValueError("Hijri year must be positive")
    cycle_position = ((year - 1) % 30) + 1
    return cycle_position in _LEAP_YEAR_OFFSETS


def hijri_month_length(year: int, month: HijriMonth) -> int:
    """Length of `month` in `year`."""
    if month is HijriMonth.DHU_AL_HIJJAH:
        return 30 if is_hijri_leap_year(year) else 29
    # Odd months: 30; even months: 29 (1-indexed).
    return 30 if month.value % 2 == 1 else 29


def hijri_year_length(year: int) -> int:
    """Length of `year` in days (354 or 355)."""
    return 355 if is_hijri_leap_year(year) else 354


@dataclass(frozen=True)
class HijriDate:
    """One Hijri date."""

    year: int
    month: HijriMonth
    day: int

    def __post_init__(self) -> None:
        if self.year <= 0:
            raise ValueError("year must be positive")
        if self.year > 10_000:
            raise ValueError("year > 10000 suspicious")
        if self.day <= 0:
            raise ValueError("day must be positive")
        max_day = hijri_month_length(self.year, self.month)
        if self.day > max_day:
            raise ValueError(f"day {self.day} > {max_day} for {self.month.name} {self.year}")

    def __lt__(self, other: "HijriDate") -> bool:
        return (self.year, self.month.value, self.day) < (other.year, other.month.value, other.day)


# Civil-epoch Julian Day Number (JDN) of 1 Muharram 1 AH.
# Using the proleptic Gregorian: 16 July 622 CE Julian → JDN 1948440.
_HIJRI_EPOCH_JDN: int = 1_948_440


def _hijri_to_jdn(d: HijriDate) -> int:
    """Convert a HijriDate to Julian Day Number via the tabular formula."""
    y, m, day = d.year, d.month.value, d.day
    # Days from start of year to start of month (1-indexed).
    days_before_month = 0
    for i in range(1, m):
        days_before_month += hijri_month_length(y, HijriMonth(i))
    # Days from epoch to start of year.
    full_cycles = (y - 1) // 30
    days_in_full_cycles = full_cycles * (354 * 19 + 355 * 11)
    remaining_years = (y - 1) % 30
    days_in_remaining_years = 0
    for yr_offset in range(remaining_years):
        yr = full_cycles * 30 + yr_offset + 1
        days_in_remaining_years += hijri_year_length(yr)
    return (
        _HIJRI_EPOCH_JDN
        + days_in_full_cycles
        + days_in_remaining_years
        + days_before_month
        + day
        - 1
    )


def _jdn_to_hijri(jdn: int) -> HijriDate:
    """Inverse of `_hijri_to_jdn`."""
    if jdn < _HIJRI_EPOCH_JDN:
        raise ValueError("jdn before Hijri epoch")
    days = jdn - _HIJRI_EPOCH_JDN
    # Compute year.
    cycle_length = 354 * 19 + 355 * 11  # = 10631
    full_cycles, rem = divmod(days, cycle_length)
    base_year = full_cycles * 30 + 1
    year = base_year
    while True:
        yr_len = hijri_year_length(year)
        if rem < yr_len:
            break
        rem -= yr_len
        year += 1
    # Compute month.
    month = 1
    while True:
        m_len = hijri_month_length(year, HijriMonth(month))
        if rem < m_len:
            break
        rem -= m_len
        month += 1
    day = rem + 1
    return HijriDate(year=year, month=HijriMonth(month), day=day)


def _gregorian_to_jdn(d: date) -> int:
    """Proleptic Gregorian → JDN. Standard algorithm."""
    y, m, day = d.year, d.month, d.day
    a = (14 - m) // 12
    yy = y + 4800 - a
    mm = m + 12 * a - 3
    return day + (153 * mm + 2) // 5 + 365 * yy + yy // 4 - yy // 100 + yy // 400 - 32045


def _jdn_to_gregorian(jdn: int) -> date:
    """Inverse of `_gregorian_to_jdn`."""
    a = jdn + 32044
    b = (4 * a + 3) // 146097
    c = a - (146097 * b) // 4
    d_ = (4 * c + 3) // 1461
    e = c - (1461 * d_) // 4
    m_ = (5 * e + 2) // 153
    day = e - (153 * m_ + 2) // 5 + 1
    month = m_ + 3 - 12 * (m_ // 10)
    year = 100 * b + d_ - 4800 + (m_ // 10)
    return date(year, month, day)


def gregorian_to_hijri(d: date) -> HijriDate:
    """Convert a proleptic Gregorian date to a tabular Hijri date."""
    return _jdn_to_hijri(_gregorian_to_jdn(d))


def hijri_to_gregorian(d: HijriDate) -> date:
    """Convert a tabular Hijri date to a proleptic Gregorian date."""
    return _jdn_to_gregorian(_hijri_to_jdn(d))


def hijri_difference_days(a: HijriDate, b: HijriDate) -> int:
    """Days from `a` to `b` (positive if b > a)."""
    return _hijri_to_jdn(b) - _hijri_to_jdn(a)


def add_days(d: HijriDate, days: int) -> HijriDate:
    """Add `days` (can be negative) to a Hijri date."""
    return _jdn_to_hijri(_hijri_to_jdn(d) + days)


def month_name(month: HijriMonth) -> str:
    return _MONTH_NAMES_EN[month]


def render_hijri(d: HijriDate) -> str:
    """Operator-readable Hijri date."""
    return f"{d.day} {month_name(d.month)} {d.year} AH"


def render_dual(g: date, h: HijriDate | None = None) -> str:
    """Render a Gregorian + Hijri dual-date string.

    If `h` is None, computes the Hijri date from the Gregorian one.
    """
    hijri = h if h is not None else gregorian_to_hijri(g)
    return f"{g.isoformat()} / {render_hijri(hijri)}"
