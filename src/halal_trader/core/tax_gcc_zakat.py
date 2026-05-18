"""GCC Zakat-only report builder — Round-5 Wave 18.D.

Saudi Arabia + UAE have no income tax for individuals; the year-end
report for these jurisdictions is **Zakat-only**, framed in the local
currency. This module composes the report from the existing zakat
calculator (Wave 1.J) + multi-currency translator (Wave 18.H), with
GCC-specific formatting + jurisdiction-specific Zakat-rate selection.

Pinned semantics:

- **Closed-set GccCountry ladder** (SAUDI / UAE / BAHRAIN / KUWAIT /
  QATAR / OMAN).
- **Local-currency rendering** — defaults to country's official code.
- **Zakat rate is jurisdiction-default** — Saudi: 2.5% Sunni
  consensus; Bahrain: similar; Kuwait/Oman: 2.5%; UAE: 2.5%.
- **No-secret-leak pin** on render output (no per-bank balances).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from halal_trader.halal.zakat import (
    ZakatCalculation,
    ZakatPolicy,
)


class GccCountry(str, Enum):
    """Closed-set GCC countries."""

    SAUDI = "saudi"
    UAE = "uae"
    BAHRAIN = "bahrain"
    KUWAIT = "kuwait"
    QATAR = "qatar"
    OMAN = "oman"


# Country → (default reporting currency, default zakat rate)
_COUNTRY_DEFAULTS: dict[GccCountry, tuple[str, float]] = {
    GccCountry.SAUDI: ("SAR", 0.025),
    GccCountry.UAE: ("AED", 0.025),
    GccCountry.BAHRAIN: ("BHD", 0.025),
    GccCountry.KUWAIT: ("KWD", 0.025),
    GccCountry.QATAR: ("QAR", 0.025),
    GccCountry.OMAN: ("OMR", 0.025),
}


@dataclass(frozen=True)
class GccZakatReport:
    """The composed report."""

    country: GccCountry
    operator_handle: str
    reporting_currency: str
    reporting_period_start: date
    reporting_period_end: date
    zakat_calculation: ZakatCalculation
    additional_charity_paid: float

    def __post_init__(self) -> None:
        if not self.operator_handle or not self.operator_handle.strip():
            raise ValueError("operator_handle must be non-empty")
        if "@" in self.operator_handle:
            raise ValueError("operator_handle must be a handle, not an email")
        if self.reporting_period_end < self.reporting_period_start:
            raise ValueError("reporting_period_end before period_start")
        if self.additional_charity_paid < 0:
            raise ValueError("additional_charity_paid must be non-negative")
        if not self.reporting_currency or len(self.reporting_currency) > 8:
            raise ValueError("reporting_currency must be a non-empty short code")


def default_currency_for(country: GccCountry) -> str:
    return _COUNTRY_DEFAULTS[country][0]


def default_zakat_rate_for(country: GccCountry) -> float:
    return _COUNTRY_DEFAULTS[country][1]


def default_policy_for(country: GccCountry) -> ZakatPolicy:
    return ZakatPolicy(zakat_rate=default_zakat_rate_for(country))


def build_report(
    *,
    country: GccCountry,
    operator_handle: str,
    reporting_period_start: date,
    reporting_period_end: date,
    zakat_calculation: ZakatCalculation,
    additional_charity_paid: float = 0.0,
    reporting_currency: str | None = None,
) -> GccZakatReport:
    """Compose a GCC Zakat-only report."""
    currency = reporting_currency or default_currency_for(country)
    return GccZakatReport(
        country=country,
        operator_handle=operator_handle,
        reporting_currency=currency,
        reporting_period_start=reporting_period_start,
        reporting_period_end=reporting_period_end,
        zakat_calculation=zakat_calculation,
        additional_charity_paid=additional_charity_paid,
    )


_FORBIDDEN_RENDER_TOKENS: tuple[str, ...] = (
    "@",
    "zoom.us",
    "meet.google",
    "private_email",
    "+1-",
    "Authorization",
    "IBAN",
    "Bank-",
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_report(report: GccZakatReport) -> str:
    z = report.zakat_calculation
    head = (
        f"GCC Zakat report — {report.country.value.upper()} — "
        f"{report.operator_handle} ({report.reporting_period_start.isoformat()}→"
        f"{report.reporting_period_end.isoformat()})"
    )
    lines = [
        head,
        f"  net_assets: {z.net_assets:.2f} {report.reporting_currency}",
        f"  nisab: {z.nisab_value:.2f} {report.reporting_currency} ({z.basis_used.value})",
        f"  meets_nisab: {z.meets_nisab}",
        f"  zakat_owed: {z.zakat_owed:.2f} {report.reporting_currency}",
        f"  additional_charity_paid: {report.additional_charity_paid:.2f} "
        f"{report.reporting_currency}",
    ]
    if z.hawl_due_date is not None:
        lines.append(f"  hawl_due_date: {z.hawl_due_date.isoformat()}")
    return _scrub("\n".join(lines))
