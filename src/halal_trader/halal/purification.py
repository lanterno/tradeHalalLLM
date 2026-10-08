"""Dividend purification calculator behind the admin form.

Many Shariah-screened stocks still pay incidental haram revenue
(typically a small interest-bearing investment portfolio). The standard
practice is **purification**: estimate the haram portion of the
dividend you receive and donate that fraction to charity. This module
provides the data model + math for one such obligation.

Inputs:

* ``dividend_amount_usd`` — the gross dividend received.
* ``haram_revenue_pct`` — the screening provider's published estimate
  (screening providers publish this; default 0 if unknown so a missing
  value never *under*-tags the obligation).

Output: a :class:`PurificationEntry` capturing the obligation, for the
admin calculator. The persisted dividend ledger is
``compliance/purification.py`` (``purification_accruals``): it reads the
haram share from the in-house screen's impure-income ratio at each
ex-date and the holdings from the broker ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from halal_trader.domain.money import quantize_usd, to_decimal


@dataclass(frozen=True)
class PurificationEntry:
    """One dividend → one purification obligation."""

    symbol: str
    dividend_usd: Decimal
    haram_pct: Decimal
    purification_usd: Decimal
    received_at: datetime
    notes: str = ""


def compute_purification(
    *,
    symbol: str,
    dividend_usd: float | Decimal,
    haram_revenue_pct: float | Decimal = 0.0,
    received_at: datetime | None = None,
    notes: str = "",
) -> PurificationEntry:
    """Build a :class:`PurificationEntry` for one received dividend.

    Negative dividends (e.g. a dividend reversal on a corporate action
    correction) are clamped to zero — purification is a one-way
    obligation, never a credit to the operator.
    """
    div = to_decimal(dividend_usd)
    if div < 0:
        div = Decimal("0")
    pct = to_decimal(haram_revenue_pct)
    pct = max(Decimal("0"), min(pct, Decimal("1")))
    purification = quantize_usd(div * pct)
    return PurificationEntry(
        symbol=symbol.upper(),
        dividend_usd=quantize_usd(div),
        haram_pct=pct,
        purification_usd=purification,
        received_at=received_at or datetime.now(UTC),
        notes=notes,
    )
