"""Dividend purification ledger.

Many Shariah-screened stocks still pay incidental haram revenue
(typically a small interest-bearing investment portfolio). The standard
practice is **purification**: estimate the haram portion of the
dividend you receive and donate that fraction to charity. This module
provides the data model + math for tracking those obligations.

Inputs:

* ``dividend_amount_usd`` — the gross dividend received.
* ``haram_revenue_pct`` — the screening provider's published estimate
  (Zoya / IdealRatings publish this; default 0 if unknown so a missing
  value never *under*-tags the obligation).

Output: a :class:`PurificationEntry` capturing the obligation, ready to
be persisted (DB table TBD in 3.6b) and surfaced to the operator for
manual donation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    paid_at: datetime | None = None  # set when the operator records the donation

    @property
    def is_outstanding(self) -> bool:
        return self.paid_at is None


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


@dataclass
class PurificationLedger:
    """In-memory append-only ledger of purification obligations.

    Persistence (DB table + Alembic migration) lands in 3.6b. Today this
    is operator-side bookkeeping that surfaces in CLI exports — the
    crucial property is that *no obligation can be silently discarded*,
    only marked paid.
    """

    entries: list[PurificationEntry] = field(default_factory=list)

    def record(self, entry: PurificationEntry) -> None:
        self.entries.append(entry)

    def outstanding_total(self) -> Decimal:
        total = Decimal("0")
        for e in self.entries:
            if e.is_outstanding:
                total += e.purification_usd
        return quantize_usd(total)

    def paid_total(self) -> Decimal:
        total = Decimal("0")
        for e in self.entries:
            if not e.is_outstanding:
                total += e.purification_usd
        return quantize_usd(total)

    def mark_paid(self, index: int, paid_at: datetime | None = None) -> None:
        if not 0 <= index < len(self.entries):
            raise IndexError(f"no purification entry at index {index}")
        old = self.entries[index]
        self.entries[index] = PurificationEntry(
            symbol=old.symbol,
            dividend_usd=old.dividend_usd,
            haram_pct=old.haram_pct,
            purification_usd=old.purification_usd,
            received_at=old.received_at,
            notes=old.notes,
            paid_at=paid_at or datetime.now(UTC),
        )
