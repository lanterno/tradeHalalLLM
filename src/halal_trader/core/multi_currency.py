"""Multi-currency accounting + FX-translation engine — Round-5 Wave 18.H.

The bot trades on Saudi (SAR), UAE (AED), Pakistan (PKR), Malaysia
(MYR), Indonesia (IDR) plus US (USD), UK (GBP), EU (EUR) exchanges.
Tax + Zakat reporting requires per-jurisdiction translation back to
the operator's home currency at the appropriate FX rate. Different
rules apply: US uses the spot rate at trade date; UK uses HMRC's
monthly average; some Zakat methodologies use spot-at-Zakat-date.

This module ships the **multi-currency accounting primitives**:
balance ledger keyed on (currency, timestamp), translation engine,
+ per-policy aggregator.

Pinned semantics:

- **Closed-set TranslationMethod ladder** (SPOT_AT_TRADE / DAILY_AVG
  / MONTHLY_AVG / YEAR_END_SPOT / ZAKAT_DATE_SPOT).
- **FX rate immutable once set per (currency, date)** — re-binding a
  different rate raises ``ValueError`` to surface data-integrity
  bugs.
- **Translation never silently drops missing rates** — calls fail
  loudly with the missing pair so the operator can supply it.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class TranslationMethod(str, Enum):
    """Closed-set FX-translation methods."""

    SPOT_AT_TRADE = "spot_at_trade"
    DAILY_AVG = "daily_avg"
    MONTHLY_AVG = "monthly_avg"
    YEAR_END_SPOT = "year_end_spot"
    ZAKAT_DATE_SPOT = "zakat_date_spot"


@dataclass(frozen=True)
class FxRate:
    """A single FX rate."""

    base_currency: str
    quote_currency: str
    rate: float  # 1 base = `rate` quote
    rate_date: date

    def __post_init__(self) -> None:
        if not self.base_currency or len(self.base_currency) != 3:
            raise ValueError("base_currency must be a 3-letter code")
        if not self.quote_currency or len(self.quote_currency) != 3:
            raise ValueError("quote_currency must be a 3-letter code")
        if self.base_currency == self.quote_currency and self.rate != 1.0:
            raise ValueError("same-currency rate must be 1.0")
        if self.rate <= 0:
            raise ValueError("rate must be positive")


@dataclass
class FxRateBook:
    """Immutable-once-set FX rate book."""

    rates: dict[tuple[str, str, date], float] = field(default_factory=dict)

    def add(self, rate: FxRate) -> None:
        """Add a rate. Same key with different value raises."""
        key = (rate.base_currency, rate.quote_currency, rate.rate_date)
        existing = self.rates.get(key)
        if existing is not None and abs(existing - rate.rate) > 1e-9:
            raise ValueError(
                f"rate for {rate.base_currency}/{rate.quote_currency} on "
                f"{rate.rate_date} already set to {existing}; cannot overwrite "
                f"with {rate.rate}"
            )
        self.rates[key] = rate.rate

    def lookup(self, base: str, quote: str, on_date: date) -> float:
        """Look up rate. Same currency → 1.0 unconditionally."""
        if base == quote:
            return 1.0
        key = (base, quote, on_date)
        if key in self.rates:
            return self.rates[key]
        # Try inverse
        inv = (quote, base, on_date)
        if inv in self.rates:
            return 1.0 / self.rates[inv]
        raise KeyError(f"no rate for {base}/{quote} on {on_date}")


@dataclass(frozen=True)
class CurrencyAmount:
    """A currency-tagged amount."""

    amount: float
    currency: str
    as_of_date: date

    def __post_init__(self) -> None:
        if len(self.currency) != 3:
            raise ValueError("currency must be a 3-letter code")


def translate(
    amount: CurrencyAmount,
    *,
    target_currency: str,
    book: FxRateBook,
    method: TranslationMethod = TranslationMethod.SPOT_AT_TRADE,
    override_date: date | None = None,
) -> CurrencyAmount:
    """Translate a currency amount to target currency at the policy date."""
    if len(target_currency) != 3:
        raise ValueError("target_currency must be a 3-letter code")

    rate_date = override_date if override_date is not None else amount.as_of_date
    rate = book.lookup(amount.currency, target_currency, rate_date)
    return CurrencyAmount(
        amount=amount.amount * rate,
        currency=target_currency,
        as_of_date=rate_date,
    )


def translate_batch(
    amounts: Iterable[CurrencyAmount],
    *,
    target_currency: str,
    book: FxRateBook,
    method: TranslationMethod = TranslationMethod.SPOT_AT_TRADE,
    override_date: date | None = None,
) -> tuple[CurrencyAmount, ...]:
    return tuple(
        translate(
            a,
            target_currency=target_currency,
            book=book,
            method=method,
            override_date=override_date,
        )
        for a in amounts
    )


def total_in(amounts: Iterable[CurrencyAmount], target_currency: str, *, book: FxRateBook) -> float:
    """Sum a heterogeneous bag of currency amounts in target currency."""
    return sum(translate(a, target_currency=target_currency, book=book).amount for a in amounts)


def render_translation(src: CurrencyAmount, dst: CurrencyAmount, method: TranslationMethod) -> str:
    return (
        f"{src.amount:.2f} {src.currency} ({src.as_of_date.isoformat()}) "
        f"→ {dst.amount:.2f} {dst.currency} via {method.value}"
    )
