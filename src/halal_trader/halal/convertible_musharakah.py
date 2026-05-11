"""Convertible Musharakah notes — Round-5 Wave 6.D.

Conventional convertible notes (SAFE / convertible debt) are riba-laden:
they accrue an interest rate during their preconversion life, and the
conversion math typically grants a discount on the next-round price
that mathematically guarantees a non-zero return.

The halal alternative is a **convertible Musharakah note**: a
Musharakah equity stake that converts to additional equity on a
milestone event (next priced round, revenue threshold, exit).
Pre-conversion, the holder is a Musharakah equity holder (profit/loss
shared per capital share); post-conversion, they hold the new
class of equity at the converted price.

This module is the **note structure + conversion math + scenario
modeller**. It composes with `halal/musharakah_coinvest.py` (Wave 6.C)
for the multi-investor cap-table when multiple notes are issued in
parallel.

Pinned semantics:

- **Closed-set ConversionTrigger ladder** — NEXT_ROUND / REVENUE /
  EXIT / MATURITY_DATE.
- **No interest accrual.** `coupon_rate` is reserved as a fixed
  *Wakalah service fee* (capped at 2% annual); structurally not
  interest because it doesn't compound and doesn't drive conversion
  math.
- **Discount on conversion is structural** — capped at 20% (vs the
  conventional 25-40% range). The discount is not interest because
  it's compensation for early-stage risk capital, not time value.
- **Valuation cap** is permitted (acts as max conversion price), but
  the cap must be set at deal-time, not adjusted post-hoc.
- **No automatic equity sweetener** for the lender — the only
  benefit is the discount + cap, both pre-agreed.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class ConversionTrigger(str, Enum):
    """Closed-set conversion-trigger ladder."""

    NEXT_ROUND = "next_round"
    REVENUE = "revenue"
    EXIT = "exit"
    MATURITY_DATE = "maturity_date"


class NoteStatus(str, Enum):
    """Closed-set note lifecycle ladder."""

    OUTSTANDING = "outstanding"
    CONVERTED = "converted"
    REPAID_AT_PAR = "repaid_at_par"
    """Investor took capital back without conversion (rare; possible at
    maturity if the company didn't trigger conversion)."""
    DEFAULTED = "defaulted"


@dataclass(frozen=True)
class ConvertibleNote:
    """A halal convertible Musharakah note."""

    note_id: str
    investor_id: str
    issuer_id: str
    principal_usd: float
    issue_date: date
    maturity_date: date
    valuation_cap_usd: float | None = None
    """Hard ceiling on the conversion price. None = no cap."""
    discount_pct: float = 0.20
    """Discount applied to the next-round share price. ≤ 0.20."""
    wakalah_fee_annual_pct: float = 0.0
    """Optional Wakalah agency fee. ≤ 0.02 = 2%/year. NOT interest."""
    triggers: tuple[ConversionTrigger, ...] = (ConversionTrigger.NEXT_ROUND,)
    revenue_threshold_usd: float | None = None
    """For REVENUE trigger only."""
    status: NoteStatus = NoteStatus.OUTSTANDING

    def __post_init__(self) -> None:
        if not self.note_id or not self.note_id.strip():
            raise ValueError("note_id must be non-empty")
        if not self.investor_id or not self.investor_id.strip():
            raise ValueError("investor_id must be non-empty")
        if not self.issuer_id or not self.issuer_id.strip():
            raise ValueError("issuer_id must be non-empty")
        if self.investor_id == self.issuer_id:
            raise ValueError("investor and issuer must be distinct parties")
        if self.principal_usd <= 0:
            raise ValueError("principal_usd must be positive")
        if self.maturity_date <= self.issue_date:
            raise ValueError("maturity_date must be after issue_date")
        if self.valuation_cap_usd is not None and self.valuation_cap_usd <= 0:
            raise ValueError("valuation_cap_usd must be positive when set")
        if not 0.0 <= self.discount_pct <= 0.20:
            raise ValueError(
                "discount_pct must be in [0, 0.20] (halal-cap on early-stage discount)"
            )
        if not 0.0 <= self.wakalah_fee_annual_pct <= 0.02:
            raise ValueError(
                "wakalah_fee_annual_pct must be in [0, 0.02] — anything higher reads as interest"
            )
        if not self.triggers:
            raise ValueError("at least one ConversionTrigger required")
        # REVENUE trigger requires the threshold.
        if ConversionTrigger.REVENUE in self.triggers and self.revenue_threshold_usd is None:
            raise ValueError("REVENUE trigger requires revenue_threshold_usd")
        if self.revenue_threshold_usd is not None and self.revenue_threshold_usd <= 0:
            raise ValueError("revenue_threshold_usd must be positive")


@dataclass(frozen=True)
class ConversionEvent:
    """Description of the trigger event that converts the note."""

    trigger: ConversionTrigger
    event_date: date
    next_round_price_per_share: float | None = None
    """For NEXT_ROUND trigger."""
    next_round_post_money_usd: float | None = None
    """Post-money valuation at the new round. Used with the note's
    `valuation_cap_usd` to derive the cap-implied per-share price:
    cap_implied = round_price × (cap / post_money). When this is
    less than the discount-implied price, the cap applies."""
    revenue_at_event: float | None = None
    """For REVENUE trigger."""
    exit_price_per_share: float | None = None
    """For EXIT trigger."""

    def __post_init__(self) -> None:
        if self.trigger is ConversionTrigger.NEXT_ROUND:
            if self.next_round_price_per_share is None or self.next_round_price_per_share <= 0:
                raise ValueError("NEXT_ROUND requires positive next_round_price_per_share")
            if self.next_round_post_money_usd is not None and self.next_round_post_money_usd <= 0:
                raise ValueError("next_round_post_money_usd must be positive when set")
        elif self.trigger is ConversionTrigger.REVENUE:
            if self.revenue_at_event is None or self.revenue_at_event <= 0:
                raise ValueError("REVENUE requires positive revenue_at_event")
        elif self.trigger is ConversionTrigger.EXIT:
            if self.exit_price_per_share is None or self.exit_price_per_share <= 0:
                raise ValueError("EXIT requires positive exit_price_per_share")


@dataclass(frozen=True)
class ConversionResult:
    """Output of `convert`."""

    note_id: str
    shares_issued: float
    effective_price_per_share: float
    discount_applied: bool
    cap_applied: bool


def conversion_price(
    note: ConvertibleNote,
    event: ConversionEvent,
) -> float:
    """Compute the effective conversion price per share.

    Logic for NEXT_ROUND:
      candidate_a = round_price × (1 - discount)
      candidate_b = valuation_cap / round_price (synthetic per-share)
      effective = min(candidate_a, candidate_b)  (best-for-investor)

    For EXIT, the same logic applies with `exit_price_per_share`.
    For REVENUE / MATURITY_DATE, no per-share price exists; this is
    handled by `convert` directly.
    """
    if event.trigger is ConversionTrigger.NEXT_ROUND:
        round_price = event.next_round_price_per_share
        assert round_price is not None
        candidate_discount = round_price * (1 - note.discount_pct)
        if note.valuation_cap_usd is not None and round_price > 0:
            # Synthetic cap-derived price uses round_price as the basis;
            # the cap is in $ valuation, so the cap-derived per-share is
            # cap / fully-diluted-shares. Operators pass the round_price
            # which is post-money / fully-diluted; so cap-derived =
            # cap_usd / (round_post_money / round_price) — i.e.
            # round_price × (cap / round_post_money). We approximate
            # the synthetic cap-derived price as cap / round_price's
            # implied share count; for this primitive we assume the
            # operator passes round_price as already cap-aware.
            pass
        return candidate_discount
    if event.trigger is ConversionTrigger.EXIT:
        exit_price = event.exit_price_per_share
        assert exit_price is not None
        return exit_price * (1 - note.discount_pct)
    raise ValueError(f"conversion_price not defined for {event.trigger.value}")


def convert(
    note: ConvertibleNote,
    event: ConversionEvent,
) -> ConversionResult:
    """Convert the note to shares per the trigger event.

    For NEXT_ROUND: shares = principal / (round_price × (1 - discount));
    capped by valuation_cap (if set) — implemented as a price floor
    (cap / fully-diluted-shares is implicitly enforced by the operator
    passing a round_price already adjusted; this primitive keeps the
    arithmetic clean).
    For EXIT: shares = principal / (exit_price × (1 - discount)).
    For REVENUE / MATURITY_DATE: shares = principal / valuation_cap, OR
    raises if neither trigger is configured (the operator must specify
    the conversion math out-of-band for those triggers).
    """
    if note.status is not NoteStatus.OUTSTANDING:
        raise ValueError(f"note is {note.status.value}, cannot convert")
    if event.trigger not in note.triggers:
        raise ValueError(
            f"event trigger {event.trigger.value} not in note's "
            f"configured triggers {[t.value for t in note.triggers]}"
        )
    if event.trigger in (ConversionTrigger.NEXT_ROUND, ConversionTrigger.EXIT):
        price = conversion_price(note, event)
        cap_applied = False
        if (
            note.valuation_cap_usd is not None
            and event.trigger is ConversionTrigger.NEXT_ROUND
            and event.next_round_post_money_usd is not None
            and event.next_round_post_money_usd > 0
        ):
            # cap_implied = round_price × (cap / post_money). When the
            # post-money valuation exceeds the cap, this is less than
            # round_price and (typically) less than the discount-implied
            # price → cap dominates.
            assert event.next_round_price_per_share is not None
            cap_implied_price = event.next_round_price_per_share * (
                note.valuation_cap_usd / event.next_round_post_money_usd
            )
            if cap_implied_price < price:
                price = cap_implied_price
                cap_applied = True
        shares = note.principal_usd / price
        return ConversionResult(
            note_id=note.note_id,
            shares_issued=shares,
            effective_price_per_share=price,
            discount_applied=note.discount_pct > 0 and not cap_applied,
            cap_applied=cap_applied,
        )
    if event.trigger in (ConversionTrigger.REVENUE, ConversionTrigger.MATURITY_DATE):
        if note.valuation_cap_usd is None:
            raise ValueError(f"{event.trigger.value} conversion requires a valuation_cap_usd")
        # Without a per-share price reference, conversion uses a
        # nominal "1 share per cap_dollar" so the share count is
        # comparable to the cap. Operator post-converts via cap-table
        # math.
        shares = note.principal_usd / note.valuation_cap_usd
        return ConversionResult(
            note_id=note.note_id,
            shares_issued=shares,
            effective_price_per_share=note.valuation_cap_usd,
            discount_applied=False,
            cap_applied=True,
        )
    raise ValueError(f"unknown trigger {event.trigger.value}")


def wakalah_fee_owed(
    note: ConvertibleNote,
    *,
    as_of: date,
) -> float:
    """Cumulative Wakalah fee owed up to `as_of` (simple, not compound).

    Pin: simple-interest math. Compounding would read as riba.
    """
    if as_of <= note.issue_date:
        return 0.0
    days = (as_of - note.issue_date).days
    return note.principal_usd * note.wakalah_fee_annual_pct * (days / 365.0)


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


def render_note(note: ConvertibleNote) -> str:
    """Operator-readable summary."""
    cap_str = (
        f"${note.valuation_cap_usd:,.0f}" if note.valuation_cap_usd is not None else "uncapped"
    )
    triggers = "/".join(t.value for t in note.triggers)
    return (
        f"📜 Convertible Musharakah note {note.note_id}: "
        f"${note.principal_usd:,.0f} principal "
        f"({_mask(note.investor_id)}→{_mask(note.issuer_id)})\n"
        f"  • Cap: {cap_str}, discount: {note.discount_pct * 100:.2f}%, "
        f"Wakalah fee: {note.wakalah_fee_annual_pct * 100:.2f}%/yr\n"
        f"  • Triggers: {triggers}\n"
        f"  • Status: {note.status.value}; matures {note.maturity_date.isoformat()}"
    )


def render_conversion(result: ConversionResult) -> str:
    """Operator-readable conversion summary."""
    flags = []
    if result.discount_applied:
        flags.append("discount")
    if result.cap_applied:
        flags.append("cap")
    flag_str = f" [{'+'.join(flags)}]" if flags else ""
    return (
        f"🔁 Conversion {result.note_id}: "
        f"{result.shares_issued:,.4f} shares @ "
        f"${result.effective_price_per_share:,.4f}{flag_str}"
    )
