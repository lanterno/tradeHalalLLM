"""Periodic purification disbursement scheduling and receipts.

Round-4 wave 2.D: builds on the existing per-trade /
per-dividend `PurificationEntry` ledger by adding the periodic
disbursement layer — monthly, quarterly, or yearly bundles that the
operator settles in one transaction to a configured charity.

Two responsibilities:

* **Period grouping.** Given a list of `PurificationEntry` rows,
  bucket them into UTC calendar periods (`monthly` / `quarterly` /
  `yearly`) on each entry's `received_at`. The bucket key is a
  human-readable label (`"2026-Q1"`, `"2026-03"`, `"2026"`) that
  doubles as the receipt's identifier.
* **Receipt rendering.** Per-period, produce a `DisbursementReceipt`
  with the totals, the per-symbol breakdown, and a markdown body
  suitable for emailing the operator and (later) signing into a
  PDF for the charity. Rendering is text-only here; PDF / signed
  attachment lives behind the email pipeline in a follow-up.

Halal alignment: the scheduler reports *what's due*, not what's
been paid. Marking entries paid is the operator's explicit
acknowledgement of the disbursement (records `paid_at` on the
underlying entries). Pin: the scheduler MUST NOT auto-mark
entries as paid — that's a one-way audit-trail commitment that
needs human consent.

Pure-Python; no DB, no network. The caller fetches entries from
the ledger, hands them in, gets back receipts.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Iterable

from halal_trader.halal.purification import PurificationEntry

# ── Period vocabulary ─────────────────────────────────────


class Period(str, Enum):
    """Disbursement cadence the operator picks."""

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


def _period_label(received_at: datetime, period: Period) -> str:
    """Map a UTC instant to its bucket label.

    Pin the formats so the receipt identifier is stable across
    machines and locales — a `2026-Q1` slug must mean exactly Jan-
    Mar 2026 UTC every time.
    """
    if received_at.tzinfo is None:
        # Treat naive timestamps as UTC — every other layer in the
        # bot stores tz-aware UTC, so a naive value is a bug
        # downstream; we coerce here rather than crash because the
        # scheduler must be tolerant of legacy ledger rows.
        received_at = received_at.replace(tzinfo=UTC)
    received_at = received_at.astimezone(UTC)
    if period == Period.MONTHLY:
        return f"{received_at.year:04d}-{received_at.month:02d}"
    if period == Period.QUARTERLY:
        quarter = (received_at.month - 1) // 3 + 1
        return f"{received_at.year:04d}-Q{quarter}"
    if period == Period.YEARLY:
        return f"{received_at.year:04d}"
    raise ValueError(f"unknown period {period!r}")


# ── Receipt dataclass ─────────────────────────────────────


@dataclass(frozen=True)
class SymbolBreakdown:
    """How much purification is owed against a single symbol in a
    given period. Useful for the receipt body and for spotting
    concentration ("80% of this quarter's obligation is from one
    holding")."""

    symbol: str
    entry_count: int
    purification_usd: Decimal


@dataclass(frozen=True)
class DisbursementReceipt:
    """One period's worth of outstanding purification.

    ``label`` is the period slug (`"2026-Q1"` etc.) used as both
    the human heading and a stable filename when later persisted
    as PDF.

    ``entries`` is the immutable list of underlying purification
    entries — caller can serialise the lot for the audit trail.
    Marking them paid is an explicit follow-up step, not part of
    rendering.
    """

    label: str
    period: Period
    starts_at: datetime
    ends_at: datetime
    total_usd: Decimal
    entry_count: int
    breakdown: list[SymbolBreakdown]
    entries: list[PurificationEntry]
    markdown: str = ""

    @property
    def is_empty(self) -> bool:
        return self.total_usd <= Decimal("0")


def _period_bounds(label: str, period: Period) -> tuple[datetime, datetime]:
    """Inverse of `_period_label` — given a slug, recover the UTC
    [start, end) bounds. Used to fill the receipt's date range
    for human-readable rendering.
    """
    if period == Period.MONTHLY:
        year, month = int(label[0:4]), int(label[5:7])
        starts = datetime(year, month, 1, tzinfo=UTC)
        if month == 12:
            ends = datetime(year + 1, 1, 1, tzinfo=UTC)
        else:
            ends = datetime(year, month + 1, 1, tzinfo=UTC)
        return starts, ends
    if period == Period.QUARTERLY:
        year, q = int(label[0:4]), int(label[6])
        start_month = (q - 1) * 3 + 1
        starts = datetime(year, start_month, 1, tzinfo=UTC)
        end_month = start_month + 3
        if end_month > 12:
            ends = datetime(year + 1, 1, 1, tzinfo=UTC)
        else:
            ends = datetime(year, end_month, 1, tzinfo=UTC)
        return starts, ends
    if period == Period.YEARLY:
        year = int(label)
        return (
            datetime(year, 1, 1, tzinfo=UTC),
            datetime(year + 1, 1, 1, tzinfo=UTC),
        )
    raise ValueError(f"unknown period {period!r}")


def _render_markdown(receipt: DisbursementReceipt, charity: str | None) -> str:
    """Build the operator-facing receipt body."""
    lines = [
        f"# Purification disbursement · {receipt.label}",
        "",
        (f"**Period:** {receipt.starts_at:%Y-%m-%d} → {receipt.ends_at:%Y-%m-%d} (UTC)"),
        f"**Cadence:** {receipt.period.value}",
        f"**Total due:** ${receipt.total_usd:,.2f} across {receipt.entry_count} entries",
    ]
    if charity:
        lines.append(f"**Disbursement target:** {charity}")
    lines.append("")
    if receipt.is_empty:
        lines.append("_No purification is owed for this period._")
        return "\n".join(lines)

    lines.append("## Breakdown by symbol")
    lines.append("")
    lines.append("| Symbol | Entries | Amount (USD) |")
    lines.append("| --- | --- | --- |")
    for b in receipt.breakdown:
        lines.append(f"| {b.symbol} | {b.entry_count} | ${b.purification_usd:,.2f} |")
    lines.append("")
    lines.append("## Underlying entries")
    lines.append("")
    for e in receipt.entries:
        lines.append(
            f"- {e.received_at:%Y-%m-%d} · `{e.symbol}` · "
            f"div ${e.dividend_usd:,.2f} × {e.haram_pct:.2%} = "
            f"**${e.purification_usd:,.2f}**"
        )
    lines.append("")
    lines.append(
        "Mark each entry paid in the ledger after disbursement — the scheduler does not auto-mark."
    )
    return "\n".join(lines)


# ── Scheduling ────────────────────────────────────────────


def schedule_disbursements(
    entries: Iterable[PurificationEntry],
    *,
    period: Period = Period.QUARTERLY,
    include_paid: bool = False,
    charity: str | None = None,
) -> list[DisbursementReceipt]:
    """Group entries into per-period receipts.

    ``include_paid`` is False by default: only the *outstanding*
    obligation is rendered, since that's what the operator needs to
    settle. Pass True for a backfill / audit run that wants the
    full historical picture.

    ``charity`` is purely informational — when supplied, it appears
    on each receipt's "Disbursement target" line so a future PDF /
    email pipeline can pre-fill it.

    Receipts are returned in chronological label order so the
    dashboard / CLI can render the oldest-first list without sorting.
    """
    buckets: dict[str, list[PurificationEntry]] = defaultdict(list)
    for entry in entries:
        if not include_paid and not entry.is_outstanding:
            continue
        label = _period_label(entry.received_at, period)
        buckets[label].append(entry)

    receipts: list[DisbursementReceipt] = []
    for label in sorted(buckets):
        bucket_entries = buckets[label]
        # Stable per-symbol aggregation in input-order of first
        # appearance — matches what the operator sees in the ledger.
        per_symbol: dict[str, list[PurificationEntry]] = defaultdict(list)
        for entry in bucket_entries:
            per_symbol[entry.symbol].append(entry)
        breakdown = [
            SymbolBreakdown(
                symbol=symbol,
                entry_count=len(rows),
                purification_usd=sum((r.purification_usd for r in rows), Decimal("0")),
            )
            for symbol, rows in per_symbol.items()
        ]
        # Sort breakdown by descending USD so the operator's eye
        # lands on the concentration first.
        breakdown.sort(key=lambda b: b.purification_usd, reverse=True)
        total = sum((e.purification_usd for e in bucket_entries), Decimal("0"))
        starts, ends = _period_bounds(label, period)
        receipt = DisbursementReceipt(
            label=label,
            period=period,
            starts_at=starts,
            ends_at=ends,
            total_usd=total,
            entry_count=len(bucket_entries),
            breakdown=breakdown,
            entries=list(bucket_entries),
        )
        markdown = _render_markdown(receipt, charity)
        # Re-construct with the rendered markdown — frozen
        # dataclass forbids mutation.
        receipt = DisbursementReceipt(
            label=receipt.label,
            period=receipt.period,
            starts_at=receipt.starts_at,
            ends_at=receipt.ends_at,
            total_usd=receipt.total_usd,
            entry_count=receipt.entry_count,
            breakdown=receipt.breakdown,
            entries=receipt.entries,
            markdown=markdown,
        )
        receipts.append(receipt)
    return receipts


def upcoming_due(
    entries: Iterable[PurificationEntry],
    *,
    now: datetime | None = None,
    period: Period = Period.QUARTERLY,
) -> DisbursementReceipt | None:
    """Convenience: return the receipt for the *current* period
    that's still building up obligations.

    Useful for a dashboard tile that says "this quarter so far you
    owe $X across N holdings". Returns None when no entries fall
    in the active period.
    """
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    target_label = _period_label(now, period)
    receipts = schedule_disbursements(entries, period=period)
    for r in receipts:
        if r.label == target_label:
            return r
    return None


__all__ = [
    "DisbursementReceipt",
    "Period",
    "SymbolBreakdown",
    "schedule_disbursements",
    "upcoming_due",
]
